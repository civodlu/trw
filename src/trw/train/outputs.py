import torch
import functools
import collections
import torch.nn as nn
from trw.train import utils
from trw.train import metrics
from trw.train.sequence_array import sample_uid_name as default_sample_uid_name
from trw.train import losses


class Output:
    """
    This is a tag name to find the output reference back from `outputs`
    """
    output_ref_tag = 'output_ref'

    def __init__(self, output, criterion_fn, collect_output=False, sample_uid_name=None):
        """
        :param output: a `torch.Tensor` to be recorded
        :param criterion_fn: the criterion function to be used to evaluate the output
        :param collect_output:
        :pram sample_uid_name: collect sample UID along with the output
        """
        self.output = output
        self.criterion_fn = criterion_fn
        self.collect_output = collect_output
        self.sample_uid_name = sample_uid_name

    def evaluate_batch(self, batch, is_training):
        """
        Evaluate a batch of data and extract important outputs
        :param batch: the batch of data
        :param is_training: if True, this was a training batch
        :return: a dictionary
        """
        assert 0, 'this needs to be implemented in derived classes!'
        
    def loss_term_cleanup(self, loss_term):
        """
        This function is called for each batch just before switching to another batch.

        It can be used to clean up large arrays stored
        """
        pass
        
    def extract_history(self, outputs):
        """
        Summarizes epoch statistics from the calculated outputs to populate an history
        :param outputs: the aggregated `evaluate_batch` output
        :return: a dictionary
        """
        return None


class OutputEmbedding(Output):
    """
    Represent an embedding

    This is only used to record a tensor that we consider an embedding (e.g., to be exported to tensorboard)
    """
    def __init__(self, output, clean_loss_term_each_batch=False):
        """
        
        Args:
            output: the output from which the embedding will be created
            clean_loss_term_each_batch: if True, the loss term output will be removed from the output in
                order to free memory just before the next batch. For example, if we want to collect statistics
                on the embedding, we do not need to keep track of the output embedding and in particular for
                large embeddings
        """
        super().__init__(output=output, criterion_fn=None, collect_output=True)
        self.clean_loss_term_each_batch = clean_loss_term_each_batch

    def evaluate_batch(self, batch, is_training):
        loss_term = collections.OrderedDict()

        loss_term['output'] = utils.to_value(self.output)
        loss_term[Output.output_ref_tag] = self  # keep a back reference
        return loss_term
    
    def loss_term_cleanup(self, loss_term):
        if self.clean_loss_term_each_batch:
            del loss_term['output']
            del self.output
            self.output = None
            
            
def segmentation_criteria_ce_dice(output, truth, ce_weight=0.5):
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')(output, truth)
    dice_loss = 0
    loss = ce_weight * cross_entropy_loss + (1 - ce_weight) * dice_loss
    return loss
    

class OutputSegmentation(Output):
    """
    Segmentation output
    """
    def __init__(
            self,
            output,
            target_name,
            criterion_fn=losses.DiceLoss,
            collect_only_non_training_output=False,
            metrics=metrics.default_segmentation_metrics(),
            loss_reduction=torch.mean,
            weight_name=None,
            loss_scaling=1.0,
            output_postprocessing=functools.partial(torch.argmax, dim=-1)):
        """

        :param output:
        :param target_name:
        :param criterion_fn:
        :param collect_only_non_training_output:
        :param metrics:
        :param loss_reduction:
        :param weight_name: if not None, the weight name. the loss of each sample will be weighted by this vector
        :param loss_scaling: scale the loss by a scalar
        :param output_postprocessing:
        """
        super().__init__(output=output, criterion_fn=criterion_fn, collect_output=False)
        self.target_name = target_name
        self.loss_reduction = loss_reduction
        self.output_postprocessing = output_postprocessing
        self.collect_only_non_training_output = collect_only_non_training_output
        self.metrics = metrics
        self.weight_name = weight_name
        self.loss_scaling = loss_scaling

    def extract_history(self, outputs):
        history = collections.OrderedDict()
        for metric in self.metrics:
            r = metric(outputs)
            if r is not None:
                metric_name, metric_value = r
                history[metric_name] = metric_value
        return history

    def evaluate_batch(self, batch, is_training):
        truth = batch.get(self.target_name)
        assert truth is not None, 'classes `{}` is missing in current batch!'.format(self.target_name)

        loss_term = {}
        losses = self.criterion_fn()(self.output, truth)
        assert isinstance(losses, torch.Tensor), 'must `loss` be a `torch.Tensor`'
        assert utils.len_batch(batch) == losses.shape[0], 'loss must have 1 element per sample'

        # do NOT keep the original output else memory will be an issue
        del self.output
        self.output = None

        if self.weight_name is not None:
            weights = batch.get(self.weight_name)
            assert weights is not None, 'weight `` could not be found!'.format(self.weight_name)
            assert len(weights) == len(losses), 'must have a weight per sample'
            assert len(weights.shape) == 1, 'must be a 1D vector'

            # expand to same shape size so that we can easily broadcast the weight
            weights = weights.reshape([weights.shape[0]] + [1] * (len(losses.shape) - 1))
            
        else:
            weights = torch.ones_like(losses)

        # weight the loss of each sample by the corresponding weight
        weighted_losses = weights * losses

        loss_term['losses'] = weighted_losses.data
        loss_term['loss'] = self.loss_scaling * self.loss_reduction(weighted_losses)  # here we MUST be able to calculate the gradient so don't detach
        loss_term[Output.output_ref_tag] = self  # keep a back reference
        return loss_term


class OutputClassification(Output):
    """
    Classification output
    """
    def __init__(
            self,
            output,
            classes_name,
            criterion_fn=lambda: nn.CrossEntropyLoss(reduction='none'),
            collect_output=True,
            collect_only_non_training_output=False,
            metrics=metrics.default_classification_metrics(),
            loss_reduction=torch.mean,
            weight_name=None,
            loss_scaling=1.0,
            output_postprocessing=functools.partial(torch.argmax, dim=-1),
            maybe_optional=False,
            sample_uid_name=default_sample_uid_name):
        """
        
        Args:
            output:
            classes_name:
            criterion_fn:
            collect_output:
            collect_only_non_training_output:
            metrics:
            loss_reduction:
            weight_name: if not None, the weight name. the loss of each sample will be weighted by this vector
            loss_scaling: scale the loss by a scalar
            output_postprocessing:
            maybe_optional: if True, the loss term may be considered optional if the ground truth is not part of the batch
            sample_uid_name (str): if not None, collect the sample UID
        """
        super().__init__(output=output, criterion_fn=criterion_fn, collect_output=collect_output, sample_uid_name=sample_uid_name)
        self.classes_name = classes_name
        self.loss_reduction = loss_reduction
        self.output_postprocessing = output_postprocessing
        self.collect_only_non_training_output = collect_only_non_training_output
        self.metrics = metrics
        self.loss_scaling = loss_scaling
        self.weight_name = weight_name
        self.maybe_optional = maybe_optional

    def extract_history(self, outputs):
        history = collections.OrderedDict()
        for metric in self.metrics:
            r = metric(outputs)
            if r is not None:
                metric_name, metric_value = r
                history[metric_name] = metric_value
        return history

    def evaluate_batch(self, batch, is_training):
        truth = batch.get(self.classes_name)
        if truth is None and self.maybe_optional:
            return None
        assert truth is not None, 'classes `{}` is missing in current batch. `maybe_optional` is False'.format(self.classes_name)
        assert len(truth) == len(self.output), 'expected len output == len truth! found=({}, {})'.format(len(self.output), len(truth))
        assert isinstance(truth, torch.Tensor), 'feature={} must be a torch.Tensor!'.format(self.classes_name)
        assert truth.dtype == torch.long, 'the truth vector must be a `long` type feature={}'.format(self.classes_name)
        
        # if exception: THCudaCheck FAIL error=59 : device-side assert triggered, it could be something
        # due to truth not within the expected bound
        #truth_np = truth.cpu().data.numpy()
        #import numpy as np
        #print(np.max(truth_np), np.min(truth_np))

        loss_term = {}
        losses = self.criterion_fn()(self.output, truth)
        assert isinstance(losses, torch.Tensor), 'must `loss` be a `torch.Tensor`'
        assert len(losses.shape) == 1, 'loss must be a 1D Tensor'
        assert utils.len_batch(batch) == losses.shape[0], 'loos must have 1 element per sample'
        if self.collect_output:
            # we may not want to collect any outputs or training outputs to save some time
            if not self.collect_only_non_training_output or not is_training:
                # detach the output so as not to calculate gradients. Keep the truth so that we
                # can calculate statistics (e.g., accuracy, FP/FN...)
                loss_term['output_raw'] = utils.to_value(self.output)
                loss_term['output'] = utils.to_value(self.output_postprocessing(self.output.data))
                loss_term['output_truth'] = utils.to_value(truth)

        if self.sample_uid_name is not None and self.sample_uid_name in batch:
            loss_term['uid'] = utils.to_value(batch[self.sample_uid_name])

        # do NOT keep the original output else memory will be an issue
        del self.output
        self.output = None

        if self.weight_name is not None:
            weights = batch.get(self.weight_name)
            assert weights is not None, 'weight `` could not be found!'.format(self.weight_name)
            assert len(weights) == len(losses), 'must have a weight per sample'
            assert len(weights.shape) == 1, 'must be a 1D vector'
        else:
            weights = torch.ones_like(losses)

        # weight the loss of each sample by the corresponding weight
        weighted_losses = weights * losses

        # TODO label smoothing
        loss_term['losses'] = weighted_losses.data
        loss_term['loss'] = self.loss_scaling * self.loss_reduction(weighted_losses)  # here we MUST be able to calculate the gradient so don't detach
        loss_term[Output.output_ref_tag] = self  # keep a back reference
        return loss_term


def mean_all(x):
    """
    :param x: a Tensor
    :return: the mean of all values
    """
    return torch.mean(x.view((-1)))


class OutputRegression(Output):
    """
    Regression output
    """
    def __init__(
            self,
            output,
            target_name,
            criterion_fn=lambda: nn.MSELoss(reduction='none'),
            collect_output=True,
            collect_only_non_training_output=False,
            metrics=metrics.default_regression_metrics(),
            loss_reduction=mean_all,
            weight_name=None,
            loss_scaling=1.0,
            output_postprocessing=lambda x: x):
        """

        :param output:
        :param target_name:
        :param criterion_fn:
        :param collect_output:
        :param collect_only_non_training_output:
        :param metrics:
        :param loss_reduction:
        :param weight_name: if not None, the weight name. the loss of each sample will be weighted by this vector
        :param loss_scaling: scale the loss by a scalar
        :param output_postprocessing:
        """
        super().__init__(output=output, criterion_fn=criterion_fn, collect_output=collect_output)
        self.target_name = target_name
        self.loss_reduction = loss_reduction
        self.output_postprocessing = output_postprocessing
        self.collect_only_non_training_output = collect_only_non_training_output
        self.metrics = metrics
        self.weight_name = weight_name
        self.loss_scaling = loss_scaling

    def extract_history(self, outputs):
        history = collections.OrderedDict()
        for metric in self.metrics:
            r = metric(outputs)
            if r is not None:
                metric_name, metric_value = r
                history[metric_name] = metric_value
        return history

    def evaluate_batch(self, batch, is_training):
        truth = batch.get(self.target_name)
        assert truth is not None, 'classes `{}` is missing in current batch!'.format(self.target_name)

        loss_term = {}
        losses = self.criterion_fn()(self.output, truth)
        assert isinstance(losses, torch.Tensor), 'must `loss` be a `torch.Tensor`'
        assert utils.len_batch(batch) == losses.shape[0], 'loos must have 1 element per sample'
        if self.collect_output:
            # we may not want to collect any outputs or training outputs to save some time
            if not self.collect_only_non_training_output or not is_training:
                # detach the output so as not to calculate gradients. Keep the truth so that we
                # can calculate statistics (e.g., accuracy, FP/FN...)
                loss_term['output_raw'] = utils.to_value(self.output)
                loss_term['output'] = utils.to_value(self.output_postprocessing(self.output.data))
                loss_term['output_truth'] = utils.to_value(truth)

        # do NOT keep the original output else memory will be an issue
        del self.output
        self.output = None

        if self.weight_name is not None:
            weights = batch.get(self.weight_name)
            assert weights is not None, 'weight `` could not be found!'.format(self.weight_name)
            assert len(weights) == len(losses), 'must have a weight per sample'
            assert len(weights.shape) == 1, 'must be a 1D vector'
        else:
            weights = torch.ones_like(losses)

        # weight the loss of each sample by the corresponding weight
        weighted_losses = weights * losses

        loss_term['losses'] = weighted_losses.data
        loss_term['loss'] = self.loss_scaling * self.loss_reduction(weighted_losses)  # here we MUST be able to calculate the gradient so don't detach
        loss_term[Output.output_ref_tag] = self  # keep a back reference
        return loss_term
    

class OutputRecord(Output):
    """
    Record the raw value, but do not compute any loss from it.

    This is useful, e.g., to collect UIDs so that we can save them in the network result and
    further post-process it (e.g., k-fold cross validation)
    """
    
    def __init__(
            self,
            output):
        """

        Args:
            output: the output value to record. May be of any type.
        """
        super().__init__(output=output, criterion_fn=None, collect_output=True)
    
    def evaluate_batch(self, batch, is_training):
        nb_samples = utils.len_batch(batch)
        
        assert len(self.output) == nb_samples, 'one output for each sample is required'
    
        loss_term = {
            'output': utils.to_value(self.output),
            'losses': torch.zeros([nb_samples], dtype=torch.float32),
            'loss': 0.0,
            Output.output_ref_tag: self,
        }
        
        # do NOT keep the original output else memory will be an issue
        del self.output
        self.output = None
    
        return loss_term
