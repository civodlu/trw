import torch
import functools
import collections
import torch.nn as nn
from trw.train import metrics
from trw.train.sequence_array import sample_uid_name as default_sample_uid_name
from trw.train import losses
from trw.train import utilities


def dict_torch_values_to_numpy(d):
    """
    Transform all torch.Tensor to numpy arrays of a dictionary like object
    """
    if d is None:
        return
    assert isinstance(d, collections.Mapping), 'must be a dict like object'

    for name, value in d.items():
        if isinstance(value, torch.Tensor):
            d[name] = utilities.to_value(value)




class Output:
    """
    This is a tag name to find the output reference back from `outputs`
    """
    output_ref_tag = 'output_ref'

    def __init__(self, metrics, output, criterion_fn, collect_output=False, sample_uid_name=None):
        """
        :param metrics: the metrics to be reported for each output
        :param output: a `torch.Tensor` to be recorded
        :param criterion_fn: the criterion function to be used to evaluate the output
        :param collect_output:
        :pram sample_uid_name: collect sample UID along with the output
        """
        self.output = output
        self.criterion_fn = criterion_fn
        self.collect_output = collect_output
        self.metrics = metrics

        # this can be used to collect the UIDs of the sample the output was calculated from.
        # this can be particularly useful for various tasks: track data augmentation,
        self.sample_uid_name = sample_uid_name

    def evaluate_batch(self, batch, is_training):
        """
        Evaluate a batch of data and extract important outputs
        :param batch: the batch of data
        :param is_training: if True, this was a training batch
        :return: tuple(a dictionary of values, dictionary of metrics)
        """
        assert 0, 'this needs to be implemented in derived classes!'
        
    def loss_term_cleanup(self, loss_term):
        """
        This function is called for each batch just before switching to another batch.

        It can be used to clean up large arrays stored or release CUDA memory
        """
        dict_torch_values_to_numpy(loss_term)
        metrics_results = loss_term.get('metrics_results')
        if metrics_results is not None:
            dict_torch_values_to_numpy(metrics_results)



def extract_metrics(metrics_outputs, outputs):
    """
    Extract metrics from an output

    Args:
        metrics: a list of metrics
        outputs: the result of `Output.evaluate_batch`

    Returns:
        a dictionary of key, value
    """
    history = collections.OrderedDict()
    for metric in metrics_outputs:
        r = metric(outputs)
        if r is not None:
            assert isinstance(r, collections.Mapping), 'must be a dict like structure'
            history.update(r)
    return history


class OutputEmbedding(Output):
    """
    Represent an embedding

    This is only used to record a tensor that we consider an embedding (e.g., to be exported to tensorboard)
    """
    def __init__(self, output, clean_loss_term_each_batch=False, sample_uid_name=default_sample_uid_name):
        """
        
        Args:
            output: the output from which the embedding will be created
            clean_loss_term_each_batch: if True, the loss term output will be removed from the output in
                order to free memory just before the next batch. For example, if we want to collect statistics
                on the embedding, we do not need to keep track of the output embedding and in particular for
                large embeddings
            sample_uid_name: UID name to be used for collecting the embedding of the samples
        """
        super().__init__(
            output=output,
            criterion_fn=None,
            collect_output=True,
            sample_uid_name=sample_uid_name,
            metrics=None)
        self.clean_loss_term_each_batch = clean_loss_term_each_batch

    def evaluate_batch(self, batch, is_training):
        loss_term = collections.OrderedDict()

        loss_term['output'] = utilities.to_value(self.output)
        loss_term[Output.output_ref_tag] = self  # keep a back reference
        return loss_term
    
    def loss_term_cleanup(self, loss_term):
        if self.clean_loss_term_each_batch:
            del loss_term['output']
            del self.output
            self.output = None


def segmentation_criteria_ce_dice(output, truth, ce_weight=0.5):
    """
    loss combining cross entropy and multiclass dice

    Args:
        output: the output value
        truth: the truth
        ce_weight: the weight of the cross entropy to use. This controls the importance of the
            cross entropy loss to the overall segmentation loss

    Returns:
        a torch tensor
    """
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')(output, truth)
    cross_entropy_loss = cross_entropy_loss.mean(tuple(range(1, len(cross_entropy_loss.shape))))
    
    dice_loss = losses.LossDiceMulticlass()(output, truth)
    loss = ce_weight * cross_entropy_loss + (1 - ce_weight) * dice_loss
    return loss


def segmentation_output_postprocessing(mask_pb):
    """
    Post-process the mask probability of the segmentation into discrete segmentation map
    """
    return torch.unsqueeze(torch.argmax(mask_pb))


class OutputSegmentation(Output):
    """
    Segmentation output
    """
    def __init__(
            self,
            output,
            target_name,
            criterion_fn=lambda: segmentation_criteria_ce_dice,
            collect_only_non_training_output=True,
            metrics=metrics.default_segmentation_metrics(),
            loss_reduction=torch.mean,
            weight_name=None,
            loss_scaling=1.0,
            output_postprocessing=functools.partial(torch.argmax, dim=1),
            sample_uid_name=default_sample_uid_name):
        """

        :param output:
        :param target_name:
        :param criterion_fn:
        :param metrics:
        :param loss_reduction:
        :param weight_name: if not None, the weight name. the loss of each sample will be weighted by this vector
        :param loss_scaling: scale the loss by a scalar
        :param output_postprocessing:
        :param collect_only_non_training_output: if True, only non-training output will be collected
        """
        super().__init__(
            output=output,
            criterion_fn=criterion_fn,
            collect_output=False,
            sample_uid_name=sample_uid_name,
            metrics=metrics)

        self.target_name = target_name
        self.loss_reduction = loss_reduction
        self.output_postprocessing = output_postprocessing
        self.weight_name = weight_name
        self.loss_scaling = loss_scaling
        self.collect_only_non_training_output = collect_only_non_training_output

    def evaluate_batch(self, batch, is_training):
        truth = batch.get(self.target_name)
        assert truth is not None, 'classes `{}` is missing in current batch!'.format(self.target_name)

        max_index = int(torch.max(truth).cpu().numpy())
        min_index = int(torch.min(truth).cpu().numpy())
        assert max_index < self.output.shape[1], f'index out of bound. Got={max_index}, maximum={self.output.shape[1]}. Make sure the input data is correct.'
        assert min_index >= 0, f'incorrect index! got={min_index}'

        loss_term = {}
        losses = self.criterion_fn()(self.output, truth)
        assert isinstance(losses, torch.Tensor), 'must `loss` be a `torch.Tensor`'
        assert utilities.len_batch(batch) == losses.shape[0], 'loss must have 1 element per sample'

        # keep these as torch variable since metrics may be slow to calculate with numpy (e.g., dice)
        if (is_training and not self.collect_only_non_training_output) or not is_training:
            loss_term['output_raw'] = self.output
            loss_term['output'] = self.output_postprocessing(self.output.data)
            loss_term['output_truth'] = truth

        # do NOT keep the original output else memory will be an issue
        # (e.g., CUDA device)
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
            
        if self.sample_uid_name is not None and self.sample_uid_name in batch:
            loss_term['uid'] = utilities.to_value(batch[self.sample_uid_name])

        # weight the loss of each sample by the corresponding weight
        weighted_losses = weights * losses

        loss_term['losses'] = weighted_losses.data
        loss_term['loss'] = self.loss_scaling * self.loss_reduction(weighted_losses)  # here we MUST be able to calculate the gradient so don't detach
        loss_term[Output.output_ref_tag] = self  # keep a back reference

        loss_term['metrics_results'] = extract_metrics(self.metrics, loss_term)
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
            output_postprocessing=functools.partial(torch.argmax, dim=1),  # we want to export the class (i.e., the channel component)
            maybe_optional=False,
            sample_uid_name=default_sample_uid_name):
        """
        
        Args:
            output: the raw output values
            classes_name: the name of the features to be used as target. Targets must be a 1D vector of integers
            criterion_fn: the criterion to minimize between the output and the target
            collect_output: if True, the output values will be collected (and possibly exported for debug purposes)
            collect_only_non_training_output: if True, only the non-training splits will have the outputs collected
            metrics: the metrics to be reported each epoch
            loss_reduction:
            weight_name: if not None, the weight name. the loss of each sample will be weighted by this vector
            loss_scaling: scale the loss by a scalar
            output_postprocessing: the output will be postprocessed by this function. For example,
                we could extract the final classification instead of the loggit
            maybe_optional: if True, the loss term may be considered optional if the ground truth is not part of the batch
            sample_uid_name (str): if not None, collect the sample UID
        """
        super().__init__(
            output=output,
            criterion_fn=criterion_fn,
            collect_output=collect_output,
            sample_uid_name=sample_uid_name,
            metrics=metrics)
        self.classes_name = classes_name
        self.loss_reduction = loss_reduction
        self.output_postprocessing = output_postprocessing
        self.collect_only_non_training_output = collect_only_non_training_output
        self.loss_scaling = loss_scaling
        self.weight_name = weight_name
        self.maybe_optional = maybe_optional

    def evaluate_batch(self, batch, is_training):
        truth = batch.get(self.classes_name)
        if truth is None and self.maybe_optional:
            return None
        assert truth is not None, 'classes `{}` is missing in current batch. `maybe_optional` is False'.format(self.classes_name)
        assert len(truth) == len(self.output), 'expected len output == len truth! found=({}, {})'.format(len(self.output), len(truth))
        assert isinstance(truth, torch.Tensor), 'feature={} must be a torch.Tensor!'.format(self.classes_name)
        assert truth.dtype == torch.long, 'the truth vector must be a `long` type feature={}'.format(self.classes_name)
        
        # make sure the class is not out of bound. This is a very common mistake!
        max_index = int(torch.max(truth).cpu().numpy())
        min_index = int(torch.min(truth).cpu().numpy())
        assert max_index < self.output.shape[1], f'index out of bound. Got={max_index}, maximum={self.output.shape[1]}. Make sure the input data is correct.'
        assert min_index >= 0, f'incorrect index! got={min_index}'

        loss_term = {}
        losses = self.criterion_fn()(self.output, truth)
        assert isinstance(losses, torch.Tensor), 'must `loss` be a `torch.Tensor`'
        assert len(losses.shape) == 1, 'loss must be a 1D Tensor'
        assert utilities.len_batch(batch) == losses.shape[0], 'loos must have 1 element per sample'
        if self.collect_output:
            # we may not want to collect any outputs or training outputs to save some time
            if not self.collect_only_non_training_output or not is_training:
                # detach the output so as not to calculate gradients. Keep the truth so that we
                # can calculate statistics (e.g., accuracy, FP/FN...)
                loss_term['output_raw'] = self.output
                loss_term['output'] = self.output_postprocessing(self.output.data)
                loss_term['output_truth'] = truth

        if self.sample_uid_name is not None and self.sample_uid_name in batch:
            loss_term['uid'] = utilities.to_value(batch[self.sample_uid_name])

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
        loss_term['metrics_results'] = extract_metrics(self.metrics, loss_term)
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
            output_postprocessing=lambda x: x,
            sample_uid_name=default_sample_uid_name):
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
        super().__init__(
            output=output,
            criterion_fn=criterion_fn,
            collect_output=collect_output,
            sample_uid_name=sample_uid_name,
            metrics=metrics)
        self.target_name = target_name
        self.loss_reduction = loss_reduction
        self.output_postprocessing = output_postprocessing
        self.collect_only_non_training_output = collect_only_non_training_output
        self.weight_name = weight_name
        self.loss_scaling = loss_scaling

    def evaluate_batch(self, batch, is_training):
        truth = batch.get(self.target_name)
        assert truth is not None, 'classes `{}` is missing in current batch!'.format(self.target_name)

        loss_term = {}
        losses = self.criterion_fn()(self.output, truth)
        assert isinstance(losses, torch.Tensor), 'must `loss` be a `torch.Tensor`'
        assert utilities.len_batch(batch) == losses.shape[0], 'loos must have 1 element per sample'
        if self.collect_output:
            # we may not want to collect any outputs or training outputs to save some time
            if not self.collect_only_non_training_output or not is_training:
                # detach the output so as not to calculate gradients. Keep the truth so that we
                # can calculate statistics (e.g., accuracy, FP/FN...)
                loss_term['output_raw'] = self.output
                loss_term['output'] = self.output_postprocessing(self.output.data)
                loss_term['output_truth'] = truth

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
            
        if self.sample_uid_name is not None and self.sample_uid_name in batch:
            loss_term['uid'] = utilities.to_value(batch[self.sample_uid_name])

        # weight the loss of each sample by the corresponding weight
        weighted_losses = weights * losses

        loss_term['losses'] = weighted_losses.data
        loss_term['loss'] = self.loss_scaling * self.loss_reduction(weighted_losses)  # here we MUST be able to calculate the gradient so don't detach
        loss_term[Output.output_ref_tag] = self  # keep a back reference
        loss_term['metrics_results'] = extract_metrics(self.metrics, loss_term)
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
        super().__init__(output=output, criterion_fn=None, collect_output=True, metrics=None)
    
    def evaluate_batch(self, batch, is_training):
        nb_samples = utilities.len_batch(batch)
        
        assert len(self.output) == nb_samples, 'one output for each sample is required'
    
        loss_term = {
            'output': utilities.to_value(self.output),
            'losses': torch.zeros([nb_samples], dtype=torch.float32),
            'loss': 0.0,
            Output.output_ref_tag: self,
        }
        
        # do NOT keep the original output else memory will be an issue
        del self.output
        self.output = None
    
        return loss_term
