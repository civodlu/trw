import torch
import torch.optim
import torch.nn
import collections
import functools
import logging
import numpy as np
import numbers
import os
import pickle
import time
import itertools
from trw.train import outputs
from trw.train import utils
from trw.train import callback_model_summary
from trw.train import callback_data_summary
from trw.train import callback_export_classification_errors
from trw.train import callback_epoch_summary
from trw.train import callback_export_classification_report
from trw.train import callback_export_history
from trw.train import callback_save_last_model
from trw.train import callback_tensorboard
from trw.train import callback_tensorboard_record_history
from trw.train import callback_tensorboard_embedding
from trw.train import callback_tensorboard_record_model
from trw.train import callback_export_samples
from trw.train import callback_export_augmentations
from trw.train import callback_export_best_history
from trw.train import callback_learning_rate_finder
from trw.train import callback_learning_rate_recorder
from trw.train import callback_explain_decision
from trw.train import callback_export_classification_by_epoch

logger = logging.getLogger(__name__)


def postprocess_batch(dataset_name, split_name, batch, callbacks_per_batch):
    """
    Post process a batch of data (e.g., this can be useful to add additional
    data to the current batch)

    Args:
        dataset_name: the name of the dataset the `batch` belongs to
        split_name: the name of the split the `batch` belongs to
        batch: the current batch of data
        callbacks_per_batch: the callbacks (a list) to be executed for each batch.
            Each callback must be callable with `(dataset_name, split_name, batch)`.
            if `None`, no callbacks
    """

    # always useful: for example if we do a model composed of multiple sub-models (one per dataset)
    # we would need to know what sub-model to use
    batch['dataset_name'] = dataset_name
    batch['split_name'] = split_name
    
    if callbacks_per_batch is not None:
        for callback in callbacks_per_batch:
            callback(dataset_name, split_name, batch)


def prepare_loss_terms(outputs, batch, is_training):
    """
    Return the loss_terms for the given outputs
    """
    loss_terms = collections.OrderedDict()
    for output_name, output in outputs.items():
        loss_term = output.evaluate_batch(batch, is_training)
        if loss_term is not None:
            loss_terms[output_name] = loss_term
    return loss_terms


def create_losses_fn(datasets, generic_loss):
    """
    Create a dictionary of loss functions for each of the dataset

    Args:
        datasets: the datasets
        generic_loss: a loss function

    Returns:
        A dictionary of losses for each of the dataset
    """
    losses_fn = collections.OrderedDict()
    for dataset_name in datasets.keys():
        losses_fn[dataset_name] = generic_loss
    return losses_fn


def create_scheduler_step_lr(optimizer, step_size=30, gamma=0.1):
    """
    Create a learning rate scheduler. Every `step_size`, the learning late will be multiplied by `gamma`

    Args:
        optimizer: the optimizer
        step_size: every number of epochs composing one step. Each step the learning rate will be decreased
        gamma: apply this factor to the learning rate every time it is adjusted

    Returns:
        a learning rate scheduler
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def create_optimizers_fn(datasets, model, optimizer_fn, scheduler_fn=None):
    """
    Create an optimizer and scheduler

    Note:
        if model is an instance of`ModuleDict`, then the optimizer will only consider the parameters
        `model[dataset_name].parameters()` else `model.parameters()`

    Args:
        datasets: a dictionary of dataset
        model: the model. Should be a `Module` or a `ModuleDict`
        optimizer_fn: the functor to instantiate the optimizer
        scheduler_fn: the functor to instantiate the scheduler. May be None, in that case
            there will be no scheduler

    Returns:
        a dict of optimizers, one per dataset
    """

    schedulers = None
    if scheduler_fn is not None:
        schedulers = collections.OrderedDict()
    optimizers = collections.OrderedDict()
    for dataset_name in datasets.keys():
        if isinstance(model, torch.nn.ModuleDict):
            # this is a collection of model. Assumed we have a different model
            # per dataset to be optimized
            sub_model = model[dataset_name]
            optimizer = optimizer_fn(sub_model.parameters())
        else:
            optimizer = optimizer_fn(model.parameters())
        
        optimizers[dataset_name] = optimizer

        if schedulers is not None:
            scheduler = scheduler_fn(optimizer)
            schedulers[dataset_name] = scheduler

    return optimizers, schedulers


def create_adam_optimizers_fn(datasets, model, learning_rate, scheduler_fn=None):
    """
    Create an ADAM optimizer for each of the dataset with optional scheduler

    Args:
        datasets: a dictionary of dataset
        model: a model to optimize
        learning_rate: the initial learning rate
        scheduler_fn: a scheduler, or `None`

    Returns:
        An optimizer
    """
    optimizer_fn = functools.partial(torch.optim.Adam, lr=learning_rate)
    return create_optimizers_fn(datasets, model, optimizer_fn, scheduler_fn)


def create_adam_optimizers_scheduler_step_lr_fn(datasets, model, learning_rate, step_size, gamma):
    """
    Create an ADAM optimizer for each of the dataset with optional scheduler

    Args:
        datasets: a dictionary of dataset
        model: a model to optimize
        learning_rate: the initial learning rate
        step_size: the number of epoch composing a step. Each step the learning rate will be multiplied by `gamma`
        gamma: the factor to apply to the learning rate every step

    Returns:
        An optimizer with a step scheduler
    """
    scheduler_fn = functools.partial(create_scheduler_step_lr, step_size=step_size, gamma=gamma)
    return create_adam_optimizers_fn(datasets, model, learning_rate=learning_rate, scheduler_fn=scheduler_fn)


def create_sgd_optimizers_fn(datasets, model, learning_rate, momentum=0.9, scheduler_fn=None):
    """
        Create a Stochastic gradient descent optimizer for each of the dataset with optional scheduler

        Args:
            datasets: a dictionary of dataset
            model: a model to optimize
            learning_rate: the initial learning rate
            scheduler_fn: a scheduler, or `None`
            momentum: the momentum of the SGD

        Returns:
            An optimizer
        """
    optimizer_fn = functools.partial(torch.optim.SGD, lr=learning_rate, momentum=momentum)
    return create_optimizers_fn(datasets, model, optimizer_fn, scheduler_fn)


def create_sgd_optimizers_scheduler_step_lr_fn(datasets, model, learning_rate, step_size, gamma):
    """
        Create a Stochastic gradient descent optimizer for each of the dataset with step learning rate scheduler

        Args:
            datasets: a dictionary of dataset
            model: a model to optimize
            learning_rate: the initial learning rate
            step_size: the number of epoch composing a step. Each step the learning rate will be multiplied by `gamma`
            gamma: the factor to apply to the learning rate every step

        Returns:
            An optimizer with a step scheduler
        """
    scheduler_fn = functools.partial(create_scheduler_step_lr, step_size=step_size, gamma=gamma)
    return create_sgd_optimizers_fn(datasets, model, learning_rate=learning_rate, scheduler_fn=scheduler_fn)


def generic_aggregate_loss_terms(loss_terms_history):
    """
    Aggregate the loss terms for all the steps of an epoch

    Args:
        loss_terms_history: a list of loss terms

    Returns:
        a tuple `output, history`. `output` is maintained alive only during the current epoch.
            `history` is kept in memory during the whole training
    """

    if loss_terms_history is None or len(loss_terms_history) == 0:
        return {}, []

    output_step = collections.OrderedDict()
    history_step = collections.OrderedDict()
    first = loss_terms_history[0]
    for loss_term_name in first.keys():
        aggregated = collections.OrderedDict()
        if loss_terms_history is not None and len(loss_terms_history) > 0:
            # initalization: create a list of 1 element
            first_term = loss_terms_history[0][loss_term_name]
            for name, value in first_term.items():
                aggregated[name] = [value]

            # aggregate all the loss terms apart from the first term
            for loss_terms in loss_terms_history[1:]:
                loss_term_of_interest = loss_terms.get(loss_term_name)
                assert loss_term_of_interest is not None, 'we must have the same loss terms for all the steps of a given iteration. E.g., `{}` is missing!'.format(loss_term_name)
                if loss_term_of_interest is not None:
                    for name, value in loss_term_of_interest.items():
                        aggregated[name].append(value)

            # finally, depending on the type, do aggregation
            aggregated_output = collections.OrderedDict()
            for name, values in aggregated.items():
                value = values[0]
                if isinstance(value, np.ndarray):
                    # concatenate tensors (e.g., softmax output)
                    aggregated_output[name] = np.concatenate(values)
                elif isinstance(value, numbers.Number):
                    # average numbers (e.g., losses)
                    aggregated_output[name] = np.sum(values) / len(values)
                elif isinstance(value, torch.Tensor):
                    if len(value.shape) > 0:
                        aggregated_output[name] = torch.cat(values)
                    else:
                        aggregated_output[name] = torch.sum(torch.stack(values)) / len(values)
                elif isinstance(value, outputs.Output):
                    aggregated_output[name] = values[0]
                elif isinstance(value, list):
                    aggregated_output[name] = list(itertools.chain.from_iterable(values))
                else:
                    assert 0, 'this type=`{}` for name={} is not handled!'.format(type(value), name)
            output_step[loss_term_name] = aggregated_output

            output_ref = aggregated_output.get(outputs.Output.output_ref_tag)
            if output_ref is not None:
                h = output_ref.extract_history(aggregated_output)
                if h is not None:
                    history_step[loss_term_name] = h

    return output_step, history_step


def loss_term_cleanup(loss_terms):
    """
    Perform cleanup on all the loss terms

    Requires ``outputs.Output.output_ref_tag`` tag for each loss term, else no cleanup will be done
    for this loss term.

    Args:
        loss_terms: the loss terms to be cleaned up
    """
    for name, loss_term in loss_terms.items():
        ref = loss_term.get(outputs.Output.output_ref_tag)
        if ref is not None:
            ref.loss_term_cleanup(loss_term)


@utils.time_it()
def train_loop(
        device,
        dataset_name,
        split_name,
        split,
        optimizer,
        model,
        loss_fn,
        history,
        callbacks_per_batch,
        callbacks_per_batch_loss_terms,
        apply_backward=True):
    """
    Run the train loop (i.e., the model parameters will be updated)

    Note:
        If `callbacks_per_batch` or `callbacks_per_batch_loss_terms` raise an exception
        `StopIteration`, the train loop will be stopped

    Args:
        device: the device to be used to optimize the model
        dataset_name: the name of the dataset
        split_name: the name of the split
        split: a dictionary of feature name and values
        optimizer: an optimizer to optimize the model
        model: the model to be optimized
        loss_fn: the loss function
        history: a list of history step
        callbacks_per_batch: the callbacks to be performed on each batch. if `None`, no callbacks to be run
        callbacks_per_batch_loss_terms: the callbacks to be performed on each loss term. if `None`, no callbacks to be run
        apply_backward: if True, the gradient will be back-propagated
    """
    #assert optimizer is not None, 'optimizer can\'t be None'

    # make sure the model is in training mode (e.g., batch norm, dropout)
    model.train()

    all_loss_terms = []
    
    total_batch_processing_time = 0.0
    batch_processing_last = time.perf_counter()
    loop_started = time.perf_counter()
    total_collate_and_postprocess = 0.0
    nb_samples = 0
    try:
        for i, batch in enumerate(split):
            # calculate the time for batch processing. In particular
            # this may be significant when using large data augmentations
            # and useful to optimize the data processing pipeline
            current_batch_processing = time.perf_counter() - batch_processing_last
            total_batch_processing_time += current_batch_processing

            total_collate_and_postprocess_start = time.perf_counter()
            batch = utils.transfer_batch_to_device(batch, device)
            #batch = utils.default_collate_fn(batch, device=device, non_blocking=True)
            
            postprocess_batch(dataset_name, split_name, batch, callbacks_per_batch)
            total_collate_and_postprocess_end = time.perf_counter()
            total_collate_and_postprocess += total_collate_and_postprocess_end - total_collate_and_postprocess_start
            
            optimizer.zero_grad()
    
            outputs = model(batch)
            
            assert isinstance(outputs, collections.Mapping), 'model must create a dict of outputs'
            loss_terms = prepare_loss_terms(outputs, batch, is_training=True)
    
            all_loss_terms.append(loss_terms)
            loss = loss_fn(dataset_name, batch, loss_terms)
            if optimizer is not None and apply_backward:
                # if there is no optimizer, it means we did not want to change the parameters
                loss.backward()
            loss_terms['overall_loss'] = {'loss': loss}
            optimizer.step()

            if callbacks_per_batch_loss_terms is not None:
                for callback in callbacks_per_batch_loss_terms:
                    callback(dataset_name, split_name, batch, loss_terms)

            loss_term_cleanup(loss_terms)
            batch_processing_last = time.perf_counter()
            nb_samples += utils.len_batch(batch)

    except StopIteration:
        pass
    loop_ended = time.perf_counter()
    
    logger.debug('nb_samples={}, train_loop total_batch_processing_time={}, loop_time={}, collate_and_postprocess={}, dataset_name={}, split_name={}'.format(
        nb_samples,
        total_batch_processing_time,
        loop_ended - loop_started,
        total_collate_and_postprocess,
        dataset_name,
        split_name))
    return all_loss_terms


def eval_loop(
        device,
        dataset_name,
        split_name,
        split,
        model,
        loss_fn,
        history,
        callbacks_per_batch=None,
        callbacks_per_batch_loss_terms=None):
    """
    Run the eval loop (i.e., the model parameters will NOT be updated)
    
    Note:
        If `callback_per_batch` or `callbacks_per_batch_loss_terms` raise `StopIteration`, the eval loop will be stopped
    :param device:
    :param dataset_name:
    :param split_name:
    :param split:
    :param model:
    :param loss_fn:
    :param history:
    :param callbacks_per_batch:
    :param callbacks_per_batch_loss_terms:
    :return:
    """
    all_loss_terms = []

    # make sure the model is in eval mode so that non essential operations are removed (e.g., batch norm, dropout)
    model.eval()
    
    try:
        for i, batch in enumerate(split):
            batch = utils.transfer_batch_to_device(batch, device=device)
            postprocess_batch(dataset_name, split_name, batch, callbacks_per_batch)
            with torch.no_grad():  # do not keep track of the gradient as we are just evaluating
                outputs = model(batch)
                loss_terms = prepare_loss_terms(outputs, batch, is_training=False)
                loss = loss_fn(dataset_name, batch, loss_terms)
                loss_terms['overall_loss'] = {'loss': loss}
                all_loss_terms.append(loss_terms)
                if callbacks_per_batch_loss_terms is not None:
                    for callback in callbacks_per_batch_loss_terms:
                        callback(dataset_name, split_name, batch, loss_terms)
                        
                # clean the loss terms (e.g., free memory)
                loss_term_cleanup(loss_terms)

    except StopIteration:
        pass
    return all_loss_terms


def epoch_train_eval(
        options,
        datasets,
        optimizers,
        model,
        losses,
        schedulers,
        history,
        callbacks_per_batch,
        callbacks_per_batch_loss_terms,
        run_eval,
        eval_loop_fn=eval_loop,
        train_loop_fn=train_loop):
    """
    Orchestrate the train and evaluation loops

    :param options:
    :param datasets:
    :param optimizers: if None, no optimization will be performed on the train split else a dictionary of
    optimizers (on for each dataset)
    :param model:
    :param losses:
    :param schedulers:
    :param history:
    :param callbacks_per_batch:
    :param callbacks_per_batch_loss_terms:
    :param run_eval: if True, run the evaluation
    :param eval_loop_fn: the eval function to be used
    :param train_loop_fn: the train function to be used
    :return:
    """
    device = options['workflow_options']['device']
    train_split_name = options['workflow_options']['train_split']
    history_by_dataset_epoch = collections.OrderedDict()
    outputs_by_dataset_epoch = collections.OrderedDict()
    for dataset_name, dataset in datasets.items():
        optimizer = None
        if optimizers is not None:
            optimizer = optimizers.get(dataset_name)
        loss_fn = losses[dataset_name]
        scheduler = None
        if schedulers is not None:
            scheduler = schedulers.get(dataset_name)

        dataset_history = collections.OrderedDict()
        dataset_outputs = collections.OrderedDict()
        for split_name, split in dataset.items():
            if optimizers is not None and split_name == train_split_name:
                # * only the split `train_split_name` is considered as training, all
                # other splits are for evaluation only
                # * if we don't have optimizers, we do NOT want to accumulate gradients
                # due to memory constraints, we must use the `torch.no_grad` context manager
                # so run it with the eval loop
                all_loss_terms = train_loop_fn(
                    device,
                    dataset_name,
                    split_name,
                    split,
                    optimizer,
                    model,
                    loss_fn,
                    history,
                    callbacks_per_batch=callbacks_per_batch,
                    callbacks_per_batch_loss_terms=callbacks_per_batch_loss_terms)
            else:
                if not run_eval or eval_loop_fn is None:
                    # we should not run the evaluation. Skip this!
                    continue

                all_loss_terms = eval_loop_fn(
                    device,
                    dataset_name,
                    split_name,
                    split,
                    model,
                    loss_fn,
                    history,
                    callbacks_per_batch=callbacks_per_batch,
                    callbacks_per_batch_loss_terms=callbacks_per_batch_loss_terms)

            assert isinstance(all_loss_terms, collections.Sequence), '`all_loss_terms` must be a sequence'

            epoch_outputs, epoch_history = generic_aggregate_loss_terms(all_loss_terms)
            dataset_history[split_name] = epoch_history
            dataset_outputs[split_name] = epoch_outputs

        history_by_dataset_epoch[dataset_name] = dataset_history
        outputs_by_dataset_epoch[dataset_name] = dataset_outputs

        if scheduler is not None:
            scheduler.step()

    return outputs_by_dataset_epoch, history_by_dataset_epoch


default_logger = utils.log_and_print


def default_pre_training_callbacks(logger=default_logger, with_lr_finder=False, with_export_augmentations=True):
    """
    Default callbacks to be performed before the fitting of the model
    """
    callbacks = [
        callback_tensorboard.CallbackClearTensorboardLog(),  # make sure the previous model train log is removed
        callback_model_summary.CallbackModelSummary(logger=logger),
        callback_data_summary.CallbackDataSummary(logger=logger),
        callback_export_samples.CallbackExportSamples(dirname='random_samples'),
    ]
    
    if with_export_augmentations:
        callbacks.append(callback_export_augmentations.CallbackExportAugmentations())

    if with_lr_finder:
        # this may take some time, hence the reason it is disabled by default
        callbacks.append(callback_learning_rate_finder.CallbackLearningRateFinder())

    return callbacks


def default_per_epoch_callbacks(logger=default_logger):
    """
    Default callbacks to be performed at the end of each epoch
    """
    return [
        callback_export_classification_by_epoch.CallbackExportClassificationByEpoch(),

        callback_learning_rate_recorder.CallbackLearningRateRecorder(),

        callback_epoch_summary.CallbackEpochSummary(logger=logger),
        callback_tensorboard_record_history.CallbackTensorboardRecordHistory(),
    ]


def default_post_training_callbacks(embedding_name='embedding', dataset_name=None, split_name='valid', discard_train_error_export=False, export_errors=True, explain_decision=False):
    """
    Default callbacks to be performed after the model has been trained
    """
    callbacks = [
        callback_save_last_model.CallbackSaveLastModel(),
    ]

    if export_errors:
        callbacks.append(callback_export_classification_errors.CallbackExportClassificationErrors(discard_train=discard_train_error_export))

    callbacks += [
        callback_export_classification_report.CallbackExportClassificationReport(),
        callback_export_history.CallbackExportHistory(),
        callback_export_best_history.CallbackExportBestHistory(),
        callback_tensorboard_embedding.CallbackTensorboardEmbedding(
            embedding_name=embedding_name,
            dataset_name=dataset_name,
            split_name=split_name),
        callback_tensorboard_record_model.CallbackTensorboardRecordModel(),
    ]

    if explain_decision:
        callbacks.append(callback_explain_decision.CallbackExplainDecision(split_name=split_name))

    return callbacks


def default_sum_all_losses(dataset_name, batch, loss_terms):
    """
    Default loss is the sum of all loss terms
    """
    sum_losses = 0.0
    for name, loss_term in loss_terms.items():
        loss = loss_term.get('loss')
        if loss is not None:
            # if the loss term doesn't contain a `loss` attribute, it means
            # this is not used during optimization (e.g., embedding output)
            sum_losses += loss
    return sum_losses


def trainer_callbacks_per_batch(dataset_name, split_name, batch):
    """
    Postprocessing step to be run on the batches (e.g., if we have functors, run the functor and replace it)
    
    :param dataset_name:
    :param split_name:
    :param batch:
    :return:
    """
    for name, value in batch.items():
        # if we have a callable as a batch value, run it and replace its value by the results of the functor
        # (e.g., GAN `z` randomly generated)
        if isinstance(value, collections.Callable):
            batch[name] = value(batch)


class Trainer:
    """
    This is the main class to train a model
    """
    def __init__(
            self,
            callbacks_per_batch_fn=None,
            callbacks_per_batch_loss_terms_fn=None,
            callbacks_per_epoch_fn=default_per_epoch_callbacks,
            callbacks_pre_training_fn=default_pre_training_callbacks,
            callbacks_post_training_fn=default_post_training_callbacks,
            trainer_callbacks_per_batch=trainer_callbacks_per_batch,
            run_epoch_fn=epoch_train_eval):
        """
        Args:
            callbacks_per_batch_fn: functor returning a list of callbacks. Call back must be
                callable with `(dataset_name, split_name, batch)`. Each call back will be called
                on each batch before the model is invoked

            callbacks_per_batch_loss_terms_fn: a functor returning a list of callbacks. Each callback will be executed
                after each model(batch) call. Must be callable with `dataset_name, split_name, batch, loss_terms`

            callbacks_per_epoch_fn: a functor returning a list of callbacks. Each callback will be executed at
                the end of every epoch

            callbacks_pre_training_fn: a functor returning a list of callbacks. Each callback will be executed
                before the training is started

            callbacks_post_training_fn: a functor returning a list of callbacks. Each callback will be executed
                after the training is started

            trainer_callbacks_per_batch: Postprocessing step to be run on the batches

            run_epoch_fn: the function to be used to perform training and evaluation
        """

        self.callbacks_per_batch_fn = callbacks_per_batch_fn
        self.callbacks_per_epoch_fn = callbacks_per_epoch_fn
        self.callbacks_pre_training_fn = callbacks_pre_training_fn
        self.callbacks_post_training_fn = callbacks_post_training_fn
        self.callbacks_per_batch_loss_terms_fn = callbacks_per_batch_loss_terms_fn
        self.trainer_callbacks_per_batch = trainer_callbacks_per_batch
        self.run_epoch_fn = run_epoch_fn

    @staticmethod
    def save_model(model, result, path):
        """
        Save a model
        :param model: a PyTorch model
        :param result: None or the result of the model
        :param path: where to store the model. The result will be saved at `path + '.result'`
        """
        result_cp = None
        if result is not None:
            import copy
            # we don't want this function to have side effects so copy
            # the result and strip what can't be pickled
            result_cp = copy.copy(result)
            if 'outputs' in result_cp is not None:
                result_cp['outputs'] = strip_unpickable(result_cp['outputs'])

        result_cp_path = path + '.result'
        with open(result_cp_path, 'wb') as f:
            pickle.dump(result_cp, f)
        torch.save(model, path)

    @staticmethod
    def load_model(path, with_result=False):
        """
        load a saved model

        :param with_result: if True, the results of the model will be loaded
        :param path: where to store the model. result's will be loaded from `path + '.result'`
        :return: a tuple `model, result`
        """
        result = None
        if with_result:
            result_path = path + '.result'
            with open(result_path, 'wb') as f:
                result = pickle.load(f)
        model = torch.load(path)
        return model, result

    def fit(self, options, inputs_fn, model_fn, optimizers_fn,
            losses_fn=default_sum_all_losses,
            loss_creator=create_losses_fn,
            run_prefix='default',
            with_final_evaluation=True,
            eval_every_X_epoch=1):
        """
        Fit the model

        Requirements:

        * enough main memory to store the outputs of all the datasets of a single epoch.
            If this cannot be satisfied, sub-sample the epoch so that it can fit in main memory.
        
        Notes:

        * if a feature value is Callable, its value will be replaced by the result of the call
            (e.g., this can be useful to generate `z` embedding in GANs)

        :param options:
        :param inputs_fn: a functor returning a dictionary of datasets. Alternatively, datasets infos can be specified.
                        `inputs_fn` must return one of:

                        * datasets: dictionary of dataset
                        * (datasets, datasets_infos): dictionary of dataset and additional infos
                        
                        We define:

                        * datasets: a dictionary of dataset. a dataset is a dictionary of splits. a split is a dictionary of batched features.
                        * Datasets infos are additional infos useful for the debugging of the dataset (e.g., class mappings, sample UIDs).
                        Datasets infos are typically much smaller than datasets should be loaded in loadable in memory

        :param model_fn: a functor with parameter `options` and returning a `Module` or a `ModuleDict`
        
        Depending of the type of the model, this is how it will be used:

        * `Module`: optimizer will optimize `model.parameters()`
        * `ModuleDict`: for each dataset name, the optimizer will optimize
            `model[dataset_name].parameters()`. Note that a `forward` method will need to be implemented

        :param losses_fn:
        :param optimizers_fn:
        :param loss_creator:
        :param eval_every_X_epoch: evaluate the model every `X` epochs
        :param run_prefix: the prefix of the output folder
        :param with_final_evaluation: if True, once the model is fitted, evaluate all the data again in eval mode
        :return: a tuple `model, result`
        """
        # set up our log path. This is where all the analysis of the model will be exported
        log_path = os.path.join(
            options['workflow_options']['logging_directory'],
            run_prefix + '_r{}'.format(options['workflow_options']['trainer_run']))
        options['workflow_options']['current_logging_directory'] = log_path

        # now clear our log path to remove previous files if needed
        utils.create_or_recreate_folder(log_path)
        
        if len(logging.root.handlers) == 0:
            # there is no logger configured, so add a basic one
            logging.basicConfig(
                filename=os.path.join(options['workflow_options']['logging_directory'], 'logging.txt'),
                format='%(asctime)s %(levelname)s %(name)s %(message)s',
                level=logging.DEBUG,
                filemode='w')
        
        # here we want to have our logging per training run, so add a handler
        handler = logging.FileHandler(os.path.join(log_path, 'trainer.txt'))
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)

        # instantiate the datasets, model, optimizers and losses
        logger.info('started Trainer.fit(). Options={}'.format(options))

        datasets_infos = None
        datasets = inputs_fn()
        assert datasets is not None, '`datasets` is None!'
        if isinstance(datasets, tuple):
            if len(datasets) == 2:
                logger.info('inputs_fn specified `datasets, datasets_infos`')
                datasets, datasets_infos = datasets
            else:
                assert 0, 'expected tuple `datasets` or `datasets, datasets_infos`'
            
        model = model_fn(options)
        
        if isinstance(model, torch.nn.ModuleDict):
            # if we have sub-models, we MUST define a `forward` method
            # to orchestrate the calls of sub-models
            assert 'forward' in dir(model)
        
        # migrate the model to the specified device
        device = options['workflow_options']['device']
        model.to(device)
        
        # instantiate the optimizer and scheduler
        optimizers, schedulers = optimizers_fn(datasets, model)

        losses = loss_creator(datasets, losses_fn)

        num_epochs = options['training_parameters']['num_epochs']

        if isinstance(optimizers, tuple):
            assert len(optimizers) == 2, 'expected tuple(optimizer, scheduler)'
            optimizers, schedulers = optimizers

        history = []

        callbacks_per_batch = []
        if self.trainer_callbacks_per_batch is not None:
            callbacks_per_batch.append(self.trainer_callbacks_per_batch)
        if self.callbacks_per_batch_fn is not None:
            callbacks_per_batch += self.callbacks_per_batch_fn()

        callbacks_per_batch_loss_terms = []
        if self.callbacks_per_batch_loss_terms_fn is not None:
            callbacks_per_batch_loss_terms += self.callbacks_per_batch_loss_terms_fn()

        # run the callbacks  before training
        if self.callbacks_pre_training_fn is not None:
            callbacks = self.callbacks_pre_training_fn()
            for callback in callbacks:
                callback(options, history, model, losses=losses, outputs=None, datasets=datasets,
                         datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch, optimizers_fn=optimizers_fn, optimizers=optimizers)
                #try:
                #    callback(options, history, model, losses=losses, outputs=None,
                #             datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch, optimizers_fn=optimizers_fn, optimizers=optimizers)
                #except Exception as e:
                #    print('callback={} failed with exception={}'.format(callback, e))
                #    logger.error('callback={} failed with exception={}'.format(callback, e))

        if self.callbacks_per_epoch_fn is not None:
            callbacks_per_epoch = self.callbacks_per_epoch_fn()
        else:
            callbacks_per_epoch = []
        for epoch in range(num_epochs):
            run_eval = epoch == 0 or (epoch + 1) % eval_every_X_epoch == 0

            outputs_epoch, history_epoch = self.run_epoch_fn(
                options,
                datasets,
                optimizers,
                model,
                losses,
                schedulers,
                history,
                callbacks_per_batch,
                callbacks_per_batch_loss_terms,
                run_eval=run_eval)
            history.append(history_epoch)

            for callback in callbacks_per_epoch:
                callback(options, history, model, losses=losses, outputs=outputs_epoch,
                         datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch, optimizers_fn=optimizers_fn, optimizers=optimizers)
                #try:
                #    callback(options, history, model, losses=losses, outputs=outputs_epoch,
                #             datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch, optimizers_fn=optimizers_fn, optimizers=optimizers)
                #except Exception as e:
                #    logger.error('callback={} failed with exception={}'.format(callback, e))

        # finally run the post-training callbacks
        outputs_epoch = None
        if with_final_evaluation:
            outputs_epoch, history_epoch = self.run_epoch_fn(
                options,
                datasets,
                None,
                model,
                losses,
                None,
                history,
                callbacks_per_batch,
                callbacks_per_batch_loss_terms,
                run_eval=True)
            history.append(history_epoch)

        if self.callbacks_post_training_fn is not None:
            callbacks_post_training = self.callbacks_post_training_fn()
            for callback in callbacks_post_training:
                callback(options, history, model, losses=losses, outputs=outputs_epoch, datasets=datasets,
                         datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch, optimizers_fn=optimizers_fn)
                #try:
                #    callback(options, history, model, losses=losses, outputs=outputs_epoch,
                #             datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch, optimizers_fn=optimizers_fn)
                #except Exception as e:
                #    print('callback={} failed with exception={}'.format(callback, e))
                #    logger.error('callback={} failed with exception={}'.format(callback, e))

        # increment the number of runs
        options['workflow_options']['trainer_run'] += 1

        logging.root.removeHandler(handler)

        return model, {
            'history': history,
            'options': options,
            'outputs': outputs_epoch,
            'datasets_infos': datasets_infos
        }


def strip_unpickable(outputs):
    """
    Remove the objects that cannot be pickled
    """

    # TODO not very nice code. Can we generalize this?
    o_d = collections.OrderedDict()
    for dataset_name, dataset in outputs.items():
        o_s = collections.OrderedDict()
        for split_name, split in dataset.items():
            o_n = collections.OrderedDict()
            for output_name, output in split.items():
                o_o = collections.OrderedDict()
                for metric_name, metric in output.items():
                    if 'output_ref' != metric_name:
                        o_o[metric_name] = metric
                o_n[output_name] = o_o
            o_s[split_name] = o_n
        o_d[dataset_name] = o_s
    return o_d


def run_trainer_repeat(
        trainer,
        options, inputs_fn, model_fn, optimizers_fn,
        losses_fn=default_sum_all_losses,
        loss_creator=create_losses_fn,
        run_prefix='default',
        eval_every_X_epoch=1,
        number_of_training_runs=10):
    """
    Manages multiple run of a trainer for example to repeat the training and have an idea of the variance of a model

    :param trainer:
    :param options:
    :param inputs_fn:
    :param model_fn:
    :param optimizers_fn:
    :param losses_fn:
    :param loss_creator:
    :param run_prefix:
    :param eval_every_X_epoch:
    :param number_of_training_runs:
    :return: a tuple `model, result` of the last model trained
    """
    model = None
    result = None
    for n in range(number_of_training_runs):
        logger.info('training run=%d' % n)

        model, result = trainer.fit(
            options,
            inputs_fn,
            model_fn,
            optimizers_fn,
            losses_fn=losses_fn,
            loss_creator=loss_creator,
            run_prefix=run_prefix,
            eval_every_X_epoch=eval_every_X_epoch)

    return model, result


