import sqlite3

import torch
import torch.optim
import torch.nn
import collections
import logging
import numpy as np
import numbers
import os
import pickle
import time
import itertools
from trw.train import outputs_trw
from trw.train import callback_reporting_epoch_summary
from trw.train import callback_reporting_export_samples
from trw.train import callback_reporting_start_server
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
from trw.train import callback_worst_samples_by_epoch
from trw.train import callback_activation_statistics
from trw.train import callback_zip_sources
from trw.train import callback_export_convolution_kernel
from trw.train import callback_reporting_epoch_summary
from trw.train import utilities

logger = logging.getLogger(__name__)


def postprocess_batch(dataset_name, split_name, batch, callbacks_per_batch, batch_id=None):
    """
    Post process a batch of data (e.g., this can be useful to add additional
    data to the current batch)

    Args:
        dataset_name (str): the name of the dataset the `batch` belongs to
        split_name (str): the name of the split the `batch` belongs to
        batch: the current batch of data
        callbacks_per_batch (list): the callbacks to be executed for each batch.
            Each callback must be callable with `(dataset_name, split_name, batch)`.
            if `None`, no callbacks
        batch_id: indicate the current batch within an epoch. May be ``None``. This can be useful
            for embedding optimizer within a module (e.g., scheduler support)
    """

    # always useful: for example if we do a model composed of multiple sub-models (one per dataset)
    # we would need to know what sub-model to use
    batch['dataset_name'] = dataset_name
    batch['split_name'] = split_name
    if batch_id is not None:
        batch['batch_id'] = batch_id
    
    if callbacks_per_batch is not None:
        for callback in callbacks_per_batch:
            callback(dataset_name, split_name, batch)


def prepare_loss_terms(outputs, batch, is_training):
    """
    Return the loss_terms for the given outputs
    """
    loss_terms = collections.OrderedDict()
    for output_name, output in outputs.items():
        assert isinstance(output, outputs_trw.Output), f'output must be a `trw.train.Output`' \
                                                       f' instance. Got={type(output)}'
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


def aggregate_values(values):
    if len(values) == 0:
        return None
    value = values[0]
    if isinstance(value, np.ndarray):
        # concatenate tensors (e.g., softmax output)
        if len(value.shape) == 0:
            return np.average(values)
        else:
            return np.concatenate(values)
    elif isinstance(value, numbers.Number):
        # average numbers (e.g., losses)
        return np.sum(values) / len(values)
    elif isinstance(value, torch.Tensor):
        if len(value.shape) > 0:
            return torch.cat(values)
        else:
            return torch.sum(torch.stack(values)) / len(values)
    elif isinstance(value, outputs_trw.Output):
        return values[0]
    elif isinstance(value, list):
        return list(itertools.chain.from_iterable(values))
    else:
        assert 0, 'this type=`{}` is not handled!'.format(type(value))


def aggregate_list_of_dicts(list_of_dicts):
    if len(list_of_dicts) == 0:
        return {}

    keys = list_of_dicts[0].keys()
    aggregated = collections.OrderedDict()
    for key in keys:
        values = [dict[key] for dict in list_of_dicts]
        values = [v for v in values if v is not None]
        aggregated[key] = aggregate_values(values)
    return aggregated


def aggregate_list_of_metrics(list_of_metrics):
    if len(list_of_metrics) == 0:
        return {}

    keys = list_of_metrics[0].keys()
    aggregated = collections.OrderedDict()
    for key in keys:
        values = [dict[key] for dict in list_of_metrics]
        aggregated_values = key.aggregate_metrics(values)
        for name, value in aggregated_values.items():
            aggregated[name] = value
    return aggregated


def generic_aggregate_loss_terms(loss_terms_history):
    """
    Aggregate the loss terms for all the internal_nodes of an epoch

    Args:
        loss_terms_history: a list of loss terms

    Returns:
        a tuple `output, history`. `output` is maintained alive only during the current epoch.
            `history` is kept in memory during the whole training
    """

    if loss_terms_history is None or len(loss_terms_history) == 0:
        return {}, []

    output_names = loss_terms_history[0].keys()

    # aggregate outputs and metrics by output name
    aggregated_outputs = collections.OrderedDict()
    aggregated_metrics = collections.OrderedDict()
    for output_name in output_names:
        loss_term_outputs = []
        loss_term_metrics_results = []
        if output_name == 'overall_loss':
            continue
        for loss_term in loss_terms_history:
            loss_term_output = loss_term[output_name]
            loss_term_metrics_result = loss_term_output.get('metrics_results')
            if loss_term_metrics_result is not None:
                del loss_term_output['metrics_results']
                loss_term_metrics_results.append(loss_term_metrics_result)
            loss_term_outputs.append(loss_term_output)

        aggregated_outputs[output_name] = aggregate_list_of_dicts(loss_term_outputs)
        aggregated_metrics[output_name] = aggregate_list_of_metrics(loss_term_metrics_results)

    # keep the `overall_loss` in the metrics
    overall_losses = []
    for loss_terms in loss_terms_history:
        loss = loss_terms.get('overall_loss')
        if loss is not None:
            overall_losses.append(loss['loss'])

    if len(overall_losses) > 0:
        loss = aggregate_values(overall_losses)
        aggregated_metrics['overall_loss'] = {'loss': loss}

    return aggregated_outputs, aggregated_metrics


def loss_term_cleanup(loss_terms):
    """
    Perform cleanup on all the loss terms

    Requires ``outputs.Output.output_ref_tag`` tag for each loss term, else no cleanup will be done
    for this loss term.

    Args:
        loss_terms: the loss terms to be cleaned up
    """
    for name, loss_term in loss_terms.items():
        ref = loss_term.get(outputs_trw.Output.output_ref_tag)
        if ref is not None:
            ref.loss_term_cleanup(loss_term)


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
        callbacks_per_batch_loss_terms):
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

    Notes:
        if ``optimizer`` is None, there MUST be a ``.backward()`` to free graph and memory.
    """
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
            assert isinstance(batch, collections.Mapping), 'batch must be a mapping of (feature name, feature values)'
            # calculate the time for batch processing. In particular
            # this may be significant when using large data augmentations
            # and useful to optimize the data processing pipeline
            current_batch_processing = time.perf_counter() - batch_processing_last
            total_batch_processing_time += current_batch_processing

            total_collate_and_postprocess_start = time.perf_counter()
            batch = utilities.transfer_batch_to_device(batch, device)
            
            postprocess_batch(dataset_name, split_name, batch, callbacks_per_batch)
            total_collate_and_postprocess_end = time.perf_counter()
            total_collate_and_postprocess += total_collate_and_postprocess_end - total_collate_and_postprocess_start

            if optimizer is not None:
                optimizer.zero_grad()

            outputs = model(batch)

            assert isinstance(outputs, collections.Mapping), 'model must create a dict of outputs'
            loss_terms = prepare_loss_terms(outputs, batch, is_training=True)
            loss = loss_fn(dataset_name, batch, loss_terms)

            if optimizer is not None and isinstance(loss, torch.Tensor):
                if isinstance(loss, torch.Tensor):
                    # if there is no optimizer, it means we did not want to change the parameters
                    loss.backward()
                else:
                    logger.warning('No backward calculated for={}/{}'.format(dataset_name, split_name))
            loss_terms['overall_loss'] = {'loss': float(utilities.to_value(loss))}
            
            if callbacks_per_batch_loss_terms is not None:
                for callback in callbacks_per_batch_loss_terms:
                    callback(dataset_name, split_name, batch, loss_terms)

            # call optimizer step after the callbacks (e.g., a callback could be used to clip the gradient)
            if optimizer is not None:
                optimizer.step()

            # once we are done, we want to perform some cleanup. For example, we do NOT want to keep CUDA based
            # tensors in the output so we can run clean up to transfer CUDA based memory to numpy
            loss_term_cleanup(loss_terms)

            all_loss_terms.append(loss_terms)
            batch_processing_last = time.perf_counter()
            nb_samples += utilities.len_batch(batch)

    except StopIteration:
        pass
    loop_ended = time.perf_counter()
    
    logger.debug('nb_samples={}, train_loop total_batch_processing_time={}, loop_time={},'
                 ' collate_and_postprocess={}, dataset_name={}, split_name={}'.format(
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
            assert isinstance(batch, collections.Mapping), 'batch must be a mapping of (feature name, feature values)'
            batch = utilities.transfer_batch_to_device(batch, device=device)
            postprocess_batch(dataset_name, split_name, batch, callbacks_per_batch)
            with torch.no_grad():  # do not keep track of the gradient as we are just evaluating
                outputs = model(batch)
                loss_terms = prepare_loss_terms(outputs, batch, is_training=False)
                loss = loss_fn(dataset_name, batch, loss_terms)
                loss_terms['overall_loss'] = {'loss': float(utilities.to_value(loss))}
                all_loss_terms.append(loss_terms)

                if callbacks_per_batch_loss_terms is not None:
                    for callback in callbacks_per_batch_loss_terms:
                        callback(dataset_name, split_name, batch, loss_terms)
                        
                # clean the loss terms (e.g., free memory)
                loss_term_cleanup(loss_terms)

    except StopIteration:
        pass
    return all_loss_terms


def approximate_batch_size_from_loss_terms(all_loss_terms):
    """
    Calculate on approximation of the number of samples from the loss terms. Error can be up to the number of
    samples within one batch
    """
    for name, values in all_loss_terms[0].items():
        s = utilities.len_batch(values)
        if s != 0:
            return s * len(all_loss_terms)
    return 0


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
        force_eval_mode,
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
    :param force_eval_mode: if ``True``, the train split will be evaluated using the eval loop instead of train loop.
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
            time_start = time.perf_counter()
            if split_name == train_split_name and not force_eval_mode:
                # * only the split `train_split_name` is considered as training, all
                # other splits are for evaluation only
                # * if we don't have optimizers, we still want to have
                # gradients (e.g., for model with their own internal optimizers)
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
            time_end = time.perf_counter()
            assert isinstance(all_loss_terms, collections.Sequence), '`all_loss_terms` must be a sequence'

            if len(all_loss_terms) != 0:
                epoch_outputs, epoch_history = generic_aggregate_loss_terms(all_loss_terms)
                epoch_history['info'] = {
                    'time': time_end - time_start,
                    'nb_samples': approximate_batch_size_from_loss_terms(all_loss_terms)
                }
                dataset_history[split_name] = epoch_history
                dataset_outputs[split_name] = epoch_outputs

        history_by_dataset_epoch[dataset_name] = dataset_history
        outputs_by_dataset_epoch[dataset_name] = dataset_outputs

        if scheduler is not None:
            scheduler.step()

    return outputs_by_dataset_epoch, history_by_dataset_epoch


default_logger = utilities.log_and_print


def default_pre_training_callbacks(logger=default_logger, with_lr_finder=False, with_export_augmentations=True, with_reporting_server=True):
    """
    Default callbacks to be performed before the fitting of the model
    """
    callbacks = [
        callback_tensorboard.CallbackClearTensorboardLog(),  # make sure the previous model train log is removed
        callback_model_summary.CallbackModelSummary(logger=logger),
        callback_data_summary.CallbackDataSummary(logger=logger),
        callback_zip_sources.CallbackZipSources(folders_to_record=os.path.join(os.path.dirname(__file__), '..', '..')),
        callback_reporting_export_samples.CallbackReportingExportSamples(table_name='random_samples'),
    ]

    if with_reporting_server:
        callbacks.append(callback_reporting_start_server.CallbackReportingStartServer())
    
    if with_export_augmentations:
        callbacks.append(callback_export_augmentations.CallbackExportAugmentations())

    if with_lr_finder:
        # this may take some time, hence the reason it is disabled by default
        callbacks.append(callback_learning_rate_finder.CallbackLearningRateFinder())

    return callbacks


def default_per_epoch_callbacks(
        logger=default_logger,
        with_worst_samples_by_epoch=True,
        with_activation_statistics=False,
        convolutional_kernel_export_frequency=None):
    """
    Default callbacks to be performed at the end of each epoch
    """
    callbacks = [
        callback_learning_rate_recorder.CallbackLearningRateRecorder(),
        callback_epoch_summary.CallbackEpochSummary(logger=logger),
        callback_tensorboard_record_history.CallbackTensorboardRecordHistory(),
        callback_reporting_epoch_summary.CallbackReportingRecordHistory(),
    ]

    if convolutional_kernel_export_frequency is not None:
        callbacks.append(callback_export_convolution_kernel.CallbackExportConvolutionKernel(
            export_frequency=convolutional_kernel_export_frequency))

    if with_worst_samples_by_epoch:
        callbacks.append(callback_worst_samples_by_epoch.CallbackWorstSamplesByEpoch())

    if with_activation_statistics:
        callbacks.append(callback_activation_statistics.CallbackActivationStatistics())

    return callbacks


def default_post_training_callbacks(
        embedding_name='embedding',
        dataset_name=None,
        split_name=None,
        discard_train_error_export=False,
        export_errors=True,
        explain_decision=True):
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
    def save_model(model, result, path, pickle_module=pickle):
        """
        Save a model to file

        Args:
            model: the model to serialize
            result: an optional result file associated with the model
            path: the base path to save the model
            pickle_module: the serialization module that will be used to save the model and results

        """
        result_cp = None
        sql_database = None
        if result is not None:
            import copy
            # we don't want this function to have side effects so copy
            # the result and strip what can't be pickled
            result_cp = copy.copy(result)

            if 'outputs' in result_cp is not None:
                result_cp['outputs'] = strip_unpickable(result_cp['outputs'])

            sql_database = utilities.safe_lookup(result_cp, 'options', 'workflow_options', 'sql_database')
            if sql_database is not None:
                del result_cp['options']['workflow_options']['sql_database']

        result_cp_path = path + '.result'
        with open(result_cp_path, 'wb') as f:
            pickle_module.dump(result_cp, f)
        torch.save(model, path, pickle_module=pickle_module)

        if sql_database is not None:
            # TODO find a cleaner and generic way of doing this...
            result_cp['options']['workflow_options']['sql_database'] = sql_database

    @staticmethod
    def load_model(path, with_result=False, device=None, pickle_module=pickle):
        """
        load a saved model

        Args:
            path: where to store the model. result's will be loaded from `path + '.result'`
            with_result: if True, the results of the model will be loaded
            device: where to load the model. For example, models are typically trained on GPU,
                but for deployment, CPU might be good enough. If `None`, use the same device as
                when the model was exported
            pickle_module: the de-serialization module to be used to load model and results

        Returns:
            a tuple `model, result`
        """
        result = None
        if with_result:
            result_path = path + '.result'
            with open(result_path, 'rb') as f:
                result = pickle_module.load(f)
        model = torch.load(path, map_location=device, pickle_module=pickle_module)
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
        utilities.create_or_recreate_folder(log_path)
        
        if len(logging.root.handlers) == 0:
            # there is no logger configured, so add a basic one
            logging.basicConfig(
                filename=os.path.join(options['workflow_options']['logging_directory'], 'logging.txt'),
                format='%(asctime)s %(levelname)s %(name)s %(message)s',
                level=logging.DEBUG,
                filemode='w')

        # create the reporting SQL database
        sql_path = os.path.join(options['workflow_options']['current_logging_directory'], 'reporting_sqlite.db')
        sql = sqlite3.connect(sql_path)
        options['workflow_options']['sql_database'] = sql
        options['workflow_options']['sql_database_path'] = sql_path
        options['workflow_options']['sql_database_view_path'] = sql_path.replace('.db', '.json')

        # here we want to have our logging per training run, so add a handler
        handler = logging.FileHandler(os.path.join(log_path, 'trainer.txt'))
        formatter = utilities.RuntimeFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)

        # instantiate the datasets, model, optimizers and losses
        logger.info('started Trainer.fit(). Options={}'.format(options))

        datasets_infos = None
        logger.info('creating datasets...')
        datasets = inputs_fn()
        logger.info('datasets created successfully!')
        assert datasets is not None, '`datasets` is None!'
        if isinstance(datasets, tuple):
            if len(datasets) == 2:
                logger.info('inputs_fn specified `datasets, datasets_infos`')
                datasets, datasets_infos = datasets
            else:
                assert 0, 'expected tuple `datasets` or `datasets, datasets_infos`'

        logger.info('creating model...')
        model = model_fn(options)
        logger.info('model created successfully!')
        
        if isinstance(model, torch.nn.ModuleDict):
            # if we have sub-models, we MUST define a `forward` method
            # to orchestrate the calls of sub-models
            assert 'forward' in dir(model)
        
        # migrate the model to the specified device
        device = options['workflow_options']['device']

        logger.info('model moved to device={}'.format(device))
        model.to(device)
        
        # instantiate the optimizer and scheduler
        logger.info('creating optimizers...')
        if optimizers_fn is not None:
            optimizers, schedulers = optimizers_fn(datasets, model)
            logger.info('optimizers created successfully!')
        else:
            logger.info('optimizer fn is None! No optimizer created.')
            optimizers, schedulers = None, None

        logger.info('creating losses...')
        losses = loss_creator(datasets, losses_fn)
        logger.info('losses created successfully!')

        num_epochs = options['training_parameters']['num_epochs']

        if isinstance(optimizers, tuple):
            assert len(optimizers) == 2, 'expected tuple(optimizer, scheduler)'
            optimizers, schedulers = optimizers

        history = []

        logger.info('creating callbacks...')
        if self.callbacks_per_epoch_fn is not None:
            callbacks_per_epoch = self.callbacks_per_epoch_fn()
        else:
            callbacks_per_epoch = []
            
        callbacks_per_batch = []
        if self.trainer_callbacks_per_batch is not None:
            callbacks_per_batch.append(self.trainer_callbacks_per_batch)
        if self.callbacks_per_batch_fn is not None:
            callbacks_per_batch += self.callbacks_per_batch_fn()

        callbacks_per_batch_loss_terms = []
        if self.callbacks_per_batch_loss_terms_fn is not None:
            callbacks_per_batch_loss_terms += self.callbacks_per_batch_loss_terms_fn()
        logger.info('callbacks created successfully!')

        # run the callbacks  before training
        if self.callbacks_pre_training_fn is not None:
            logger.info('running pre-training callbacks...')
            callbacks = self.callbacks_pre_training_fn()
            for callback in callbacks:
                callback(options, history, model, losses=losses, outputs=None, datasets=datasets,
                         datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch,
                         optimizers_fn=optimizers_fn, optimizers=optimizers)
                #try:
                #    callback(options, history, model, losses=losses, outputs=None,
                #             datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch, optimizers_fn=optimizers_fn, optimizers=optimizers)
                #except Exception as e:
                #    print('callback={} failed with exception={}'.format(callback, e))
                #    logger.error('callback={} failed with exception={}'.format(callback, e))
            logger.info('pre-training callbacks completed!')

        for epoch in range(num_epochs):
            logger.info('started training epoch {}'.format(epoch))
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
                run_eval=run_eval,
                force_eval_mode=False)
            history.append(history_epoch)

            logger.info('finished training epoch {}'.format(epoch))

            last_epoch = epoch + 1 == num_epochs

            logger.info('callbacks started')
            for callback in callbacks_per_epoch:
                callback(options, history, model, losses=losses, outputs=outputs_epoch,
                         datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch,
                         optimizers_fn=optimizers_fn, optimizers=optimizers, last_epoch=last_epoch)
                #try:
                #    callback(options, history, model, losses=losses, outputs=outputs_epoch,
                #             datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch,
                #             optimizers_fn=optimizers_fn, optimizers=optimizers, last_epoch=last_epoch)
                #except Exception as e:
                #    logger.error('callback={} failed with exception={}'.format(callback, e))

            logger.info('callbacks epoch {} finished'.format(epoch))

        # finally run the post-training callbacks
        outputs_epoch = None
        if with_final_evaluation:
            logger.info('started final evaluation...')
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
                run_eval=True,
                force_eval_mode=True)
            logger.info('finished final evaluation...')
            history.append(history_epoch)

        if self.callbacks_post_training_fn is not None:
            logger.info('started post training callbacks...')
            callbacks_post_training = self.callbacks_post_training_fn()
            for callback in callbacks_post_training:
                callback(options, history, model, losses=losses, outputs=outputs_epoch, datasets=datasets,
                         datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch,
                         optimizers_fn=optimizers_fn)
                #try:
                #    callback(options, history, model, losses=losses, outputs=outputs_epoch,
                #             datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch, optimizers_fn=optimizers_fn)
                #except Exception as e:
                #    print('callback={} failed with exception={}'.format(callback, e))
                #    logger.error('callback={} failed with exception={}'.format(callback, e))

            logger.info('finished post training callbacks...')

            del callbacks_post_training
            logger.info('deleted post training callbacks!')

        # increment the number of runs
        options['workflow_options']['trainer_run'] += 1

        logger.info('removing logging handlers...')
        logging.root.removeHandler(handler)

        logger.info('training completed!')

        sql.commit()
        sql.close()

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
    if outputs is None:
        return None

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

    Args:
        trainer:
        options:
        inputs_fn:
        model_fn:
        optimizers_fn:
        losses_fn:
        loss_creator:
        run_prefix:
        eval_every_X_epoch:
        number_of_training_runs:

    Returns:
        a tuple `model, result` of the last model trained
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


