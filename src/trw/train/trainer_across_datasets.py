import collections
import time
import logging
import torch
import numpy as np


import trw
import trw.utils
from trw.train import utilities
from trw.train.utilities import prepare_loss_terms
from .trainer import eval_loop,generic_aggregate_loss_terms,approximate_batch_size_from_loss_terms, postprocess_batch, loss_term_cleanup

logger = logging.getLogger(__name__)


def train_loop_across_datasets(
        device,
        datasets,
        train_split_name,
        optimizer,
        model,
        losses,
        history,
        callbacks_per_batch,
        callbacks_per_batch_loss_terms):
    """
    Run the train loop (i.e., the model parameters will be updated)
    First, an aggregate batch is created by concatenating a single batch for each datasets;
    then forward and backward pass are performed.

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

    all_loss_terms_by_dataset = collections.OrderedDict(
                    {dataset_name: [] for dataset_name in datasets.keys()
                    if train_split_name in datasets[dataset_name].keys()})
    iterators_by_dataset = collections.OrderedDict(
                    {dataset_name:iter(datasets[dataset_name][train_split_name]) for dataset_name in datasets.keys()
                    if train_split_name in datasets[dataset_name].keys()})
    has_stopped_iteration_by_dataset = collections.OrderedDict(
                    {dataset_name:False for dataset_name in datasets.keys()
                    if train_split_name in datasets[dataset_name].keys()})


    total_batch_processing_time = {dataset_name:0.0 for dataset_name in datasets.keys()}
    batch_processing_last = {dataset_name:0.0 for dataset_name in datasets.keys()}
    loop_started = {dataset_name:0.0 for dataset_name in datasets.keys()}
    total_collate_and_postprocess = {dataset_name:0.0 for dataset_name in datasets.keys()}
    nb_samples = {dataset_name:0 for dataset_name in datasets.keys()}

    i = 0
    while not all([s for s in has_stopped_iteration_by_dataset.values()]):
        loss_terms = collections.OrderedDict()
        loss_across_datasets = 0
        if optimizer is not None:
            optimizer.zero_grad()
        single_batches = collections.OrderedDict()
        single_batches_lenghts = []
        for dataset_name in iterators_by_dataset.keys():
            if not has_stopped_iteration_by_dataset[dataset_name]:
                if i==0:
                    batch_processing_last[dataset_name] = time.perf_counter()
                    loop_started[dataset_name] = time.perf_counter()
                try:
                    single_batch = next(iterators_by_dataset[dataset_name])

                    assert isinstance(single_batch, collections.Mapping), 'batch must be a mapping of (feature name, feature values)'

                    single_batches_lenghts.append(trw.utils.len_batch(single_batch))

                    # calculate the time for batch processing. In particular
                    # this may be significant when using large data augmentations
                    # and useful to optimize the data processing pipeline
                    current_batch_processing = time.perf_counter() - batch_processing_last[dataset_name]
                    total_batch_processing_time[dataset_name] += current_batch_processing

                    total_collate_and_postprocess_start = time.perf_counter()
                    single_batch = utilities.transfer_batch_to_device(single_batch, device)

                    postprocess_batch(dataset_name, train_split_name, single_batch, callbacks_per_batch, batch_id=i)
                    total_collate_and_postprocess_end = time.perf_counter()
                    total_collate_and_postprocess[dataset_name] += total_collate_and_postprocess_end - total_collate_and_postprocess_start

                    single_batches[dataset_name] = single_batch


                except StopIteration:
                    has_stopped_iteration_by_dataset[dataset_name] = True

                    loop_ended = time.perf_counter()

                    logger.debug('nb_samples={}, train_loop total_batch_processing_time={}, loop_time={},'
                                 ' collate_and_postprocess={}, dataset_name={}, split_name={}'.format(
                        nb_samples[dataset_name],
                        total_batch_processing_time[dataset_name],
                        loop_ended - loop_started[dataset_name],
                        total_collate_and_postprocess[dataset_name],
                        dataset_name,
                        train_split_name))



        if not all([s for s in has_stopped_iteration_by_dataset.values()]):

            batch = trw.train.default_collate_fn([v for v in single_batches.values()],None)
            # set a single batch id, else the batch is incorrectly scattered across GPUs when using torch.nn.DataParallel
            batch['batch_id'] = i

            outputs = model(batch)
            if outputs is None:
                # skip this batch
                continue

            assert isinstance(outputs, collections.Mapping), 'model must create a dict of outputs'

            outputs_values = [o.output for o in outputs.values()]
            single_batches_pos = np.concatenate([[0],np.cumsum(single_batches_lenghts)])

            n=0
            for dataset_name in iterators_by_dataset.keys():
                if not has_stopped_iteration_by_dataset[dataset_name]:
                    for k, output in enumerate(outputs.values()):
                        output.output = outputs_values[k][single_batches_pos[n]:single_batches_pos[n+1],...]
                    n+=1
                    loss_terms[dataset_name] = prepare_loss_terms(outputs, single_batches[dataset_name], is_training=True)
                    loss_fn = losses[dataset_name]
                    loss = loss_fn(dataset_name, single_batches[dataset_name], loss_terms[dataset_name])
                    loss_across_datasets+=loss


                    loss_terms[dataset_name]['overall_loss'] = {'loss': float(trw.utils.to_value(loss))}

                    if callbacks_per_batch_loss_terms is not None:
                        for callback in callbacks_per_batch_loss_terms:
                            callback(dataset_name, train_split_name, single_batches[dataset_name], loss_terms[dataset_name])


                    batch_processing_last[dataset_name] = time.perf_counter()
                    nb_samples[dataset_name] += trw.utils.len_batch(single_batches[dataset_name])


            if optimizer is not None and isinstance(loss_across_datasets, torch.Tensor):
                if isinstance(loss_across_datasets, torch.Tensor):
                    # if there is no optimizer, it means we did not want to change the parameters
                    loss_across_datasets.backward()
                else:
                    logger.warning('No backward calculated')

            # call optimizer step after the callbacks (e.g., a callback could be used to clip the gradient)
            if optimizer is not None:
                optimizer.step()

            for dataset_name in iterators_by_dataset.keys():
                if not has_stopped_iteration_by_dataset[dataset_name]:
                    # once we are done, we want to perform some cleanup. For example, we do NOT want to keep CUDA based
                    # tensors in the output so we can run clean up to transfer CUDA based memory to numpy
                    loss_term_cleanup(loss_terms[dataset_name])

                    all_loss_terms_by_dataset[dataset_name].append(loss_terms[dataset_name])

        i+=1

    return  all_loss_terms_by_dataset


def epoch_train_eval_across_datasets(
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
        train_loop_fn=train_loop_across_datasets):
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

    # train loop
    if not force_eval_mode:
        optimizer = None
        if optimizers is not None:
            optimizer = optimizers.get([i for i in datasets.keys()][0])
        scheduler = None
        if schedulers is not None:
            scheduler = schedulers.get([i for i in datasets.keys()][0])

        time_start = time.perf_counter()
        all_loss_terms_by_dataset = train_loop_fn(
            device,
            datasets,
            train_split_name,
            optimizer,
            model,
            losses,
            history,
            callbacks_per_batch=callbacks_per_batch,
            callbacks_per_batch_loss_terms=callbacks_per_batch_loss_terms)
        time_end = time.perf_counter()

        for dataset_name, dataset in datasets.items():
            history_by_dataset_epoch[dataset_name] = collections.OrderedDict()
            outputs_by_dataset_epoch[dataset_name] = collections.OrderedDict()
            for split_name, split in dataset.items():
                if split_name == train_split_name:
                    assert isinstance(all_loss_terms_by_dataset[dataset_name], collections.Sequence), '`all_loss_terms` must be a sequence'

                    if len(all_loss_terms_by_dataset[dataset_name]) != 0:
                        epoch_outputs, epoch_history = generic_aggregate_loss_terms(all_loss_terms_by_dataset[dataset_name])
                        epoch_history['info'] = {
                            'time': time_end - time_start,
                            'nb_samples': approximate_batch_size_from_loss_terms(all_loss_terms_by_dataset[dataset_name])
                        }
                        history_by_dataset_epoch[dataset_name][split_name] = epoch_history
                        outputs_by_dataset_epoch[dataset_name][split_name] = epoch_outputs


        if scheduler is not None:
            scheduler.step()

    # eval loop
    for dataset_name, dataset in datasets.items():
        loss_fn = losses[dataset_name]
        if dataset_name not in history_by_dataset_epoch.keys():
            history_by_dataset_epoch[dataset_name] = collections.OrderedDict()
        if dataset_name not in outputs_by_dataset_epoch.keys():
            outputs_by_dataset_epoch[dataset_name] = collections.OrderedDict()
        for split_name, split in dataset.items():
            if split_name != train_split_name or force_eval_mode:
                time_start = time.perf_counter()
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
                    history_by_dataset_epoch[dataset_name][split_name] = epoch_history
                    outputs_by_dataset_epoch[dataset_name][split_name] = epoch_outputs



    return outputs_by_dataset_epoch, history_by_dataset_epoch