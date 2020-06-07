import logging
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from trw.reporting import len_batch, to_value
from trw.train import Callback, find_default_dataset_and_split_names, CleanAddedHooks
from trw.train.trainer import prepare_loss_terms, default_sum_all_losses

logger = logging.getLogger(__name__)


def generic_tracing():
    """
    Trace only basic building blocks to avoid too much clutter
    """
    return [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.Sequential,
        nn.LSTM,
        nn.GRU,
    ]


def collect_gradient_recursively(base_name, model, gradient_store):
    """
    Recursively collect the gradient with meaningful name

    Args:
        base_name: the name of the model
        model: the model
        gradient_store: where to store the collected gradients

    """
    for child in model.children():
        child_name = base_name + '/' + type(child).__name__
        for name, parameter in child.named_parameters(recurse=False):
            if parameter.requires_grad:
                parameter_name = child_name + '/' + name
                gradient_store[parameter_name] = to_value(parameter.grad)  # collect and detach the gradient

        collect_gradient_recursively(child_name, child, gradient_store)


def aggregate_stats(all_stats, batch_stat):
    for name, value in batch_stat.items():
        stat = all_stats.get(name)
        if stat is None:
            stat = {
                'min': 1e20,
                'max': -1e20,
                'mean': 0.0,
                'norm2': 0.0,
                'nb_items': 0
            }
            all_stats[name] = stat

        stat['min'] = min(stat['min'], np.min(value))
        stat['max'] = max(stat['max'], np.max(value))
        stat['mean'] += np.mean(value)
        stat['norm2'] += np.linalg.norm(value, 2)
        stat['nb_items'] += 1


def aggregate_stats_end(all_stats):
    for name, value in all_stats.items():
        nb_items = all_stats[name]['nb_items']
        all_stats[name]['mean'] /= nb_items
        all_stats[name]['norm2'] /= nb_items


def calculate_stats_gradient(
        model,
        sequence,
        nb_samples,
        aggregate_stats_fn=aggregate_stats,
        aggregate_stats_end_fn=aggregate_stats_end,
        modules_type_to_trace=generic_tracing()):
    """
    Collect the activation statistics and the gradient update stats for each layer

    Returns:
        a tuple (gradient stats, activation stats)
    """

    # inspired from:
    # https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8

    def register_hooks(module):
        def forward_hook(module, inputs, outputs):
            nonlocal batch_stats
            if isinstance(outputs, torch.Tensor):
                batch_stats[module] = to_value(outputs)
            else:
                # cater for the bigger usecase (module with single output)
                # if really, needed, the user can add intermediate debug
                # module
                warnings.warn(f'module={module} with output type={type(outputs)} is not handled!')

        if modules_type_to_trace is None or type(module) in modules_type_to_trace:
            module.register_forward_hook(forward_hook)

    gradient_stats = OrderedDict()
    activation_stats = OrderedDict()
    batch_stats = OrderedDict()

    total_samples = 0

    with CleanAddedHooks(model) as context:
        # must be in `train` mode to collect gradients
        model.train()
        model.apply(register_hooks)
        for batch in sequence:
            model.zero_grad()
            outputs = model(batch)
            loss_terms = prepare_loss_terms(outputs, batch, is_training=True)
            loss = default_sum_all_losses(None, batch, loss_terms)
            if loss is None:
                # there is no trainable parameter, abort!
                return None
            loss.backward()

            # aggregate the module gradients
            gradient_store = OrderedDict()
            collect_gradient_recursively(type(model).__name__, model, gradient_store)
            aggregate_stats_fn(gradient_stats, gradient_store)
            aggregate_stats_fn(activation_stats, batch_stats)

            # make sure we collect statics from a subset of the samples
            batch_size = len_batch(batch)
            total_samples += batch_size
            if total_samples >= nb_samples:
                break

        aggregate_stats_end_fn(gradient_stats)
        aggregate_stats_end_fn(activation_stats)

    # clean any gradient calculated by this module
    model.zero_grad()
    return gradient_stats, activation_stats


class CallbackReportingModelStatistics(Callback):
    """
    Report the activation and gradient statistics layer by layer
    """
    def __init__(self, dataset_name=None, split_name=None):
        self.split_name = split_name
        self.dataset_name = dataset_name

    def first_time(self, datasets):
        # here we only want to collect the kernels a single time per epoch, so fix the dataset/split names
        if self.dataset_name is None or self.split_name is None:
            self.dataset_name, self.split_name = find_default_dataset_and_split_names(
                datasets,
                default_dataset_name=self.dataset_name,
                default_split_name=self.split_name)

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):

        if self.dataset_name is None or self.split_name is None:
            self.first_time(datasets)

        if self.dataset_name is None or self.split_name is None:
            logger.error('can\'t find a dataset name or split name!')
            return

        logger.info('CallbackReportingModelStatistics calculating stats...')
        logger.info('exporting to SQL...')
        logger.info('CallbackReportingModelStatistics calculating done!')
