import collections
import logging
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from trw.utils import collect_hierarchical_module_name, collect_hierarchical_parameter_name, to_value, len_batch
from trw.train.callback import Callback
from trw.train.callback_reporting_model_summary import export_table
from trw.train.utilities import update_json_config, get_device, \
    transfer_batch_to_device, CleanAddedHooks, find_default_dataset_and_split_names, prepare_loss_terms, default_sum_all_losses

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


def collect_gradient(model, gradient_store):
    """
    Collect the gradient of each parameter of a given model
    Args:
        model: the model
        gradient_store: where to store the parameter gradients

    Returns:

    """
    for p in model.parameters():
        if p.requires_grad:
            gradient_store[p] = to_value(p.grad)


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
        stat['norm2'] += np.linalg.norm(np.reshape(value, (-1)), ord=2) / value.shape[0]
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
    device = get_device(model)
    with CleanAddedHooks(model) as context:
        # must be in `train` mode to collect gradients
        model.train()
        model.apply(register_hooks)
        for batch in sequence:
            batch = transfer_batch_to_device(batch, device)
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
            collect_gradient(model, gradient_store)
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


class CallbackReportingLayerStatistics(Callback):
    """
    Report the activation and gradient statistics layer by layer
    """
    def __init__(self, dataset_name=None, split_name=None, nb_samples=500, table_name='layer_statistics'):
        """

        Args:
            dataset_name: Samples from this dataset will be used to collect statistics. If `None`, a
                dataset will be automatically selected
            split_name: Samples from this split will be used to collect statistics. If `None`, a split
                will be automatically selected
            nb_samples: the number of samples used to calculate the statistics
            table_name: the name of the SQL table where the results will be stored
        """
        self.nb_samples = nb_samples
        self.split_name = split_name
        self.dataset_name = dataset_name
        self.table_name = table_name

    def first_time(self, options, datasets):
        # here we only want to collect the kernels a single time per epoch, so fix the dataset/split names
        if self.dataset_name is None or self.split_name is None:
            self.dataset_name, self.split_name = find_default_dataset_and_split_names(
                datasets,
                default_dataset_name=self.dataset_name,
                default_split_name=self.split_name)

            # set the default parameter of the graph
            config_path = options['workflow_options']['sql_database_view_path']
            update_json_config(config_path, {
                self.table_name: {
                    'default': {
                        'X Axis': 'epoch',
                        'Y Axis': 'metric_value',
                        'Group by': 'layer',
                        'discard_axis_y': 'epoch',
                        'discard_axis_x': 'metric_value',
                        'discard_group_by': 'epoch',
                        'number_of_columns': 2,
                    }
                }
            })

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):

        if self.dataset_name is None or self.split_name is None:
            self.first_time(options, datasets)

        if self.dataset_name is None or self.split_name is None:
            logger.error('can\'t find a dataset name or split name!')
            return

        logger.info('CallbackReportingLayerStatistics calculating stats...')
        gradient_stats, activation_stats = calculate_stats_gradient(
            model,
            datasets[self.dataset_name][self.split_name],
            self.nb_samples)

        module_to_name = collect_hierarchical_module_name(type(model).__name__, model)
        parameter_to_name = collect_hierarchical_parameter_name(type(model).__name__, model, with_grad_only=True)

        layer_names = []
        epochs = []
        datasets = []
        splits = []
        metrics = []
        metric_values = []

        for parameter, values in gradient_stats.items():
            for name, value in values.items():
                if name == 'nb_items':
                    continue
                parameter_name = parameter_to_name.get(parameter)
                if parameter_name is None:
                    warnings.warn(f'module could not be recursively found! Parameter={parameter}')
                    parameter_name = str(parameter)

                layer_names.append(parameter_name)
                epochs.append(len(history))
                datasets.append(self.dataset_name)
                splits.append(self.split_name)
                metrics.append('gradient_' + name)
                metric_values.append(value)

        for layer, stats in activation_stats.items():
            for name, value in stats.items():
                if name == 'nb_items' or name == 'norm2':
                    continue

                layer_name = module_to_name.get(layer)
                if layer_name is None:
                    warnings.warn(f'module could not be recursively found! Module={layer}')
                    layer_name = str(layer)

                layer_names.append(layer_name)
                epochs.append(len(history))
                datasets.append(self.dataset_name)
                splits.append(self.split_name)
                metrics.append('activation_' + name)
                metric_values.append(value)

        table = collections.OrderedDict([
            ('layer', layer_names),
            ('epoch', epochs),
            ('dataset', datasets),
            ('split', splits),
            ('metric', metrics),
            ('metric_value', metric_values),
        ])

        logger.info('exporting to SQL...')

        export_table(
            options,
            self.table_name,
            table,
            table_role='data_graph',
            clear_existing_data=False)

        logger.info('CallbackReportingLayerStatistics calculating done!')