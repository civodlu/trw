from trw.train import callback
from trw.train import utilities
from trw.train import trainer
import collections
import numpy as np
import logging
import torch


logger = logging.getLogger(__name__)


def model_summary(model, batch, logger):
    # this code is based on https://github.com/sksq96/pytorch-summary
    # and adapted to accept dictionary as input
    def register_hook(module):
        def hook(module, input, output, batch_size=-1):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = collections.OrderedDict()

            if isinstance(input[0], collections.Mapping):
                summary[m_key]['input_shape'] = (-1)
            else:
                if isinstance(input[0], list):
                    input_shapes = [list(i.shape) for i in input[0]]
                    summary[m_key]['input_shape'] = input_shapes
                else:
                    summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                if not isinstance(output, collections.Mapping):
                    summary[m_key]['output_shape'] = list(output.size())
                    summary[m_key]['output_shape'][0] = batch_size
                else:
                    # the module has multiple outputs. Don't y_axis
                    summary[m_key]['output_shape'] = -1

            if not isinstance(output, collections.Mapping):
                with torch.no_grad():
                    mean_output = utilities.to_value(torch.mean(output))
                    std_output = utilities.to_value(torch.std(output))
                    min_output = utilities.to_value(torch.min(output))
                    max_output = utilities.to_value(torch.max(output))

                    summary[m_key]['mean_output'] = mean_output
                    summary[m_key]['std_output'] = std_output
                    summary[m_key]['min_output'] = min_output
                    summary[m_key]['max_output'] = max_output

        hooks.append(module.register_forward_hook(hook))

    summary = collections.OrderedDict()
    hooks = []
    model.apply(register_hook)
    model(batch)

    for h in hooks:
        h.remove()

    logger('--------------------------------------------------------------------------------------------------------------------')
    logger('                                     Output layer activation summary                                                ')
    logger('--------------------------------------------------------------------------------------------------------------------')
    line_new = '{:>20} {:>25} {:>25} {:>10} {:>10} {:>10} {:>10}'.format('Layer (type)', 'Input Shape', 'Output Shape', 'Mean', 'Std', 'Min', 'Max')
    logger(line_new)
    logger('====================================================================================================================')
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if not 'mean_output' in summary[layer]:
            continue

        line_new = '{:>20} {:>25} {:>25} {:>10} {:>10} {:>10} {:>10}'.format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            '{:.3f}'.format(summary[layer]['mean_output']),
            '{:.3f}'.format(summary[layer]['std_output']),
            '{:.3f}'.format(summary[layer]['min_output']),
            '{:.3f}'.format(summary[layer]['max_output']),
        )
        logger(line_new)


class CallbackActivationStatistics(callback.Callback):
    """
    Calculate activation statistics of each layer of a neural network.

    This can be useful to detect connectivity issues within the network, overflow and underflow which may impede
    the training of the network.
    """
    def __init__(self, dataset_name=None, split_name='train', logger_fn=utilities.log_and_print):
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.initialized = False
        self.logger_fn = logger_fn
        logger.info('created callback `CallbackTensorboardEmbedding`. dataset_name={}, split_name={}'.format(dataset_name, split_name))

    def first_time(self, datasets):
        self.initialized = True
        if self.dataset_name is None:
            self.dataset_name = next(iter(datasets))

        dataset = datasets.get(self.dataset_name)
        if dataset is None:
            logger.error('can\'t find dataset={}'.format(self.dataset_name))
            self.dataset_name = None
            return

        if self.split_name is None:
            self.split_name = next(iter(dataset))

        if dataset.get(self.split_name) is None:
            logger.error('can\'t find split={} for dataset={}'.format(self.dataset_name, self.split_name))
            self.dataset_name = None
            return

        logger.info('CallbackTensorboardEmbedding initialized. dataset_name={}, split_name={}'.format(self.dataset_name, self.split_name))

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('CallbackTensorboardEmbedding.__call__')
        if not self.initialized:
            self.first_time(datasets)
        if self.dataset_name is None:
            return

        device = options['workflow_options']['device']
        batch = next(iter(datasets[self.dataset_name][self.split_name]))
        batch = utilities.transfer_batch_to_device(batch, device=device)
        trainer.postprocess_batch(self.dataset_name, self.split_name, batch, callbacks_per_batch)

        model_summary(model, batch, logger=self.logger_fn)
        logger.info('successfully completed CallbackModelSummary.__call__')
