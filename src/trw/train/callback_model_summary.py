from trw.train import callback
from trw.train import utilities
from trw.train import trainer
from trw.train.data_parallel_extented import DataParallelExtended
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
            nonlocal parameters_counted

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = collections.OrderedDict()

            if isinstance(input[0], collections.Mapping):
                summary[m_key]["input_shape"] = (-1)
            else:
                if isinstance(input[0], list):
                    input_shapes = [list(i.shape) for i in input[0]]
                    summary[m_key]["input_shape"] = input_shapes
                else:
                    summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                if len(output) == 2 and isinstance(output[1], tuple):
                    # this is an RNN cell
                    summary[m_key]["output_shape"] = [-1] + list(output[0].shape[1:])
                else:
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
            else:
                if output is not None and not isinstance(output, collections.Mapping):
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size
                else:
                    # the module has multiple outputs. Don't display
                    summary[m_key]["output_shape"] = -1

            total_trainable_params = 0  # ALL parameters of the layer (i.e., in case of recursion, counts all sub-module parameters)
            params = 0  # if we have recursion, these are the unique parameters
            trainable = False
            for p in module.parameters():
                if p.requires_grad:
                    total_trainable_params += np.prod(np.asarray(p.shape))

                if p not in parameters_counted:
                    # make sure there is double counted parameters!
                    parameters_counted.add(p)
                    if p.requires_grad:
                        trainable = True
                        params += np.prod(np.asarray(p.shape))

            summary[m_key]["nb_params"] = params
            summary[m_key]["trainable"] = trainable
            summary[m_key]["total_trainable_params"] = total_trainable_params

        hooks.append(module.register_forward_hook(hook))

    parameters_counted = set()
    summary = collections.OrderedDict()
    hooks = []

    if isinstance(model, (torch.nn.DataParallel, DataParallelExtended)):
        # get the underlying module only. `DataParallel` will replicate the module on the different devices
        model = model.module

    model.apply(register_hook)
    model(batch)

    for h in hooks:
        h.remove()

    logger("---------------------------------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>25} {:>15} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Param #", "Total Param #")
    logger(line_new)
    logger("=========================================================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20} {:>25} {:>25} {:>15} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
            "{0:,}".format(summary[layer]["total_trainable_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        logger(line_new)

    # assume 4 bytes/number (float on cuda).
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(utilities.to_value(total_params) * 4. / (1024 ** 2.))

    logger("================================================================================")
    logger("Total params: {0:,}".format(total_params))
    logger("Trainable params: {0:,}".format(trainable_params))
    logger("Non-trainable params: {0:,}".format(total_params - trainable_params))
    logger("--------------------------------------------------------------------------------")
    logger("Forward/backward pass size (MB): %0.2f" % total_output_size)
    logger("Params size (MB): %0.2f" % total_params_size)
    logger("--------------------------------------------------------------------------------")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'forward_backward_size_mb': total_output_size,
        'params_size_mb': total_params_size,
    }


class CallbackModelSummary(callback.Callback):
    """
    Display important characteristics of the model (e.g., FLOPS, number of parameters, layers, shapes)
    """
    def __init__(self, logger=utilities.log_and_print, dataset_name=None, split_name=None):
        self.logger = logger
        self.dataset_name = dataset_name
        self.split_name = split_name

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('started CallbackModelSummary.__call__')
        if self.dataset_name is None:
            if self.dataset_name is None:
                self.dataset_name = next(iter(datasets))

        if self.split_name is None:
            self.split_name = next(iter(datasets[self.dataset_name]))

        batch = next(iter(datasets[self.dataset_name][self.split_name]))
        
        # make sure the input data is on the same device as the model
        device = options['workflow_options']['device']
        batch = utilities.transfer_batch_to_device(batch, device=device)
        trainer.postprocess_batch(self.dataset_name, self.split_name, batch, callbacks_per_batch)
        
        model_summary(model, batch, logger=self.logger)
        logger.info('successfully completed CallbackModelSummary.__call__')
