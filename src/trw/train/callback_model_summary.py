from . import callback
from . import utils
from . import trainer
import collections
import torch
import torch.nn as nn
import numpy as np
import logging


logger = logging.getLogger(__name__)


def summary(model, batch, logger, device):
    # this code is based on https://github.com/sksq96/pytorch-summary
    # and adapted to accept dictionary as input
    def register_hook(module):
        def hook(module, input, output, batch_size=-1):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = collections.OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not isinstance(module, (nn.Sequential, nn.ModuleList)) and module != model:
            if not hasattr(module, 'runtime_actions'):  # this is a trw.simple_layers.CompiledNet
                hooks.append(module.register_forward_hook(hook))

    summary = collections.OrderedDict()
    hooks = []
    model.apply(register_hook)
    model(batch)

    for h in hooks:
        h.remove()

    logger("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    logger(line_new)
    logger("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        logger(line_new)

    # assume 4 bytes/number (float on cuda).
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(utils.to_value(total_params) * 4. / (1024 ** 2.))

    logger("================================================================")
    logger("Total params: {0:,}".format(total_params))
    logger("Trainable params: {0:,}".format(trainable_params))
    logger("Non-trainable params: {0:,}".format(total_params - trainable_params))
    logger("----------------------------------------------------------------")
    logger("Forward/backward pass size (MB): %0.2f" % total_output_size)
    logger("Params size (MB): %0.2f" % total_params_size)
    logger("----------------------------------------------------------------")
    return trainable_params


class CallbackModelSummary(callback.Callback):
    """
    Display important characteristics of the model (e.g., FLOPS, number of parameters, layers, shapes)
    """
    def __init__(self, logger=utils.log_and_print, dataset_name=None, split_name=None):
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
        batch = utils.transfer_batch_to_device(batch, device=device)
        trainer.postprocess_batch(self.dataset_name, self.split_name, batch, callbacks_per_batch)
        
        trainable_parameters_method1 = summary(model, batch, logger=self.logger, device=device)

        # cross check to make sure there is no issue in the calculation:
        parameters = [p.shape for p in model.parameters() if p.requires_grad]
        parameters = [np.prod(p) if len(p) != 0 else 1 for p in parameters]
        trainable_parameters_method2 = np.sum(parameters)

        if trainable_parameters_method1 != trainable_parameters_method2:
            logger.warning('discrepencies in the number of trainable parameters. Found using hook={}, found using model.parameters()={}. Other numbers may not be valid.'.
                           format(trainable_parameters_method1, trainable_parameters_method2))
        else:
            logger.info('successfully completed CallbackModelSummary.__call__')
