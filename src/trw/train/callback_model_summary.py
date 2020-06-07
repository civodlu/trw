from trw.reporting import len_batch
from trw.train import callback
from trw.train import utilities
from trw.train import trainer
from trw.train.data_parallel_extented import DataParallelExtended
import collections
import numpy as np
import logging
import torch


logger = logging.getLogger(__name__)


def input_shape(i, root=True):
    if isinstance(i, collections.Mapping):
        return 'Dict'
    elif isinstance(i, torch.Tensor):
        shape = tuple(i.shape)

        if root:
            # we want to have all outputs/inputs in a consistent format (list of shapes)
            return [shape]
        return shape

    if isinstance(i, (list, tuple)):
        shapes = []
        for n in i:
            # make sure the shapes are flattened
            sub_shapes = input_shape(n, root=False)
            if isinstance(sub_shapes, list):
                shapes += sub_shapes
            else:
                shapes.append(sub_shapes)
        return shapes
    raise NotImplementedError()


def model_summary_base(model, batch):
    def register_hook(module):
        def hook(module, input, output):
            nonlocal parameters_counted

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = collections.OrderedDict()
            summary[m_key]['input_shape'] = input_shape(input)
            summary[m_key]['output_shape'] = input_shape(output)

            total_trainable_params = 0  # ALL parameters of the layer (including sub-module parameters)
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

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        total_params += summary[layer]["nb_params"]
        outputs = summary[layer]["output_shape"]
        if not isinstance(outputs, str):
            for output in outputs:
                total_output += np.prod(output[1:])  # remove the batch size
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]

    # assume 4 bytes/number (float on cuda)
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(utilities.to_value(total_params) * 4. / (1024 ** 2.))
    return summary, total_output_size, total_params_size, total_params, trainable_params


def model_summary(model, batch, logger):
    logger("---------------------------------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>25} {:>15} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Param #", "Total Param #")
    logger(line_new)
    logger("=========================================================================================================")

    summary, total_output_size, total_params_size, total_params, trainable_params = model_summary_base(model, batch)
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20} {:>25} {:>25} {:>15} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
            "{0:,}".format(summary[layer]["total_trainable_params"]),
        )

        logger(line_new)

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
