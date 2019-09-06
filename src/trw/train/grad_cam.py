from trw.train import graph_reflection
from trw.train import utils
from trw.layers import upsample
import torch
import numpy as np
import logging


logger = logging.getLogger(__name__)


class GradCam:
    """
    Gradient-weighted Class Activation Mapping

    This is based on the paper "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
    Ramprasaath R et al.
    """
    def __init__(self, find_convolution=graph_reflection.find_last_forward_convolution):
        self.find_convolution = find_convolution

    def generate_cam(self, model, inputs, target_class_name=None, target_class=None):
        """

        TODO:
            * handle multiple-inputs

        Args:
            model: the model
            inputs: the inputs to be fed to the model
            target_class_name: the output node to be used. If `None`:
                * if model output is a single tensor then use this as target output

                * else it will use the first `OutputClassification` output

            target_class: the index of the class to explain the decision. If `None`, the class output will be used

        Returns:
            a tuple (output name, a dictionary (input_name, GradCAMs))
        """
        logger.info('generate_cam, target_class_name={}, target_class={}'.format(target_class_name, target_class))
        model.eval()  # make sure we are in eval mode
        r = self.find_convolution(model, inputs)
        if r is None:
            logger.error('`find_convolution` did not find a convolution of interest. Grad-CAM not calculated!')
            return None

        #
        # Find the convolutional layer to analyse
        #
        matched_module_output = r['matched_module_output']
        model_outputs = r['outputs']
        input_name = 'default_input'
        selected_output_name = 'default_output'

        leaves = graph_reflection.find_tensor_leaves_with_grad(matched_module_output)
        if leaves is None or len(leaves) == 0:
            # some error: can't find any input leave. Was `requires_grad` set on the input?
            logger.error('`No input could be found from the convolutional layer. Was this a set-up error? '
                         '(requires_grad` MUST be set to detect inputs). Grad-CAM not calculated!')
            return None

        if len(leaves) > 1:
            # too many inputs for a convolution. CAM will not be useful in this case
            logger.error('`Too many input for the convolution layer. Expecting a sinlge input. '
                         'Grad-CAM not calculated!')
            return None
        input_to_match = leaves[0]

        #
        # perform the forward pass and get the specified output
        # important: we MUST have the model outputs and the convolution output performed
        # in the same forward pass (else we won't get the gradient)
        #
        selected_output = None
        if target_class_name is None:
            if isinstance(model_outputs, torch.Tensor):
                selected_output = model_outputs
            else:
                for output_name, output in model_outputs.items():
                    if isinstance(output, model_outputs.OutputClassification):
                        selected_output = output.output
                        selected_output_name = output_name
                        break
                if selected_output is None:
                    # can't find a proper output to use
                    logger.error('`No suitable output detected (must be derived from `trw.train.OutputClassification`! '
                                 'Grad-CAM not calculated!'.format(target_class_name))
                    return None
        else:
            selected_output = model_outputs.get(target_class_name)
            selected_output_name = target_class_name
            if selected_output is None:
                # can't find a specified output
                # can't find a proper output to use
                logger.error('`The selected output={} was not found among the output of the model! '
                             'Grad-CAM not calculated!'.format(target_class_name))
                return None
            selected_output = selected_output.output

        #
        # calculate CAM
        #
        assert len(selected_output.shape) == 2, 'it must be a batch x class shape'
        nb_classes = selected_output.shape[1]
        nb_samples = utils.len_batch(inputs)

        model_device = utils.get_device(model)
        one_hot_output = torch.FloatTensor(nb_samples, nb_classes).to(device=model_device).zero_()
        one_hot_output[:, target_class] = 1

        model.zero_grad()
        module_output_gradient = None

        def set_module_output_gradient(g):
            nonlocal module_output_gradient
            module_output_gradient = g

        # capture the gradient in the backward of the convolutional layer
        handle = matched_module_output.register_hook(set_module_output_gradient)
        selected_output.backward(gradient=one_hot_output, retain_graph=True)
        handle.remove()
        assert module_output_gradient is not None, 'can\'t find a gradient'

        cams = []
        for sample in range(nb_samples):
            assert module_output_gradient is not None, 'BUG: the gradient did not propagate to the convolutional layer'
            guided_gradients = module_output_gradient[sample]
            guided_gradients_np = utils.to_value(guided_gradients)
            mean_axis_avg = tuple(list(range(1, len(guided_gradients.shape))))  # remove the first (i.e., filters)
            weights = np.mean(guided_gradients_np, axis=mean_axis_avg)  # Take averages for each gradient

            # Create empty numpy array for cam
            matched_module_output_py = utils.to_value(matched_module_output[sample])
            conv_shape = guided_gradients_np.shape[1:]
            cam = np.ones(conv_shape, dtype=np.float32)
            # Multiply each weight with its conv output and then, sum
            for i, w in enumerate(weights):
                cam += w * matched_module_output_py[i]

            # rescale the cam to the input. This is not pretty but it works
            # for 1D, 2D, 3D data
            # The map will be in the [0..1] interval
            input_shape = input_to_match.shape[2:]  # must remove batch & filters

            cam = cam.reshape([1, 1] + list(conv_shape))
            cam = upsample(torch.from_numpy(cam), mode='linear', size=input_shape)
            cam = utils.to_value(cam)[0, 0]  # remove the sample
            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1

            cams.append(cam)

        logger.info('generate_cam successful!')

        return selected_output_name, {
            input_name: cams
        }