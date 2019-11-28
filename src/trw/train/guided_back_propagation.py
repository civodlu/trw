from trw.train import utilities
from trw.train import outputs as outputs_trw
import collections
import torch
import logging
import numpy as np


logger = logging.getLogger(__name__)


def post_process_output_id(output):
    assert isinstance(output, outputs_trw.Output), 'must be a `trw.train.Output`'
    return output.output


def post_process_output_for_gradient_attribution(output):
    """
    Postptocess the output to be suitable for gradient attribution.

    In particular, if we have a :class:`trw.train.OutputClassification`, we need to apply
    a softmax operation so that we can backpropagate the loss of a particular class with
    the appropriate value (1.0).

    Args:
        output: a :class:`trw.train.OutputClassification`

    Returns:
        a :class:`torch.Tensor`
    """
    assert isinstance(output, outputs_trw.Output), 'must be a `trw.train.Output`'
    if isinstance(output, outputs_trw.OutputClassification):
        return torch.softmax(output.output, dim=1)

    return output.output


class GuidedBackprop():
    """
    Produces gradients generated with guided back propagation from the given image

    .. warning:
        * We assume the model is built with `Relu` activation function

        * the model will be instrumented, use `trw.train.CleanAddedHooks` to remove the
            hooks once guided back-propagation is finished
    """
    def __init__(self, model, unguided_gradient=False, post_process_output=post_process_output_id):
        """

        Args:
            model: the model
            unguided_gradient: if `False`, calculate the guided gradient. If `True`, calculate the gradient only
            post_process_output: a function to post-process the output of a model so that it is suitable for gradient attribution
        """
        self.model = model
        self.forward_relu_outputs = []

        self.model.eval()
        self.unguided_gradient = unguided_gradient
        self.post_process_output = post_process_output

        if not unguided_gradient:
            self.update_relus()

    def update_relus(self):
        """
        Updates relu activation functions so that
            1- stores output in forward pass
            2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    @staticmethod
    def get_floating_inputs_with_gradients(inputs):
        """
        Extract inputs that have a gradient

        Args:
            inputs: a tensor of dictionary of tensors

        Returns:
            Return a list of tuple (name, input) for the input that have a gradient
        """
        if isinstance(inputs, collections.Mapping):
            # if `i` is not a floating point, we can't calculate the gradient anyway...
            i = [(input_name, i) for input_name, i in inputs.items() if hasattr(i, 'grad') and torch.is_floating_point(i)]
        else:
            i = [('input', inputs)]
        return i

    def __call__(self, inputs, target_class, target_class_name):
        """
        Generate the guided back-propagation gradient

        Args:
            inputs: a tensor or dictionary of tensors
            target_class: the target class to be explained
            target_class_name: the name of the output class if multiple outputs

        Returns:
            a tuple (output_name, dictionary (input, gradient))
        """
        self.model.eval()  # make sure we are in eval mode
        model_output = self.model(inputs)
        if isinstance(model_output, collections.Mapping):
            assert target_class_name is not None
            model_output = model_output.get(target_class_name)

        if not self.unguided_gradient and len(self.forward_relu_outputs) == 0:
            logger.error('GuidedBackprop.__call__: the model doesn\'t have RELU layers!')

        if model_output is None:
            return None

        if isinstance(model_output, outputs_trw.Output):
            model_output = self.post_process_output(model_output)

        assert len(model_output.shape) == 2, 'must have samples x class probabilities shape'

        self.model.zero_grad()

        # Target for back-prop
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[0][target_class] = 1.0

        # Backward pass
        model_output.backward(gradient=one_hot_output)

        inputs_kvp = GuidedBackprop.get_floating_inputs_with_gradients(inputs)

        # extract gradient
        inputs_kvp = {name: utilities.to_value(i.grad) for name, i in inputs_kvp}

        self.forward_relu_outputs = None  # clean the state
        return target_class_name, inputs_kvp

    @staticmethod
    def get_positive_negative_saliency(gradient):
        """
            Generates positive and negative saliency maps based on the gradient
        Args:
            gradient (numpy arr): Gradient of the operation to visualize
        returns:
            pos_saliency ( )
        """
        pos_saliency = (np.maximum(0, gradient) / gradient.max())
        neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
        return pos_saliency, neg_saliency
