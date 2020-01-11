import torch.nn as nn
from trw.layers.convs import ConvsBase


def convs_2d(
        channels,
        convolution_kernels=5,
        strides=1,
        pooling_size=2,
        convolution_repeats=1,
        activation=nn.ReLU,
        with_flatten=False,
        dropout_probability=None,
        batch_norm_kwargs=None,
        lrn_kwargs=None,
        padding='same',
        last_layer_is_output=False):
    """

    Args:
        channels: the number of channels
        convolution_kernels: for each convolution group, the kernel of the convolution
        strides: for each convolution group, the stride of the convolution
        pooling_size: the pooling size to be inserted after each convolution group
        convolution_repeats: the number of repeats of a convolution
        activation: the activation function
        with_flatten: if True, the last output will be flattened
        dropout_probability: if None, not dropout. Else the probability of dropout after each convolution
        batch_norm_kwargs: the batch norm kwargs. See the original torch functions for description. If None,
            no batch norm
        lrn_kwargs: the local response normalization kwargs. See the original torch functions for description. If
            None, not LRN
        padding: 'same' will add padding so that convolution output as the same size as input
        last_layer_is_output: if True, the last convolution will NOT have activation, dropout, batch norm, LRN
    """

    return ConvsBase(
        cnn_dim=2,
        channels=channels,
        convolution_kernels=convolution_kernels,
        strides=strides,
        pooling_size=pooling_size,
        convolution_repeats=convolution_repeats,
        activation=activation,
        with_flatten=with_flatten,
        batch_norm_kwargs=batch_norm_kwargs,
        lrn_kwargs=lrn_kwargs,
        dropout_probability=dropout_probability,
        padding=padding,
        last_layer_is_output=last_layer_is_output)

