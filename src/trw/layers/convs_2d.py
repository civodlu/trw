import torch.nn as nn
from .flatten import Flatten
from .utils import div_shape


def convs_2d(channels, convolution_kernels=(5, 5), strides=(1, 1), pooling_size=(2, 2), convolution_repeats=None, activation=nn.ReLU, with_flatten=False, dropout_probability=None, with_batchnorm=False, with_lrn=False, lrn_size=2, batchnorm_momentum=0.1, padding='same'):
    """
    
    Args:
        channels:
        convolution_kernels:
        strides:
        pooling_size:
        convolution_repeats:
        activation:
        with_flatten:
        dropout_probability:
        with_batchnorm:
        batchnorm_momentum:
        with_lrn:
        lrn_size:
        padding (str): if `same`, the convolution will be padded with zeros to keep the output shape the same as the input shape

    Returns:

    """
    ops = []
    nb_convs = len(channels) - 1
    if not isinstance(convolution_kernels, list):
        convolution_kernels = [convolution_kernels] * nb_convs
    if not isinstance(strides, list):
        strides = [strides] * nb_convs
    if not isinstance(pooling_size, list) and pooling_size is not None:
        pooling_size = [pooling_size] * nb_convs
    if convolution_repeats is None:
        convolution_repeats = [1] * nb_convs

    assert nb_convs == len(convolution_kernels), 'must be specified for each convolutional layer'
    assert nb_convs == len(strides), 'must be specified for each convolutional layer'
    assert nb_convs == len(pooling_size), 'must be specified for each convolutional layer'

    for n in range(len(channels) - 1):
        current = channels[n]
        next = channels[n + 1]

        p = 0
        if padding == 'same':
            p = div_shape(convolution_kernels[n], 2)

        ops.append(nn.Conv2d(current, next, kernel_size=convolution_kernels[n], stride=strides[n], padding=p))
        ops.append(activation())

        # repeat some convolutions if needed
        nb_repeats = convolution_repeats[n] - 1
        for r in range(nb_repeats):
            ops.append(nn.Conv2d(next, next, kernel_size=convolution_kernels[n], stride=strides[n], padding=p))
            ops.append(activation())

        if pooling_size is not None:
            ops.append(nn.MaxPool2d(pooling_size[n])),

        if with_batchnorm:
            ops.append(nn.BatchNorm2d(next, momentum=batchnorm_momentum))
            
        if with_lrn:
            ops.append(nn.LocalResponseNorm(lrn_size))

        if dropout_probability is not None:
            ops.append(nn.Dropout2d(p=dropout_probability))

    if with_flatten:
        ops.append(Flatten())
    return nn.Sequential(*ops)
