import torch.nn as nn
from trw.layers.flatten import Flatten
import warnings


def denses(
        sizes,
        dropout_probability=None,
        with_batchnorm=None,
        batchnorm_momentum=0.1,
        activation=nn.ReLU,
        last_layer_is_output=False,
        with_flatten=True,
        batch_norm_kwargs=None):
    """

    Args:
        sizes:
        dropout_probability:
        with_batchnorm: deprecated. Use `batch_norm_kwargs`
        batchnorm_momentum: deprecated Use `batch_norm_kwargs`
        activation: the activation to be used
        last_layer_is_output: This must be set to `True` if the last layer of dense is actually an output. If the last layer is an output,
            we should not add batch norm, dropout or activation of the last `nn.Linear`
        with_flatten: if True, the input will be flattened
        batch_norm_kwargs: specify the arguments to be used by the batch normalization layer
        
    Returns:
        a nn.Module
    """
    ops = []

    if with_batchnorm is not None:
        warnings.warn('trw.layers.denses `with_batchnorm` and `batchnorm_momentum` arguments '
                      'are deprecated. Use `batch_norm_kwargs` instead!')
        assert batch_norm_kwargs is None

    
    if with_flatten:
        ops.append(Flatten())
    
    for n in range(len(sizes) - 1):
        current = sizes[n]
        next = sizes[n + 1]

        ops.append(nn.Linear(current, next))
        if n + 2 == len(sizes):
            if not last_layer_is_output:
                ops.append(activation())
        else:
            ops.append(activation())

            if with_batchnorm is not None and with_batchnorm is True:
                # deprecated. TODO remove in next releases!
                ops.append(nn.BatchNorm1d(next, momentum=batchnorm_momentum))

            if batch_norm_kwargs is not None:
                ops.append(nn.BatchNorm1d(next, **batch_norm_kwargs))
    
            if dropout_probability is not None:
                ops.append(nn.Dropout(p=dropout_probability))
    return nn.Sequential(*ops)
