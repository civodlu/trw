import torch.nn as nn
from .flatten import Flatten


def denses(sizes, dropout_probability=None, with_batchnorm=False, batchnorm_momentum=0.1, activation=nn.ReLU, last_layer_is_output=False, with_flatten=True):
    """

    Args:
        sizes:
        dropout_probability:
        with_batchnorm:
        batchnorm_momentum:
        activation:
        last_layer_is_output: This must be set to `True` if the last layer of dense is actually an output. If the last layer is an output,
            we should not add batch norm, dropout or activation of the last `nn.Linear`
        with_flatten: if True, the input will be flattened
        
    Returns:

    """
    ops = []
    
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

            if with_batchnorm:
                ops.append(nn.BatchNorm1d(next, momentum=batchnorm_momentum))
    
            if dropout_probability is not None:
                ops.append(nn.Dropout(p=dropout_probability))
    return nn.Sequential(*ops)
