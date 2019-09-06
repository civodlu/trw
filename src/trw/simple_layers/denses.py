import trw.layers
import numpy as np

from trw.simple_layers import simple_layers


def denses(parent, sizes, *args, **kwargs):
    full_sizes = [np.prod(parent.shape[1:])] + sizes
    module = trw.layers.denses(sizes=full_sizes, *args, **kwargs, with_flatten=len(parent.shape) != 2)
    return simple_layers.SimpleModule(parent, module, shape=None)
