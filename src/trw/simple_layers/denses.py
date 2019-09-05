import trw.layers

from trw.simple_layers import simple_layers


def denses(parent, sizes, *args, **kwargs):
    full_sizes = [parent.shape[1]] + sizes
    module = trw.layers.denses(sizes=full_sizes, *args, **kwargs)
    return simple_layers.SimpleModule(parent, module, shape=None)
