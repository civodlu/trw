import trw.layers

from trw.simple_layers import simple_layers


def convs_3d(parent, channels, *args, **kwargs):
    module = trw.layers.convs_3d(input_channels=parent.shape[1], channels=channels, *args, **kwargs)
    return simple_layers.SimpleModule(parent, module, shape=None)


def convs_2d(parent, channels, *args, **kwargs):
    module = trw.layers.convs_2d(input_channels=parent.shape[1], channels=channels, *args, **kwargs)
    return simple_layers.SimpleModule(parent, module, shape=None)
