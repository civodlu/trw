import trw.layers

from trw.simple_layers import simple_layers


def convs_3d(parent, channels, *args, **kwargs):
    full_channels = [parent.shape[1]] + channels
    module = trw.layers.convs_3d(channels=full_channels, *args, **kwargs)
    return simple_layers.SimpleModule(parent, module, shape=None)


def convs_2d(parent, channels, *args, **kwargs):
    full_channels = [parent.shape[1]] + channels
    module = trw.layers.convs_2d(channels=full_channels, *args, **kwargs)
    return simple_layers.SimpleModule(parent, module, shape=None)
