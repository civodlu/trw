from trw.layers import EfficientNet, default_layer_config, MBConvN
import torch
from torch import nn


def test():
    config = default_layer_config(dimensionality=2, activation=torch.nn.SiLU)

    torch.random.manual_seed(0)
    i = torch.randn([4, 3, 64, 224])

    mbconv = MBConvN(
        config=config,
        input_channels=3,
        output_channels=10,
        expansion_factor=16,
        kernel_size=3,
        stride=2,
        p=0.1
    )

    o_mbconv = mbconv(i)

    net = EfficientNet(dimensionality=2, input_channels=3, output_channels=1000, config=config)
    o = net.feature_extractor(i)
    print('DONE')

test()