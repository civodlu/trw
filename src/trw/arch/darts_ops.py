import torch.nn as nn
import collections


DARTS_PRIMITIVES_2D = collections.OrderedDict([
    ('none', lambda C, stride, affine: Zero2d(stride)),
    ('avg_pool_3x3', lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)),
    ('max_pool_3x3', lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1)),
    ('skip_connect', lambda C, stride, affine: Identity() if stride == 1 else ReduceChannels2d(C, C, affine=affine)),
    ('sep_conv_3x3', lambda C, stride, affine: SepConv2d(C, C, 3, stride, 1, affine=affine)),
    ('sep_conv_5x5', lambda C, stride, affine: SepConv2d(C, C, 5, stride, 2, affine=affine)),
    ('dil_conv_3x3', lambda C, stride, affine: DilConv2d(C, C, 3, stride, 2, 2, affine=affine)),
    ('dil_conv_5x5', lambda C, stride, affine: DilConv2d(C, C, 5, stride, 4, 2, affine=affine)),
])

    #'sep_conv_7x7': lambda C, stride, affine: SepConv2d(C, C, 7, stride, 3, affine=affine),
    #'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
    #    nn.ReLU(inplace=False),
    #    nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
    #    nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
    #    nn.BatchNorm2d(C, affine=affine)
    #),


class ReLUConvBN2d(nn.Module):
    """
    Stack of relu-conv-bn
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class ReduceChannels2d(nn.Module):
    # TODO revalidate results using this
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()

        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(C_in, C_out, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        return self.bn(self.conv(self.relu(x)))


class Identity(nn.Module):
    """
    Identity module
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero2d(nn.Module):
    """
    zero by stride
    """
    def __init__(self, stride):
        super().__init__()

        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class DilConv2d(nn.Module):
    """
    relu-dilated conv-bn
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        """

        :param C_in:
        :param C_out:
        :param kernel_size:
        :param stride:
        :param padding: 2/4
        :param dilation: 2
        :param affine:
        """
        super().__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv2d(nn.Module):
    """
    implemented separate convolution via pytorch groups parameters
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        """

        :param C_in:
        :param C_out:
        :param kernel_size:
        :param stride:
        :param padding: 1/2
        :param affine:
        """
        super().__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


"""
class FactorizedReduce2d(nn.Module):
    # TODO remove
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()

        assert C_out % 2 == 0

        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)

        # x: torch.Size([32, 32, 32, 32])
        # conv1: [b, c_out//2, d//2, d//2]
        # conv2: []
        # out: torch.Size([32, 32, 16, 16])

        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
"""