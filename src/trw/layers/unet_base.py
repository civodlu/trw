import torch
import torch.nn as nn
from trw.layers.ops_conversion import OpsConversion
from trw.utils import upsample


class Down(nn.Module):
    def __init__(self, bloc_level, dim, in_channels, out_channels, activation_fn):
        super().__init__()

        ops = OpsConversion(dim)
        self.ops = nn.Sequential(
            ops.conv_fn(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            ops.bn_fn(out_channels),
            activation_fn()
        )

    def forward(self, x):
        return self.ops(x)


class Up(nn.Module):
    def __init__(self, bloc_level, dim, skip_channels, in_channels, out_channels, activation_fn):
        super().__init__()
        ops = OpsConversion(dim)

        self.ops_deconv = nn.Sequential(
            ops.decon_fn(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            ops.bn_fn(in_channels // 2),
            activation_fn(),
        )

        self.ops_conv = nn.Sequential(
            ops.conv_fn(in_channels // 2 + skip_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ops.bn_fn(out_channels),
            activation_fn(),
        )

        self.skip_channels = skip_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, skip, previous):
        assert skip.shape[1] == self.skip_channels
        assert previous.shape[1] == self.in_channels
        x = self.ops_deconv(previous)
        assert x.shape == skip.shape
        x = torch.cat([skip, x], dim=1)
        x = self.ops_conv(x)
        return x


class ConvBnActivation(nn.Module):
    def __init__(self, dim, in_channels, out_channels, activation_fn=nn.ReLU, kernel_size=3, stride=1, padding=1):
        super().__init__()
        ops = OpsConversion(dim)

        self.ops_conv = nn.Sequential(
            ops.conv_fn(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            ops.bn_fn(out_channels),
            activation_fn(),
        )

    def forward(self, x):
        return self.ops_conv(x)


class ConvBn(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        ops = OpsConversion(dim)

        self.ops_conv = nn.Sequential(
            ops.conv_fn(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            ops.bn_fn(out_channels),
        )

    def forward(self, x):
        return self.ops_conv(x)


class LatentConv(nn.Module):
    """
    Concatenate a latent variable (possibly resized to the input shape) and apply a convolution
    """
    def __init__(
            self,
            bloc_level,
            dim,
            input_channels,
            output_channels,
            activation_fn=nn.ReLU,
            latent_channels=None,
            kernel_size=3,
            stride=1,
            padding=1):
        super().__init__()
        self.latent_channels = latent_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation_fn = activation_fn
        self.dim = dim

        if latent_channels is None:
            latent_channels = 0

        self.ops = ConvBnActivation(dim, input_channels + latent_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, latent=None):
        if latent is not None:
            assert latent.shape[1] == self.latent_channels
            assert len(latent.shape) == self.dim + 2
            if latent.shape[2:] != x.shape[2:]:
                # we need to resize the latent variable
                latent = upsample(latent, x.shape[2:], mode='linear')
            x = torch.cat([latent, x], dim=1)

        return self.ops(x)


class UNetBase(nn.Module):
    """
    Configurable UNet-like architecture
    """
    def __init__(
            self,
            dim,
            input_channels,
            channels,
            output_channels,
            down_block_fn=Down,
            up_block_fn=Up,
            init_block_fn=ConvBnActivation,
            middle_block_fn=LatentConv,
            output_block_fn=ConvBn,
            activation_fn=nn.LeakyReLU,
            init_block_channels=None,
            latent_channels=None,
    ):
        """

        The original UNet architecture is decomposed in 5 blocks:
            - init_block_fn: will first be applied on the input
            - down_block_fn: the encoder
            - middle_block_fn: the junction between the encoder and decoder
            - up_block_fn: the decoder
            - output_block_fn: the output layer

        Args:
            dim: the dimensionality of the UNet (e.g., dim=2 for 2D images)
            input_channels: the number of channels of the input
            output_channels: the number of channels of the output
            down_block_fn: a function taking (dim, in_channels, out_channels) and returning a nn.Module
            up_block_fn: a function taking (dim, in_channels, out_channels, bilinear) and returning a nn.Module
            activation_fn: the activation to be used for up and down convolution, except the final output
            init_block_channels: the number of channels to be used by the init block. If `None`, `channels[0] // 2`
                will be used
        """
        assert len(channels) >= 1

        super().__init__()
        self.dim = dim
        self.input_channels = input_channels
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.output_channels = output_channels
        self.latent_channels = latent_channels
        self.activation_fn = activation_fn

        self.init_block = None
        self.downs = None
        self.ups = None
        self.middle_block = None

        if latent_channels is not None:
            assert middle_block_fn is not None

        self.build(init_block_fn, down_block_fn, up_block_fn, middle_block_fn, output_block_fn)

    def build(self, init_block_fn, down_block_fn, up_block_fn, middle_block_fn, output_block_fn):
        self.downs = nn.ModuleList()
        skip_channels = None

        if self.init_block_channels is None:
            out_init_channels = self.channels[0] // 2
        else:
            out_init_channels = self.init_block_channels
        self.init_block = init_block_fn(self.dim, self.input_channels, out_init_channels)

        # down blocks
        input_channels = out_init_channels
        for i, skip_channels in enumerate(self.channels):
            self.downs.append(down_block_fn(i, self.dim, input_channels, skip_channels, self.activation_fn))
            input_channels = skip_channels

        # middle blocks
        if middle_block_fn is not None:
            self.middle_block = middle_block_fn(
                len(self.channels),
                self.dim,
                skip_channels,
                skip_channels,
                self.activation_fn,
                latent_channels=self.latent_channels
            )
        else:
            self.middle_block = None

        # up blocks
        input_channels = skip_channels
        self.ups = nn.ModuleList()
        for i in range(len(self.channels)):
            if i + 1 == len(self.channels):
                # last level: use the given number of output channels
                skip_channels = out_init_channels
                out_channels = skip_channels
            else:
                # previous layer
                skip_channels = self.channels[-(i + 2)]
                out_channels = skip_channels

            self.ups.append(up_block_fn(
                len(self.channels) - i - 1,
                self.dim,
                skip_channels=skip_channels,
                in_channels=input_channels,
                out_channels=out_channels,
                activation_fn=self.activation_fn))
            input_channels = skip_channels

        # here we need to have a special output block: this
        # is because we do NOT want to add the activation for the
        # result layer (i.e., often, the output is normalized [-1, 1]
        # and we would discard the negative portion)
        self.output = output_block_fn(self.dim, out_init_channels, self.output_channels)

    def forward(self, x, latent=None):
        """
        Args:
            x: the input image
            latent: a latent variable appended by the middle block
        """
        prev = self.init_block(x)
        x_n = [prev]
        for down in self.downs:
            current = down(prev)
            x_n.append(current)
            prev = current

        if self.latent_channels is not None:
            assert latent is not None, f'missing latent variable! (latent_channels={self.latent_channels})'
            assert latent.shape[1] == self.latent_channels, f'incorrect latent. Got={latent.shape[1]}, ' \
                                                            f'expected={self.latent_channels}'

        if self.middle_block is not None:
            prev = self.middle_block(x_n[-1], latent=latent)
        else:
            prev = x_n[-1]

        for skip, up in zip(reversed(x_n[:-1]), self.ups):
            prev = up(skip, prev)

        return self.output(prev)


