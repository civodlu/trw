import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trw.layers.ops_conversion import OpsConversion


class DoubleConv(nn.Module):
    def __init__(self, dim, in_channels, out_channels, activation_fn):
        super().__init__()

        ops = OpsConversion(dim)

        self.double_conv = nn.Sequential(
            ops.conv_fn(in_channels, out_channels, kernel_size=3, padding=1),
            ops.bn_fn(out_channels),
            activation_fn(),
            ops.conv_fn(out_channels, out_channels, kernel_size=3, padding=1),
            ops.bn_fn(out_channels),
            activation_fn()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, bloc_level, dim, in_channels, out_channels, activation_fn):
        super().__init__()

        ops = OpsConversion(dim)

        self.maxpool_conv = nn.Sequential(
            ops.pool_fn(2),
            DoubleConv(dim, in_channels, out_channels, activation_fn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, bloc_level, dim, in_channels, out_channels, bilinear, activation_fn):
        super().__init__()

        ops = OpsConversion(dim)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = ops.upsample_fn(scale_factor=2, align_corners=True)
        else:
            self.up = ops.decon_fn(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(dim, in_channels, out_channels, activation_fn)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff = np.asarray(x2.size()[2:]) - np.asarray(x1.size()[2:])
        diff = diff[::-1]
        pad_min = diff // 2
        pad_max = diff - diff // 2
        padding = np.stack((pad_min, pad_max), axis=1).ravel().tolist()

        x1 = F.pad(x1, padding)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, dim, in_channels, out_channels):
        super(OutConv, self).__init__()
        ops = OpsConversion(dim)
        self.conv = ops.conv_fn(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Configurable UNet-like architecture
    """
    def __init__(
            self,
            dim,
            input_channels,
            n_outputs,
            linear_upsampling=False,
            base_filters=64,
            nb_blocs=4,
            down_block_fn=Down,
            up_block_fn=Up,
            out_block_fn=OutConv,
            activation_fn=nn.ReLU,
    ):
        """

        Args:
            dim: the dimensionality of the UNet (e.g., dim=2 for 2D images)
            input_channels: the number of channels of the input
            n_outputs: the number of channels of the output
            linear_upsampling: if True, use linear up-sampling instead of transposed convolutions
            base_filters: the number of channels to be used as base. This number is multiplied
                by 2 every time we downsample a layer
            nb_blocs: the number of downsampled layers
            down_block_fn: a function taking (dim, in_channels, out_channels) and returning a nn.Module
            up_block_fn: a function taking (dim, in_channels, out_channels, bilinear) and returning a nn.Module
            out_block_fn: a function taking (dim, in_channels, out_channels) and returning a nn.Module
            activation_fn: the activation to be used for up and down convolution, except the final output
        """
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.n_outputs = n_outputs
        self.linear_upsampling = linear_upsampling
        self.inc = DoubleConv(dim, input_channels, base_filters, activation_fn)
        self.downs = nn.ModuleList()
        self.dim = dim

        current = base_filters
        for i in range(nb_blocs):
            if i + 1 < nb_blocs:
                next = current * 2
            else:
                next = current

            self.downs.append(down_block_fn(i, dim, current,  next, activation_fn))
            current = next

        current = 2 * current
        next = current // 4
        self.ups = nn.ModuleList()
        for i in range(nb_blocs):
            self.ups.append(up_block_fn(i, dim, current, next, linear_upsampling, activation_fn))
            current = current // 2
            if i + 2 < nb_blocs:
                next = next // 2
            else:
                next = next
        self.outc = out_block_fn(dim, next, n_outputs)

    def forward(self, x):
        """
        Args:
            x: the input image
        """
        x_n = [self.inc(x)]
        prev = x_n[0]
        for i in range(len(self.downs)):
            current = self.downs[i](prev)
            x_n.append(current)
            prev = current

        nb_layers = len(self.ups)
        x = x_n[nb_layers - 1]

        #assert latent is None or latent.shape[0] == x.shape[0], f'latent N mismatch. ' \
        #                                                        f'Got={latent.shape[0]}, expected={x.shape[0]}'
        #assert (latent is None and self.latent_shape is None) or \
        #    (latent is not None and self.latent_shape is not None), f'latent mismatch. ' \
        #                                                            f'Expected={self.latent_shape[1:]}, Got={latent}'

        #if latent is not None:
        #    # if we have a latent, make sure it has the same size as the last embedding
        #    # if not, resize it!
        #    assert len(latent.shape) == self.dim + 2
        #    assert latent.shape[1] == self.latent_shape[0]
        #    if latent.shape[2:] != x.shape[2:]:
        #        # we need to resize the latent variable
        #        latent = upsample(latent, x.shape[2:], mode='nearest')

        #    # the latent is simply concatenated to the embedding
        #    x = torch.cat((x, latent), dim=1)

        for i in range(nb_layers):
            x = self.ups[i](x, x_n[nb_layers - 1 - i])
        logits = self.outc(x)
        return logits
