import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trw.layers.ops_conversion import OpsConversion


class DoubleConv(nn.Module):
    def __init__(self, dim, in_channels, out_channels):
        super().__init__()

        ops = OpsConversion(dim)

        self.double_conv = nn.Sequential(
            ops.conv_fn(in_channels, out_channels, kernel_size=3, padding=1),
            ops.bn_fn(out_channels),
            nn.ReLU(inplace=True),
            ops.conv_fn(out_channels, out_channels, kernel_size=3, padding=1),
            ops.bn_fn(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, dim, in_channels, out_channels):
        super().__init__()

        ops = OpsConversion(dim)

        self.maxpool_conv = nn.Sequential(
            ops.pool_fn(2),
            DoubleConv(dim, in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, dim, in_channels, out_channels, bilinear=True):
        super().__init__()

        ops = OpsConversion(dim)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = ops.updample_fn(scale_factor=2, align_corners=True)
        else:
            self.up = ops.decon_fn(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(dim, in_channels, out_channels)

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
    def __init__(self, dim, input_channels, n_classes, linear_upsampling=False, base_filters=64, nb_blocs=4):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.linear_upsampling = linear_upsampling
        self.inc = DoubleConv(dim, input_channels, base_filters)
        self.downs = nn.ModuleList()

        current = base_filters
        for i in range(nb_blocs):
            if i + 1 < nb_blocs:
                next = current * 2
            else:
                next = current

            self.downs.append(Down(dim, current,  next))
            current = next

        current = 2 * current
        next = current // 4
        self.ups = nn.ModuleList()
        for i in range(nb_blocs):
            self.ups.append(Up(dim, current, next, linear_upsampling))
            current = current // 2
            if i + 2 < nb_blocs:
                next = next // 2
            else:
                next = next
        self.outc = OutConv(dim, next, n_classes)

    def forward(self, x):
        x_n = [self.inc(x)]
        prev = x_n[0]
        for i in range(len(self.downs)):
            current = self.downs[i](prev)
            x_n.append(current)
            prev = current

        nb_layers = len(self.ups)
        x = x_n[nb_layers -1]
        for i in range(nb_layers):
            x = self.ups[i](x, x_n[nb_layers - 1 - i])
        logits = self.outc(x)
        return logits
