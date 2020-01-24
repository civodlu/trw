import torch.nn as nn
import functools


class OpsConversion:
    """
    Helper to create standard N-d operations
    """
    def __init__(self, dim):
        if dim == 3:
            self.conv_fn = nn.Conv3d
            self.decon_fn = nn.ConvTranspose3d
            self.pool_fn = nn.MaxPool3d
            self.bn_fn = nn.BatchNorm3d
            self.dropout_fn = nn.Dropout3d
            self.updample_fn = functools.partial(nn.Upsample, mode='trilinear')
        elif dim == 2:
            self.conv_fn = nn.Conv2d
            self.decon_fn = nn.ConvTranspose2d
            self.pool_fn = nn.MaxPool2d
            self.bn_fn = nn.BatchNorm2d
            self.dropout_fn = nn.Dropout2d
            self.updample_fn = functools.partial(nn.Upsample, mode='bilinear')
        else:
            raise NotImplemented()
