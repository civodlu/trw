import torch.nn as nn
import functools


class OpsConversion:
    """
    Helper to create standard N-d operations
    """
    def __init__(self):
        self.dim = None
        self.group_norm_fn = None
        try:
            self.group_norm_fn = nn.GroupNorm
        except:
            pass

        self.sync_bn_fn = None
        try:
            self.sync_bn_fn = nn.SyncBatchNorm
        except:
            pass

        self.layer_norm = None
        try:
            self.layer_norm = nn.LayerNorm
        except:
            pass

        self.lrn_fn = nn.LocalResponseNorm

        self.conv_fn = None
        self.decon_fn = None

        self.max_pool_fn = None
        self.avg_pool_fn = None
        self.fractional_max_pool_fn = None
        self.adaptative_max_pool_fn = None
        self.adaptative_avg_pool_fn = None

        self.dropout_fn = None
        self.dropout1d_fn = nn.Dropout

        self.alpha_dropout = None
        try:
            self.alpha_dropout = nn.AlphaDropout
        except:
            pass

        self.upsample_fn = None
        self.instance_norm = None
        self.bn_fn = None

    def set_dim(self, dim: int):
        self.dim = dim

        if dim == 3:
            self.conv_fn = nn.Conv3d
            self.decon_fn = nn.ConvTranspose3d
            self.max_pool_fn = nn.MaxPool3d
            self.avg_pool_fn = nn.AvgPool3d

            self.fractional_max_pool_fn = None
            try:
                self.fractional_max_pool_fn = nn.FractionalMaxPool3d
            except:
                pass

            self.adaptative_max_pool_fn = nn.AdaptiveMaxPool3d
            self.adaptative_avg_pool_fn = nn.AdaptiveAvgPool3d
            self.dropout_fn = nn.Dropout3d
            self.upsample_fn = functools.partial(nn.Upsample, mode='trilinear')

            self.instance_norm = nn.InstanceNorm3d
            self.bn_fn = nn.BatchNorm3d

        elif dim == 2:
            self.conv_fn = nn.Conv2d
            self.decon_fn = nn.ConvTranspose2d
            self.max_pool_fn = nn.MaxPool2d
            self.avg_pool_fn = nn.AvgPool2d

            self.fractional_max_pool_fn = None
            try:
                self.fractional_max_pool_fn = nn.FractionalMaxPool2d
            except:
                pass

            self.adaptative_max_pool_fn = nn.AdaptiveMaxPool2d
            self.adaptative_avg_pool_fn = nn.AdaptiveAvgPool2d
            self.dropout_fn = nn.Dropout2d
            self.upsample_fn = functools.partial(nn.Upsample, mode='bilinear')
            self.instance_norm = nn.InstanceNorm2d
            self.bn_fn = nn.BatchNorm2d

        elif dim == 1:
            self.conv_fn = nn.Conv1d
            self.decon_fn = nn.ConvTranspose1d
            self.max_pool_fn = nn.MaxPool1d
            self.avg_pool_fn = nn.AvgPool1d

            self.fractional_max_pool_fn = None
            self.adaptative_max_pool_fn = nn.AdaptiveMaxPool1d
            self.adaptative_avg_pool_fn = nn.AdaptiveAvgPool1d
            self.dropout_fn = nn.Dropout
            self.upsample_fn = functools.partial(nn.Upsample, mode='linear')
            self.instance_norm = nn.InstanceNorm1d
            self.bn_fn = nn.BatchNorm1d

        else:
            raise NotImplemented()
