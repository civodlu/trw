import collections
import numbers

import torch.nn as nn
from trw.layers import OpsConversion, div_shape, ModulelWithIntermediate


class ConvsTransposeBase(nn.Module, ModulelWithIntermediate):
    """
    Helper class to create sequence of transposed convolution

    This can be used to map an embedding back to image space.
    """
    def __init__(
            self,
            cnn_dim,
            input_channels,
            channels,
            convolution_kernels=5,
            strides=1,
            paddings=None,
            activation=nn.ReLU,
            dropout_probability=None,
            batch_norm_kwargs=None,
            lrn_kwargs=None,
            last_layer_is_output=False,
            squash_function=None):
        """

        Args:
            cnn_dim: the dimension of the  CNN (2 for 2D or 3 for 3D)
            input_channels: the number of input channels
            channels: the number of channels for each convolutional layer
            convolution_kernels: for each convolution group, the kernel of the convolution
            strides: for each convolution group, the stride of the convolution
            dropout_probability: if None, not dropout. Else the probability of dropout after each convolution
            batch_norm_kwargs: the batch norm kwargs. See the original torch functions for description. If None,
                no batch norm
            lrn_kwargs: the local response normalization kwargs. See the original torch functions for description. If
                None, not LRN
            last_layer_is_output: if True, the last convolution will NOT have activation, dropout, batch norm, LRN
            squash_function: a function to be applied on the reconstuction. It is common to apply
                for example ``torch.sigmoid``. If ``None``, no function applied
            paddings: the paddings added. If ``None``, half the convolution kernel will be used.
        """
        super().__init__()

        ops_conv = OpsConversion(cnn_dim)
        lrn_fn = nn.LocalResponseNorm

        # normalize the arguments
        nb_convs = len(channels)
        if not isinstance(convolution_kernels, list):
            convolution_kernels = [convolution_kernels] * nb_convs
        if not isinstance(strides, list):
            strides = [strides] * nb_convs
        if paddings is None:
            paddings = [div_shape(kernel, 2) for kernel in convolution_kernels]
        elif isinstance(paddings, numbers.Integral):
            paddings = [paddings] * nb_convs
        else:
            assert isinstance(paddings, collections.Sequence) and len(paddings) == nb_convs

        assert nb_convs == len(convolution_kernels), 'must be specified for each convolutional layer'
        assert nb_convs == len(strides), 'must be specified for each convolutional layer'
        with_batchnorm = batch_norm_kwargs is not None
        with_lrn = lrn_kwargs is not None

        layers = nn.ModuleList()

        prev = input_channels
        for n in range(len(channels)):
            current = channels[n]
            currently_last_layer = n + 1 == len(channels)

            p = paddings[n]
            ops = [ops_conv.decon_fn(
                prev,
                current,
                kernel_size=convolution_kernels[n],
                stride=strides[n],
                padding=p,
                output_padding=0)]

            if not last_layer_is_output or not currently_last_layer:
                # only use the activation if not the last layer
                ops.append(activation())

            if not last_layer_is_output or not currently_last_layer:
                # if not, we are done here: do not add batch norm, LRN, dropout....
                if with_batchnorm:
                    ops.append(ops_conv.bn_fn(current, **batch_norm_kwargs))

                if with_lrn:
                    ops.append(lrn_fn(current, **lrn_kwargs))

                if dropout_probability is not None:
                    ops.append(ops_conv.dropout_fn(p=dropout_probability))

            layers.append(nn.Sequential(*ops))
            prev = current
        self.layers = layers
        self.squash_function = squash_function

    def forward_with_intermediate(self, x):
        r = []
        for layer in self.layers:
            x = layer(x)
            r.append(x)

        if self.squash_function is not None:
            r[-1] = self.squash_function(r[-1])
        return r

    def forward_simple(self, x):
        for layer in self.layers:
            x = layer(x)

        if self.squash_function is not None:
            return self.squash_function(x)
        return x

    def forward(self, x):
        return self.forward_simple(x)
