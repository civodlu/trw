import torch.nn as nn
import numbers
from trw.layers.utils import div_shape
from trw.utils import flatten
from trw.layers.ops_conversion import OpsConversion


class ModulelWithIntermediate:
    """
    Represent a module with intermediate results
    """
    def forward_with_intermediate(self, x):
        raise NotImplemented()


class ConvsBase(nn.Module, ModulelWithIntermediate):
    """
    Generic group based convolutional network (e.g., VGG) for 2D or 3D CNN

    The following operations will take place:
        Conv group 1
            Conv 1, repeat 1
            Activation
            [...]
            Conv n, repeat n
            Activation
            Pooling
            LRN or BatchNorm
            Dropout
        [...]
        Conv group n
            [...]

    """
    def __init__(
            self,
            cnn_dim,
            input_channels,
            channels,
            convolution_kernels=5,
            strides=1,
            pooling_size=2,
            convolution_repeats=1,
            activation=nn.ReLU,
            with_flatten=False,
            dropout_probability=None,
            batch_norm_kwargs=None,
            lrn_kwargs=None,
            padding='same',
            last_layer_is_output=False,
            bias=True):
        """

        Args:
            cnn_dim: the dimension of the  CNN (2 for 2D or 3 for 3D)
            input_channels: the number of input channels
            channels: the number of channels for each convolutional layer
            convolution_kernels: for each convolution group, the kernel of the convolution
            strides: for each convolution group, the stride of the convolution
            pooling_size: the pooling size to be inserted after each convolution group
            convolution_repeats: the number of repeats of a convolution. ``1`` means no repeat.
            activation: the activation function
            with_flatten: if True, the last output will be flattened
            dropout_probability: if None, not dropout. Else the probability of dropout after each convolution
            batch_norm_kwargs: the batch norm kwargs. See the original torch functions for description. If None,
                no batch norm
            lrn_kwargs: the local response normalization kwargs. See the original torch functions for description. If
                None, not LRN
            padding: 'same' will add padding so that convolution output as the same size as input
            last_layer_is_output: if True, the last convolution will NOT have activation, dropout, batch norm, LRN
            bias: if ``True``, add a learnable bias to the convolution layer
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
        if not isinstance(pooling_size, list) and pooling_size is not None:
            pooling_size = [pooling_size] * nb_convs
        if isinstance(convolution_repeats, numbers.Number):
            convolution_repeats = [convolution_repeats] * nb_convs
        if isinstance(padding, numbers.Number):
            padding = [padding] * nb_convs
        elif isinstance(padding, str):
            pass
        else:
            assert len(padding) == nb_convs

        assert nb_convs == len(convolution_kernels), 'must be specified for each convolutional layer'
        assert nb_convs == len(strides), 'must be specified for each convolutional layer'
        assert pooling_size is None or nb_convs == len(pooling_size), 'must be specified for each convolutional layer'
        assert nb_convs == len(convolution_repeats)

        self.with_flatten = with_flatten
        with_batchnorm = batch_norm_kwargs is not None
        with_lrn = lrn_kwargs is not None

        # instantiate the net work. We do this as a nn.Module instead of simply a nn.Sequential
        # so that we can take advantage of the shared involving multiple layers in downstream tasks
        # for example in fully convolutional layers, we need to re-use intermediate results for the
        # pixel level segmentation that can't be achieved using nn.Sequential
        layers = nn.ModuleList()

        prev = input_channels
        for n in range(len(channels)):
            current = channels[n]
            currently_last_layer = n + 1 == len(channels)

            p = 0
            if padding == 'same':
                p = div_shape(convolution_kernels[n], 2)
            else:
                p = padding[n]

            ops = [ops_conv.conv_fn(prev, current, kernel_size=convolution_kernels[n], stride=strides[n], padding=p, bias=bias)]
            if not last_layer_is_output or not currently_last_layer:
                # only use the activation if not the last layer
                ops.append(activation())

            nb_repeats = convolution_repeats[n] - 1
            for r in range(nb_repeats):
                # do NOT apply strides in the repeat convolutions, else we will loose too quickly
                # the resolution
                if with_batchnorm:
                    ops.append(ops_conv.bn_fn(current, **batch_norm_kwargs))

                if with_lrn:
                    ops.append(lrn_fn(current, **lrn_kwargs))

                ops.append(ops_conv.conv_fn(current, current, kernel_size=convolution_kernels[n], stride=1, padding=p, bias=bias))

                if last_layer_is_output and currently_last_layer and r + 1 == nb_repeats:
                    # we don't want to add activation if the output is the last layer
                    break

                ops.append(activation())

            if not last_layer_is_output or not currently_last_layer:
                # if not, we are done here: do not add batch norm, LRN, dropout....
                if pooling_size is not None:
                    ops.append(ops_conv.pool_fn(pooling_size[n])),

                if with_batchnorm:
                    ops.append(ops_conv.bn_fn(current, **batch_norm_kwargs))

                if with_lrn:
                    ops.append(lrn_fn(current, **lrn_kwargs))

                if dropout_probability is not None:
                    ops.append(ops_conv.dropout_fn(p=dropout_probability))

            layers.append(nn.Sequential(*ops))
            prev = current
        self.layers = layers

    def forward_simple(self, x):
        for l in self.layers:
            x = l(x)

        if self.with_flatten:
            x = flatten(x)
        return x

    def forward_with_intermediate(self, x):
        r = []
        for l in self.layers:
            x = l(x)
            r.append(x)

        return r

    def forward(self, x):
        return self.forward_simple(x)
