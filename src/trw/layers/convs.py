import torch.nn as nn
import numbers
from trw.layers.utils import div_shape
from trw.layers.flatten import flatten


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
            last_layer_is_output=False):
        """

        Args:
            cnn_dim: the dimension of the  CNN (2 for 2D or 3 for 3D)
            channels: the number of channels
            convolution_kernels: for each convolution group, the kernel of the convolution
            strides: for each convolution group, the stride of the convolution
            pooling_size: the pooling size to be inserted after each convolution group
            convolution_repeats: the number of repeats of a convolution
            activation: the activation function
            with_flatten: if True, the last output will be flattened
            dropout_probability: if None, not dropout. Else the probability of dropout after each convolution
            batch_norm_kwargs: the batch norm kwargs. See the original torch functions for description. If None,
                no batch norm
            lrn_kwargs: the local response normalization kwargs. See the original torch functions for description. If
                None, not LRN
            padding: 'same' will add padding so that convolution output as the same size as input
            last_layer_is_output: if True, the last convolution will NOT have activation, dropout, batch norm, LRN
        """
        super().__init__()

        lrn_fn = nn.LocalResponseNorm
        if cnn_dim == 3:
            conv_fn = nn.Conv3d
            pool_fn = nn.MaxPool3d
            bn_fn = nn.BatchNorm3d
            dropout_fn = nn.Dropout3d
        elif cnn_dim == 2:
            conv_fn = nn.Conv2d
            pool_fn = nn.MaxPool2d
            bn_fn = nn.BatchNorm2d
            dropout_fn = nn.Dropout2d
        else:
            raise NotImplemented()

        # normalize the arguments
        nb_convs = len(channels) - 1
        if not isinstance(convolution_kernels, list):
            convolution_kernels = [convolution_kernels] * nb_convs
        if not isinstance(strides, list):
            strides = [strides] * nb_convs
        if not isinstance(pooling_size, list) and pooling_size is not None:
            pooling_size = [pooling_size] * nb_convs
        if isinstance(convolution_repeats, numbers.Number):
            convolution_repeats = [convolution_repeats] * nb_convs

        assert nb_convs == len(convolution_kernels), 'must be specified for each convolutional layer'
        assert nb_convs == len(strides), 'must be specified for each convolutional layer'
        assert pooling_size is None or nb_convs == len(pooling_size), 'must be specified for each convolutional layer'

        self.with_flatten = with_flatten
        with_batchnorm = batch_norm_kwargs is not None
        with_lrn = lrn_kwargs is not None

        # instantiate the net work. We do this as a nn.Module instead of simply a nn.Sequential
        # so that we can take advantage of the shared involving multiple layers in downstream tasks
        # for example in fully convolutional layers, we need to re-use intermediate results for the
        # pixel level segmentation that can't be achieved using nn.Sequential
        layers = nn.ModuleList()

        for n in range(len(channels) - 1):
            current = channels[n]
            next = channels[n + 1]
            currently_last_layer = n + 2 == len(channels)

            p = 0
            if padding == 'same':
                p = div_shape(convolution_kernels[n], 2)

            ops = [conv_fn(current, next, kernel_size=convolution_kernels[n], stride=strides[n], padding=p)]
            if not last_layer_is_output or not currently_last_layer:
                # only use the activation if not the last layer
                ops.append(activation())

            nb_repeats = convolution_repeats[n] - 1
            for r in range(nb_repeats):
                ops.append(conv_fn(next, next, kernel_size=convolution_kernels[n], stride=strides[n], padding=p))

                if last_layer_is_output and currently_last_layer and r + 1 == nb_repeats:
                    # we don't want to add activation if the output is the last layer
                    break
                ops.append(activation())

            if last_layer_is_output and currently_last_layer:
                # we are done here: do not add batch norm, LRN, dropout....
                break

            if pooling_size is not None:
                ops.append(pool_fn(pooling_size[n])),

            if with_batchnorm:
                ops.append(bn_fn(next, **batch_norm_kwargs))

            if with_lrn:
                ops.append(lrn_fn(next, **lrn_kwargs))

            if dropout_probability is not None:
                ops.append(dropout_fn(p=dropout_probability))

            layers.append(nn.Sequential(*ops))
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
