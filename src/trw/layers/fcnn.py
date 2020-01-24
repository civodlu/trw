import torch
import torch.nn as nn
from trw.layers.utils import div_shape
from trw.layers.convs import ModulelWithIntermediate
import collections
from trw.layers.ops_conversion import OpsConversion


class FullyConvolutional(nn.Module):
    """
    Construct a Fully Convolutional Neural network from a base model. This provides pixel level interpolation

    Example of a 2D network taking 1 input channel with 3 convolutions (16, 32, 64) and 3 deconvolutions (32, 16, 8):
    >>> import torch
    >>> import trw
    >>> convs = trw.layers.ConvsBase(cnn_dim=2, input_channels=1, channels=[16, 32, 64])
    >>> fcnn = trw.layers.FullyConvolutional(cnn_dim=2, base_model=convs, deconv_filters=[64, 32, 16, 8], convolution_kernels=7, strides=[2] * 3, nb_classes=2)
    >>> i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
    >>> o = fcnn(i)

    The following intermediate data will be created (concat_mode='add'):
    input = [None, 1, 32, 32]
    conv_1 = [None, 16, 16, 16]
    conv_2 = [None, 32, 8, 8]
    conv_3 = [None, 64, 4, 4]

    deconv_1 = [None, 32, 8, 8]
    deconv_2 = [None, 16, 16, 16]
    deconv_3 = [None, 8, 32, 32]
    classifier = [None, 2, 32, 32]
    """

    def __init__(self, cnn_dim, base_model, deconv_filters, convolution_kernels, strides, activation=nn.ReLU, batch_norm_kwargs={}, nb_classes=None, output_paddings=1, concat_mode='add', conv_filters=None):
        """

        Args:
            base_model: a base model. Must be an instance of :class:`trw.layers.ModulelWithIntermediate`. This
                model will return intermediate results of convolutional groups
            deconv_filters: the number of filters of the deconvolutional layers. Specified from the model output
                back to the input. It must have "intermediate results" + 1 filters. the first "intermediate results"
                filters must match the dimension of the intermediate results shape. Only the last one can be freely
                specified.

            convolution_kernels (list, int): the kernel sizes used by the deconvolutional layers. Can be an int or a list
            strides (list, int): the strides used by the deconvolutional layers. Can be an int or a list.
            output_paddings (list, int): padding added post deconvolution
            concat_mode: one of 'add' or 'concatenate'. If 'add', the skip connection will be added to the
                corresponding upconvolution (i.e., similar to the FCNN paper). If 'concatenate', the channels will
                be concatenated instead of added (i.e., similar to UNET)
            conv_filters: if concat_mode == 'concatenate', we MUST have the shapes of the intermediates
        """
        super().__init__()
        self.base_model = base_model
        self.deconv_filters = deconv_filters
        self.nb_classes = nb_classes
        self.concat_mode = concat_mode
        self.conv_filters = conv_filters

        # normalize the arguments
        nb_convs = len(deconv_filters) - 1
        if not isinstance(convolution_kernels, collections.Sequence):
            convolution_kernels = [convolution_kernels] * nb_convs
        if not isinstance(strides, collections.Sequence):
            strides = [strides] * nb_convs
        if not isinstance(output_paddings, collections.Sequence):
            output_paddings = [output_paddings] * nb_convs
        assert isinstance(base_model, ModulelWithIntermediate), 'it must be a model that returns intermediate results!'
        assert concat_mode in ('add', 'concatenate')
        if concat_mode == 'concatenate':
            assert conv_filters is not None

        # generic 2D/3D or nD once pytorch supports it
        ops_conv = OpsConversion(cnn_dim)

        assert len(convolution_kernels) == len(strides)
        assert len(convolution_kernels) == len(output_paddings)
        assert len(convolution_kernels) + 1 == len(deconv_filters), 'the last `deconv_filters` should be the ' \
                                                                    'number of output classes'

        groups_deconv = nn.ModuleList()
        prev_filter = deconv_filters[0]

        filters = deconv_filters[1:]
        for layer_n in range(len(filters)):
            current_filter = filters[layer_n]
            kernel = convolution_kernels[layer_n]
            stride = strides[layer_n]
            output_padding = output_paddings[layer_n]
            padding = div_shape(kernel, 2)

            ops = []
            ops.append(ops_conv.decon_fn(in_channels=prev_filter, out_channels=current_filter, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding))
            ops.append(activation())
            ops.append(ops_conv.bn_fn(current_filter, **batch_norm_kwargs))
            groups_deconv.append(nn.Sequential(*ops))

            if concat_mode == 'concatenate' and layer_n + 1 < len(conv_filters):
                current_filter += conv_filters[::-1][layer_n + 1]

            prev_filter = current_filter

        self.deconvolutions = groups_deconv

        if nb_classes is not None:
            self.classifier = ops_conv.conv_fn(deconv_filters[-1], nb_classes, kernel_size=1)

    def forward(self, x):
        score, _ = self.forward_with_intermediate(x)
        return score

    def forward_with_intermediate(self, x):
        intermediates_orig = self.base_model.forward_with_intermediate(x)

        if self.conv_filters is not None:
            assert len(self.conv_filters) == len(intermediates_orig)
            for layer_n, layer in enumerate(intermediates_orig):
                assert layer.shape[1] == self.conv_filters[layer_n], \
                    f'expected intermediate filters={self.conv_filters[layer_n]}, got={layer.shape[1]}'

        intermediates = list(reversed(intermediates_orig))  # make sure the intermediate are ordered last->first layer

        assert len(intermediates) + 1 == len(self.deconv_filters),\
            f'unexpected number of intermediate (N={len(intermediates)} ' \
            f'vs convolution groups (N={len(self.deconvolutions)}! ' \
            f'Expected=number of intermediate + 1 == convolution groups'

        assert intermediates[0].shape[1] == self.deconv_filters[0], f'expected output filters=' \
                                                                    f'{self.deconv_filters[0]}, ' \
                                                                    f'got={intermediates[0].shape[1]}'

        score = None
        for n in range(len(self.deconvolutions)):
            if n == 0:
                score = self.deconvolutions[n](intermediates[0])
            else:
                score = self.deconvolutions[n](score)

            if n + 1 != len(self.deconvolutions):
                # the last deconvolution is for the number of classes
                if self.concat_mode == 'add':
                    score += intermediates[n + 1]
                elif self.concat_mode == 'concatenate':
                    score = torch.cat([intermediates[n + 1], score], dim=1)
                else:
                    raise NotImplemented(f'mode={self.concat_mode}')

        if self.nb_classes is not None:
            score = self.classifier(score)

        return score, intermediates_orig
