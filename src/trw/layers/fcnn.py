import torch.nn as nn
from trw.layers.utils import div_shape
from trw.layers.convs import ModulelWithIntermediate


class FullyConvolutional(nn.Module):
    def __init__(self, cnn_dim, base_model, deconv_filters, convolution_kernels, strides, activation=nn.ReLU, batch_norm_kwargs={}, nb_classes=None, output_paddings=1):
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
        """
        super().__init__()
        self.base_model = base_model
        self.deconv_filters = deconv_filters
        self.nb_classes = nb_classes

        # normalize the arguments
        nb_convs = len(deconv_filters) - 1
        if not isinstance(convolution_kernels, list):
            convolution_kernels = [convolution_kernels] * nb_convs
        if not isinstance(strides, list):
            strides = [strides] * nb_convs
        if not isinstance(output_paddings, list):
            output_paddings = [output_paddings] * nb_convs
        assert isinstance(base_model, ModulelWithIntermediate), 'it must be a model that returns intermediate results!'

        # generic 2D/3D or nD once pytorch supports it
        if cnn_dim == 3:
            decon_fn = nn.ConvTranspose3d
            bn_fn = nn.BatchNorm3d
        elif cnn_dim == 2:
            decon_fn = nn.ConvTranspose2d
            bn_fn = nn.BatchNorm2d
        else:
            raise NotImplemented()

        assert len(convolution_kernels) == len(strides)
        assert len(convolution_kernels) == len(output_paddings)
        assert len(convolution_kernels) + 1 == len(deconv_filters), 'the last `deconv_filters` should be the ' \
                                                                    'number of output classes'

        groups_deconv = nn.ModuleList()
        for deconv_layer, filter in enumerate(deconv_filters[:-1]):
            next_filter = deconv_filters[deconv_layer + 1]
            kernel = convolution_kernels[deconv_layer]
            stride = strides[deconv_layer]
            output_padding = output_paddings[deconv_layer]
            padding = div_shape(kernel, 2)

            ops = []
            ops.append(decon_fn(in_channels=filter, out_channels=next_filter, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding))
            ops.append(activation())
            ops.append(bn_fn(next_filter, **batch_norm_kwargs))
            groups_deconv.append(nn.Sequential(*ops))

        self.deconvolutions = groups_deconv

        if nb_classes is not None:
            self.classifier = nn.Conv2d(deconv_filters[-1], nb_classes, kernel_size=1)

    def forward(self, x):
        intermediates = self.base_model.forward_with_intermediate(x)
        intermediates = list(reversed(intermediates))  # make sure the intermediate are ordered last->first layer

        assert len(intermediates) + 1 == len(self.deconv_filters),\
            f'unexpected number of intermediate (N={len(intermediates)} ' \
            f'vs convolution groups (N={len(self.deconvolutions)}! ' \
            f'Expected=number of intermediate + 1 == convolution groups'

        assert intermediates[0].shape[1] == self.deconv_filters[0], f'expected output filters=' \
                                                                    f'{self.cnn_filter_output}, ' \
                                                                    f'got={intermediates[-1].shape[1]}'

        score = None
        for n in range(len(self.deconvolutions)):
            if n == 0:
                score = self.deconvolutions[n](intermediates[0])
            else:
                score = self.deconvolutions[n](score)

            if n + 1 != len(self.deconvolutions):
                # the last deconvolution is for the number of classes
                score += intermediates[n + 1]

        if self.nb_classes is not None:
            score = self.classifier(score)

        return score
