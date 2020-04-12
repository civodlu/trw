import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trw.layers import ModulelWithIntermediate, ConvsBase, ConvsTransposeBase
from trw.transforms import batch_crop


class AutoencoderConvolutional(nn.Module, ModulelWithIntermediate):
    """
    Convolutional autoencoder

    Examples:
        Create an encoder taking 1 channel with [4, 8, 16] filters and a decoder taking as input 16 channels
        of 4x4 with [8, 4, 1] filters:
        >>> model = trw.layers.AutoencoderConvolutional(2, 1, [4, 8, 16], [8, 4, 1])
    """
    def __init__(
            self,
            cnn_dim,
            input_channels,
            encoder_channels,
            decoder_channels,
            convolution_kernels=5,
            strides=1,
            pooling_size=2,
            convolution_repeats=1,
            activation=nn.ReLU,
            dropout_probability=None,
            batch_norm_kwargs=None,
            lrn_kwargs=None,
            last_layer_is_output=False,
            force_decoded_size_same_as_input=True):
        """

        Args:
            cnn_dim:
            input_channels:
            encoder_channels:
            decoder_channels:
            convolution_kernels:
            strides:
            pooling_size:
            convolution_repeats:
            activation:
            dropout_probability:
            batch_norm_kwargs:
            lrn_kwargs:
            last_layer_is_output:
            force_decoded_size_same_as_input: this will force
        """


        super().__init__()
        self.force_decoded_size_same_as_input = force_decoded_size_same_as_input

        self.encoder = ConvsBase(
            cnn_dim=cnn_dim,
            input_channels=input_channels,
            channels=encoder_channels,
            convolution_kernels=convolution_kernels,
            strides=strides,
            pooling_size=pooling_size,
            convolution_repeats=convolution_repeats,
            activation=activation,
            dropout_probability=dropout_probability,
            batch_norm_kwargs=batch_norm_kwargs,
            lrn_kwargs=lrn_kwargs,
            last_layer_is_output=False
        )

        self.decoder = ConvsTransposeBase(
            cnn_dim=cnn_dim,
            input_channels=encoder_channels[-1],
            channels=decoder_channels,
            strides=strides * 2,
            activation=activation,
            dropout_probability=dropout_probability,
            batch_norm_kwargs=batch_norm_kwargs,
            lrn_kwargs=lrn_kwargs,
            last_layer_is_output=last_layer_is_output
        )

    def forward_simple(self, x):
        encoded_x = self.encoder(x)
        return encoded_x

    def forward_with_intermediate(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        if self.force_decoded_size_same_as_input:
            shape_x = np.asarray(x.shape[2:])
            shape_difference = shape_x - np.asarray(decoded_x.shape[2:])
            assert (shape_difference >= 0).all() or (shape_difference <= 0).all(), \
                f'Not implemented. Expected the decoded shape to ' \
                f'be smaller than x! Shape difference={shape_difference}'

            if shape_difference.max() > 0:
                # here we need to add padding
                left_padding = shape_difference // 2
                right_padding = shape_difference - left_padding

                # padding must remove N, C channels & reversed order
                padding = []
                for left, right in zip(left_padding, right_padding):
                    padding += [right, left]
                padding = list(padding[::-1])
                padded_decoded_x = F.pad(decoded_x, padding, mode='constant', value=0)
                assert padded_decoded_x.shape == x.shape
                return [encoded_x, padded_decoded_x]
            else:
                # we need to crop the image
                shape_difference = - shape_difference
                left_crop = shape_difference // 2
                right_crop = left_crop + shape_x
                cropped_decoded_x = batch_crop(decoded_x, [0] + list(left_crop), [x.shape[1]] + list(right_crop))
                return [encoded_x, cropped_decoded_x]

        return [encoded_x, decoded_x]

    def forward(self, x):
        return self.forward_simple(x)
