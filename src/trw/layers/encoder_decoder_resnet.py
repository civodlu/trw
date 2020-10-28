from typing import Sequence, Optional, Any

import torch.nn as nn
from trw.basic_typing import ConvKernels, ConvStrides, PoolingSizes, Activation
from trw.layers.blocks import ConvBlockType, BlockConvNormActivation, ConvTransposeBlockType, BlockDeconvNormActivation, \
    BlockRes


class EncoderDecoderResnet(nn.Module):
    def __init__(
            self,
            dimensionality: int,
            input_channels: int,
            output_channels: int,
            encoding_channels: Sequence[int],
            decoding_channels: Sequence[int],
            *,
            convolution_kernels: ConvKernels = 5,
            encoding_strides: ConvStrides = 2,
            decoding_strides: ConvStrides = 2,
            activation: Optional[Activation] = nn.ReLU,
            encoding_block: ConvBlockType = BlockConvNormActivation,
            decoding_block: ConvTransposeBlockType = BlockDeconvNormActivation,
            middle_block: Any = BlockRes):
        super().__init__()

    def forward(self, x):
        pass