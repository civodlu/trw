import warnings

import torch
from trw.basic_typing import TorchTensorNCX, Padding, KernelSize, Stride
from trw.layers.utils import div_shape
from trw.layers.layer_config import LayerConfig
import torch.nn as nn
from typing import Union, Dict, Optional, Sequence
from typing_extensions import Protocol  # backward compatibility for python 3.6-3.7
import copy


class BlockPool(nn.Module):
    def __init__(
            self,
            config: LayerConfig,
            kernel_size: Optional[KernelSize] = 2):

        super().__init__()

        pool_kwargs = copy.copy(config.pool_kwargs)
        if kernel_size is not None:
            pool_kwargs['kernel_size'] = kernel_size

        self.op = config.pool(**pool_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def _posprocess_padding(conv_kwargs: Dict) -> None:
    padding = conv_kwargs.get('padding')
    if padding is not None and padding == 'same':
        kernel_size = conv_kwargs.get('kernel_size')
        assert kernel_size is not None, 'missing argument `kernel_size` in convolutional arguments!'
        conv_kwargs['padding'] = div_shape(kernel_size)

    # handle differences with pytorch <= 1.0
    # where the convolution doesn't have argument `padding_mode`
    version = torch.__version__[:3]
    if 'padding_mode' in conv_kwargs:
        if version == '1.0':
            warnings.warn('convolution doesn\'t have padding_mode as argument in  pytorch <= 1.0. Argument is deleted!')
            del conv_kwargs['padding_mode']


class BlockConvNormActivation(nn.Module):
    def __init__(
            self,
            config: LayerConfig,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            stride: Optional[Stride] = None,
            padding_mode: Optional[str] = None):

        super().__init__()

        # local override of the default config
        conv_kwargs = copy.copy(config.conv_kwargs)
        if kernel_size is not None:
            conv_kwargs['kernel_size'] = kernel_size
        if padding is not None:
            conv_kwargs['padding'] = padding
        if stride is not None:
            conv_kwargs['stride'] = stride
        if padding_mode is not None:
            conv_kwargs['padding_mode'] = padding_mode

        _posprocess_padding(conv_kwargs)

        ops = [
            config.conv(
                in_channels=input_channels,
                out_channels=output_channels,
                **conv_kwargs),
        ]

        if config.norm is not None:
            ops.append(config.norm(num_features=output_channels, **config.norm_kwargs))

        if config.activation is not None:
            ops.append(config.activation(**config.activation_kwargs))
        self.ops = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


class BlockDeconvNormActivation(nn.Module):
    def __init__(
            self,
            config: LayerConfig,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            output_padding: Optional[Union[int, Sequence[int]]] = None,
            stride: Optional[Stride] = None,
            padding_mode: Optional[str] = None):

        super().__init__()

        # local override of the default config
        deconv_kwargs = copy.copy(config.deconv_kwargs)
        if kernel_size is not None:
            deconv_kwargs['kernel_size'] = kernel_size
        if padding is not None:
            deconv_kwargs['padding'] = padding
        if stride is not None:
            deconv_kwargs['stride'] = stride
        if output_padding is not None:
            deconv_kwargs['output_padding'] = output_padding
        if padding_mode is not None:
            deconv_kwargs['padding_mode'] = padding_mode

        _posprocess_padding(deconv_kwargs)

        ops = [
            config.deconv(
                in_channels=input_channels,
                out_channels=output_channels,
                **deconv_kwargs),
        ]

        if config.norm is not None:
            ops.append(config.norm(num_features=output_channels, **config.norm_kwargs))

        if config.activation is not None:
            ops.append(config.activation(**config.activation_kwargs))
        self.ops = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


class BlockUpsampleNnConvNormActivation(nn.Module):
    """
    The standard approach of producing images with deconvolution — despite its successes! — 
    has some conceptually simple issues that lead to checkerboard artifacts in produced images.

    This is an alternative block using nearest neighbor upsampling + convolution.

    See Also:
        https://distill.pub/2016/deconv-checkerboard/

    """
    def __init__(
            self,
            config: LayerConfig,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            # unused, just to follow the other upsampling interface
            output_padding: Optional[Union[int, Sequence[int]]] = None,
            stride: Optional[Stride] = None,
            padding_mode: Optional[str] = None):

        super().__init__()

        # local override of the default config
        conv_kwargs = copy.copy(config.conv_kwargs)
        if kernel_size is not None:
            conv_kwargs['kernel_size'] = kernel_size
        if padding is not None:
            conv_kwargs['padding'] = padding
        # stride is used in the upsampling
        if padding_mode is not None:
            conv_kwargs['padding_mode'] = padding_mode

        if stride is None:
            stride = config.deconv_kwargs.get('stride')

        assert stride is not None

        _posprocess_padding(conv_kwargs)

        ops = []
        if stride != 1:
            # if stride is 1, don't upsample!
            ops.append(config.ops.upsample_fn(scale_factor=stride))

        ops.append(config.conv(in_channels=input_channels,
                               out_channels=output_channels,
                               **conv_kwargs))

        if config.norm is not None:
            ops.append(config.norm(num_features=output_channels, **config.norm_kwargs))

        if config.activation is not None:
            ops.append(config.activation(**config.activation_kwargs))
        self.ops = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


class BlockUpDeconvSkipConv(nn.Module):
    def __init__(
            self,
            config: LayerConfig,
            skip_channels: int,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            output_padding: Optional[Union[int, Sequence[int]]] = None,
            stride: Optional[Stride] = None):
        super().__init__()

        self.ops_deconv = BlockDeconvNormActivation(
            config,
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            padding=padding,
            output_padding=output_padding,
            stride=stride
        )

        self.ops_conv = BlockConvNormActivation(
            config,
            input_channels=output_channels + skip_channels,
            output_channels=output_channels,
            stride=1
        )

        self.skip_channels = skip_channels
        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, skip: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        assert skip.shape[1] == self.skip_channels
        assert previous.shape[1] == self.input_channels
        x = self.ops_deconv(previous)
        assert x.shape == skip.shape, f'got shape={x.shape}, expected={skip.shape}'
        x = torch.cat([skip, x], dim=1)
        x = self.ops_conv(x)
        return x


class ConvTransposeBlockType(Protocol):
    def __call__(
            self,
            config:
            LayerConfig,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            output_padding: Optional[Union[int, Sequence[int]]] = None,
            stride: Optional[Stride] = None,
            padding_mode: Optional[str] = None) -> nn.Module:

        ...


class ConvBlockType(Protocol):
    def __call__(
            self,
            config: LayerConfig,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            stride: Optional[Stride] = None,
            padding_mode: Optional[str] = None) -> nn.Module:
        ...


class BlockRes(nn.Module):
    def __init__(
            self,
            config: LayerConfig,
            channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            padding_mode: Optional[str] = None,
            base_block: ConvBlockType = BlockConvNormActivation):
        super().__init__()

        config = copy.copy(config)
        conv_kwargs = copy.copy(config.conv_kwargs)
        if kernel_size is not None:
            conv_kwargs['kernel_size'] = kernel_size
        if padding is not None:
            conv_kwargs['padding'] = padding
        if padding_mode is not None:
            conv_kwargs['padding_mode'] = padding_mode
        config.conv_kwargs = conv_kwargs
        _posprocess_padding(conv_kwargs)

        stride = 1
        self.block_1 = base_block(
            config, channels, channels,
            kernel_size=kernel_size, padding=padding,
            stride=stride, padding_mode=padding_mode)

        self.activation = config.activation(**config.activation_kwargs)

        config.activation = None
        self.block_2 = base_block(
            config, channels, channels,
            kernel_size=kernel_size, padding=padding,
            stride=stride, padding_mode=padding_mode)

    def forward(self, x: TorchTensorNCX) -> TorchTensorNCX:
        o = self.block_1(x)
        o = self.block_2(o)
        return self.activation(x + o)
