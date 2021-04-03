import collections
import warnings
from numbers import Number

import torch
from trw.basic_typing import TorchTensorNCX, Padding, KernelSize, Stride
from trw.layers.utils import div_shape
from trw.layers.layer_config import LayerConfig
import torch.nn as nn
from typing import Union, Dict, Optional, Sequence, List
from typing_extensions import Protocol  # backward compatibility for python 3.6-3.7
import copy
import numpy as np


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


def _posprocess_padding(config: LayerConfig, conv_kwargs: Dict, ops: List[nn.Module]) -> None:
    """
    Note:
        conv_kwargs will be modified in-place. Make a copy before!
    """
    padding_same = False
    padding = conv_kwargs.get('padding')
    if padding is not None and padding == 'same':
        padding_same = True
        kernel_size = conv_kwargs.get('kernel_size')
        assert kernel_size is not None, 'missing argument `kernel_size` in convolutional arguments!'
        padding = div_shape(kernel_size)
        conv_kwargs['padding'] = padding

    # if the padding is even, it needs to be asymmetric: one side has less padding
    # than the other. Here we need to add an additional ops to perform the padding
    # since we can't do it in the convolution
    if padding is not None:
        if isinstance(padding, int):
            padding = [padding] * config.ops.dim
        else:
            assert isinstance(padding, collections.Sequence)

        kernel_size = conv_kwargs.get('kernel_size')
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * config.ops.dim
        assert kernel_size is not None
        assert len(kernel_size) == len(padding)

        is_even = 1 - np.mod(kernel_size, 2)
        if padding_same and any(is_even):
            # make sure we don't have conflicting padding info:
            # here we do not support all the possible options: if kernel is even
            # no problem but if we use even kernel, <nn.functional.pad> doesn't support all combinations.
            # this will need to be revisited when we have more support.
            padding_mode = conv_kwargs.get('padding_mode')
            if padding_mode is not None:
                if padding_mode != 'zeros':
                    warnings.warn(f'padding mode={padding_mode} is not supported with even padding!')

            #  there is even padding, add special padding op
            full_padding = []
            for k, p in zip(kernel_size, padding):
                left = k // 2
                right = p - left // 2
                # we need to reverse the dimensions, so reverse also the left/right components
                # and then reverse the whole sequence
                full_padding += [right, left]

            ops.append(config.ops.constant_padding(padding=tuple(full_padding[::-1]), value=0))
            # we have explicitly added padding, so now set to
            # convolution padding to none
            conv_kwargs['padding'] = 0

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

        ops: List[nn.Module] = []
        _posprocess_padding(config, conv_kwargs, ops)

        conv = config.conv(
            in_channels=input_channels,
            out_channels=output_channels,
            **conv_kwargs)
        ops.append(conv)

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

        ops: List[nn.Module] = []
        _posprocess_padding(config, deconv_kwargs, ops)

        deconv = config.deconv(
            in_channels=input_channels,
            out_channels=output_channels,
            **deconv_kwargs)
        ops.append(deconv)

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

        ops = []
        if (isinstance(stride, Number) and stride != 1) or (max(stride) != 1 or min(stride) != 1):
            # if stride is 1, don't upsample!
            ops.append(config.ops.upsample_fn(scale_factor=stride))

        _posprocess_padding(config, conv_kwargs, ops)
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
            deconv_block=BlockDeconvNormActivation,
            stride: Optional[Stride] = None):
        super().__init__()

        self.ops_deconv = deconv_block(
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

        # DO NOT use _posprocess_padding here. This is specific to a convolution!

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
