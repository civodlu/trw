import copy
from abc import ABC
from numbers import Number
from typing import Sequence, Callable, Optional, Union, Any

import torch
import torch.nn as nn
from trw.utils import upsample
from trw.layers.blocks import BlockConvNormActivation, BlockUpDeconvSkipConv
from trw.layers.layer_config import LayerConfig, default_layer_config


class Down(nn.Module, ABC):
    def __init__(
            self,
            layer_config: LayerConfig,
            bloc_level: int,
            input_channels: int,
            output_channels: int,
            block: nn.Module = BlockConvNormActivation,
            **block_kwargs):
        super().__init__()

        self.ops = block(
            config=layer_config,
            input_channels=input_channels,
            output_channels=output_channels,
            **block_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


class Up(nn.Module, ABC):
    def __init__(
            self,
            layer_config: LayerConfig,
            bloc_level: int,
            skip_channels: int,
            input_channels: int,
            output_channels: int,
            block: nn.Module = BlockUpDeconvSkipConv,
            **block_kwargs):
        super().__init__()
        self.ops = block(
            config=layer_config,
            skip_channels=skip_channels,
            input_channels=input_channels,
            output_channels=output_channels, **block_kwargs)

    def forward(self, skip: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        return self.ops(skip, previous)


class LatentConv(nn.Module, ABC):
    """
    Concatenate a latent variable (possibly resized to the input shape) and apply a convolution
    """
    def __init__(
            self,
            config: LayerConfig,
            bloc_level: int,
            input_channels: int,
            output_channels: int,
            latent_channels: Optional[int] = None,
            block: nn.Module = BlockConvNormActivation,
            **block_kwargs):
        super().__init__()
        self.latent_channels = latent_channels
        self.input_channels = input_channels
        self.output_channels = output_channels

        if latent_channels is None:
            latent_channels = 0

        # local copy for local configuration
        config = copy.copy(config)
        for key, value in block_kwargs.items():
            config.conv_kwargs[key] = value

        self.dim = config.ops.dim
        self.ops = block(config, input_channels + latent_channels, output_channels)

    def forward(self, x: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        if latent is not None:
            assert latent.shape[1] == self.latent_channels
            assert len(latent.shape) == self.dim + 2
            if latent.shape[2:] != x.shape[2:]:
                # we need to resize the latent variable
                latent = upsample(latent, x.shape[2:], mode='linear')
            x = torch.cat([latent, x], dim=1)

        return self.ops(x)


class UNetBase(nn.Module, ABC):
    """
    Configurable UNet-like architecture
    """
    def __init__(
            self,
            dim: int,
            input_channels: int,
            channels: Sequence[int],
            output_channels: int,
            down_block_fn: Callable[[LayerConfig, int, int, int, int, nn.Module], nn.Module] = Down,
            up_block_fn: Callable[[LayerConfig, int, int, int, nn.Module], nn.Module] = Up,
            init_block_fn: Callable[[LayerConfig, int, int], nn.Module] = BlockConvNormActivation,
            middle_block_fn: Callable[[LayerConfig, int, int, int, nn.Module], nn.Module] = LatentConv,
            output_block_fn: Callable[[LayerConfig, int, int], nn.Module] = BlockConvNormActivation,
            init_block_channels: Optional[int] = None,
            latent_channels: Optional[int] = None,
            kernel_size: Optional[int] = 3,
            strides: Union[int, Sequence[int]] = 2,
            activation: Optional[Any] = None,
            config: LayerConfig = default_layer_config(dimensionality=None)
    ):
        """

        The original UNet architecture is decomposed in 5 blocks:
            - init_block_fn: will first be applied on the input
            - down_block_fn: the encoder
            - middle_block_fn: the junction between the encoder and decoder
            - up_block_fn: the decoder
            - output_block_fn: the output layer

        Args:
            dim: the dimensionality of the UNet (e.g., dim=2 for 2D images)
            input_channels: the number of channels of the input
            output_channels: the number of channels of the output
            down_block_fn: a function taking (dim, in_channels, out_channels) and returning a nn.Module
            up_block_fn: a function taking (dim, in_channels, out_channels, bilinear) and returning a nn.Module
            init_block_channels: the number of channels to be used by the init block. If `None`, `channels[0] // 2`
                will be used
        """
        assert len(channels) >= 1
        config = copy.copy(config)
        config.set_dim(dim)
        if kernel_size is not None:
            config.conv_kwargs['kernel_size'] = kernel_size
            config.deconv_kwargs['kernel_size'] = kernel_size
        if activation is not None:
            config.activation = activation

        super().__init__()
        self.dim = dim
        self.input_channels = input_channels
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.output_channels = output_channels
        self.latent_channels = latent_channels

        if isinstance(strides, Number):
            strides = [strides] * len(channels)
        assert len(strides) == len(channels), f'expected one stride per channel-layer.' \
                                              f'Got={len(strides)} expected={len(channels)}'

        self.init_block = None
        self.downs = None
        self.ups = None
        self.middle_block = None

        if latent_channels is not None:
            assert middle_block_fn is not None

        self._build(config, init_block_fn, down_block_fn, up_block_fn, middle_block_fn, output_block_fn, strides)

    def _build(self, config, init_block_fn, down_block_fn, up_block_fn, middle_block_fn, output_block_fn, strides):
        # make only local copy of config
        config = copy.copy(config)

        self.downs = nn.ModuleList()
        skip_channels = None

        if self.init_block_channels is None:
            out_init_channels = self.channels[0] // 2
        else:
            out_init_channels = self.init_block_channels
        self.init_block = init_block_fn(config, self.input_channels, out_init_channels, stride=1)

        # down blocks
        input_channels = out_init_channels
        for i, skip_channels in enumerate(self.channels):
            self.downs.append(down_block_fn(config, i, input_channels, skip_channels, stride=strides[i]))
            input_channels = skip_channels

        # middle blocks
        if middle_block_fn is not None:
            self.middle_block = middle_block_fn(
                config=config,
                bloc_level=len(self.channels),
                input_channels=skip_channels,
                output_channels=skip_channels,
                latent_channels=self.latent_channels
            )
        else:
            self.middle_block = None

        # up blocks
        input_channels = skip_channels
        self.ups = nn.ModuleList()
        for i in range(len(self.channels)):
            if i + 1 == len(self.channels):
                # last level: use the given number of output channels
                skip_channels = out_init_channels
                out_channels = skip_channels
            else:
                # previous layer
                skip_channels = self.channels[-(i + 2)]
                out_channels = skip_channels

            stride = strides[len(strides) - i - 1]
            self.ups.append(up_block_fn(
                config,
                len(self.channels) - i - 1,
                skip_channels=skip_channels,
                input_channels=input_channels,
                output_channels=out_channels,
                output_padding=stride - 1,
                stride=stride))
            input_channels = skip_channels

        # here we need to have a special output block: this
        # is because we do NOT want to add the activation for the
        # result layer (i.e., often, the output is normalized [-1, 1]
        # and we would discard the negative portion)
        config.norm = None
        config.activation = None
        config.dropout = None
        self.output = output_block_fn(config, out_init_channels, self.output_channels)

    def forward(self, x: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: the input image
            latent: a latent variable appended by the middle block
        """
        prev = self.init_block(x)
        x_n = [prev]
        for down in self.downs:
            current = down(prev)
            x_n.append(current)
            prev = current

        if self.latent_channels is not None:
            assert latent is not None, f'missing latent variable! (latent_channels={self.latent_channels})'
            assert latent.shape[1] == self.latent_channels, f'incorrect latent. Got={latent.shape[1]}, ' \
                                                            f'expected={self.latent_channels}'

        if self.middle_block is not None:
            prev = self.middle_block(x_n[-1], latent=latent)
        else:
            prev = x_n[-1]

        for skip, up in zip(reversed(x_n[:-1]), self.ups):
            prev = up(skip, prev)

        return self.output(prev)


