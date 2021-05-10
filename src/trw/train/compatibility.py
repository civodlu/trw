import warnings

import torch.nn.functional
from torch import nn
from packaging.version import Version

torch_version = Version(torch.__version__)


def affine_grid(theta, size, align_corners):
    """
    Compatibility layer for new arguments introduced in pytorch 1.3

    See :func:`torch.nn.functional.affine_grid`
    """
    if torch_version >= Version('1.3'):
        return torch.nn.functional.affine_grid(theta=theta, size=size, align_corners=align_corners)
    else:
        if not align_corners:
            warnings.warn('`align_corners` argument is not supported in '
                          'this version and is ignored. Results may differ')

        return torch.nn.functional.affine_grid(theta=theta, size=size)


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    """
    Compatibility layer for argument change between pytorch <= 1.2 and pytorch > 1.3

    See :func:`torch.nn.functional.grid_sample`
    """
    if torch_version < Version('1.3'):
        if not align_corners:
            warnings.warn('`align_corners` argument is not supported in '
                          'this version and is ignored. Results may differ')

        return torch.nn.functional.grid_sample(
            input=input,
            grid=grid,
            mode=mode,
            padding_mode=padding_mode)
    else:
        return torch.nn.functional.grid_sample(
            input=input,
            grid=grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)


if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:
    class Swish(nn.Module):
        """
        For compatibility with old PyTorch versions
        """
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)


if hasattr(torch, 'linalg'):
    torch_linalg_norm = torch.linalg.norm
else:
    def torch_linalg_norm(input, ord, dim):
        return torch.norm(input, p=ord, dim=dim)

