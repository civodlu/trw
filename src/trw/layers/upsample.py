import torch.nn as nn


def upsample(tensor, size, mode='linear'):
    """
    Upsample a 1D, 2D, 3D tensor

    This is a wrapper around `torch.nn.Upsample` to make it more practical

    Args:
        tensor: 1D (shape = b x c x n), 2D (shape = b x c x h x w) or 3D (shape = b x c x d x h x w)
        size: if 1D, shape = n, if 2D shape = h x w, if 3D shape = d x h x w
        mode: `linear` or `nearest`

    Returns:
        an up-sampled tensor with same batch size and filter size as the input
    """

    assert len(size) + 2 == len(tensor.shape), 'shape must be only the resampled components, WITHOUT the batch and filter channels'
    assert len(tensor.shape) >= 3, 'only 1D, 2D, 3D tensors are currently handled!'
    assert len(tensor.shape) <= 5, 'only 1D, 2D, 3D tensors are currently handled!'

    if mode == 'linear':
        align_corners = False
        if len(tensor.shape) == 4:
            # 2D case
            return nn.Upsample(mode='bilinear', size=size, align_corners=align_corners).forward(tensor)
        elif len(tensor.shape) == 5:
            # 3D case
            return nn.Upsample(mode='trilinear', size=size, align_corners=align_corners).forward(tensor)
        elif len(tensor.shape) == 3:
            # 1D case
            return nn.Upsample(mode='linear', size=size, align_corners=align_corners).forward(tensor)
        else:
            assert 0, 'impossible or bug!'

    elif mode == 'nearest':
        return nn.Upsample(mode='nearest', size=size).forward(tensor)
    else:
        assert 0, 'upsample mode ({}) is not handled'.format(mode)
