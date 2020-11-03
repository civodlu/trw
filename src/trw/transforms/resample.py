from typing import Sequence

import torch
from scipy.ndimage import affine_transform
import numpy as np
from trw.basic_typing import Numeric, Tensor, Length, NumpyTensorX, TensorX
from typing_extensions import Literal


def resample_np_volume_3d(
        np_volume: NumpyTensorX,
        np_volume_spacing: Length,
        np_volume_origin: Length,
        min_bb_mm: Length,
        max_bb_mm: Length,
        resampled_spacing: Length,
        mode: Literal['reflect', 'constant', 'nearest', 'mirror', 'wrap'] = 'constant',
        constant_value: Numeric = 0.0,
        order=1) -> NumpyTensorX:
    """
    Resample a portion of a 3D volume (z, y, x) to a specified spacing/bounding box.

    Args:
        np_volume: a 3D volume
        np_volume_spacing: the spacing [sz, sy, sx] of the input volume
        np_volume_origin: the origin [z, y, x] of the input volume
        min_bb_mm: the min position [z, y, x] of the input volume to be resampled
        max_bb_mm: the max position [z, y, x] of the input volume to be resampled
        resampled_spacing: the spacing of the resampled volume
        mode: specifies how to handle the boundary. See :func:`scipy.ndimage.affine_transform`
        constant_value: if mode == `constant`, use `constant_value` as background value
        order: interpolation order [0..5]

    Returns:
        resampled volume
    """
    zooming_matrix = np.identity(3)
    zooming_matrix[0, 0] = resampled_spacing[0] / np_volume_spacing[0]
    zooming_matrix[1, 1] = resampled_spacing[1] / np_volume_spacing[1]
    zooming_matrix[2, 2] = resampled_spacing[2] / np_volume_spacing[2]

    offset = ((min_bb_mm[0] - np_volume_origin[0]) / np_volume_spacing[0],
              (min_bb_mm[1] - np_volume_origin[1]) / np_volume_spacing[1],
              (min_bb_mm[2] - np_volume_origin[2]) / np_volume_spacing[2])

    output_shape = np.ceil([
        max_bb_mm[0] - min_bb_mm[0],
        max_bb_mm[1] - min_bb_mm[1],
        max_bb_mm[2] - min_bb_mm[2],
    ]) / resampled_spacing

    if order >= 2:
        prefilter = True
    else:
        # pre-filtering is VERY slow and unnecessary for order < 2
        # so diable it
        prefilter = False

    np_volume_r = affine_transform(
        np_volume,
        zooming_matrix,
        offset=offset,
        mode=mode,
        order=1,
        prefilter=prefilter,
        cval=constant_value,
        output_shape=output_shape.astype(int))

    return np_volume_r


def resample_3d(
        volume: TensorX,
        np_volume_spacing: Length,
        np_volume_origin: Length,
        min_bb_mm: Length,
        max_bb_mm: Length,
        resampled_spacing: Length,
        mode: Literal['reflect', 'constant', 'nearest', 'mirror', 'wrap'] = 'constant',
        constant_value: Numeric = 0.0,
        order=1) -> TensorX:
    """
    Resample a portion of a 3D volume (numpy array or torch.Tensor) (z, y, x) to a specified spacing/bounding box.

    Notes:
        if the volume is a torch.tensor, it will NOT be possible to back-propagate the gradient using this
        method (i.e., we revert to a numpy based function to do the processing)

    Args:
        volume: a 3D volume
        np_volume_spacing: the spacing [sz, sy, sx] of the input volume
        np_volume_origin: the origin [z, y, x] of the input volume
        min_bb_mm: the min position [z, y, x] of the input volume to be resampled
        max_bb_mm: the max position [z, y, x] of the input volume to be resampled
        resampled_spacing: the spacing of the resampled volume
        mode: specifies how to handle the boundary. See :func:`scipy.ndimage.affine_transform`
        constant_value: if mode == `constant`, use `constant_value` as background value
        order: interpolation order [0..5]

    Returns:
        resampled volume
    """
    tensor_is_np = isinstance(volume, np.ndarray)
    if not tensor_is_np:
        assert isinstance(volume, torch.Tensor), 'volume must be a numpy.ndarray or a torch.Tensor!'
        assert volume.device == torch.device('cpu'), 'tensor must be on CPU to avoid costly data movement!'
        volume = volume.detach().numpy()  # here this method can ONLY be used

    assert len(volume.shape), 'must be a 3D tensor!'

    resampled_v = resample_np_volume_3d(
        np_volume=volume,
        np_volume_origin=np_volume_origin,
        np_volume_spacing=np_volume_spacing,
        resampled_spacing=resampled_spacing,
        min_bb_mm=min_bb_mm,
        max_bb_mm=max_bb_mm,
        order=order,
        mode=mode,
        constant_value=constant_value,
    )

    if not tensor_is_np:
        resampled_v = torch.tensor(resampled_v)

    return resampled_v
