from typing import List

import torch
import torch.nn.functional as F
from scipy.ndimage import affine_transform
import numpy as np

from ..basic_typing import Numeric, Length, NumpyTensorX, TensorX, TorchTensorNCX
from ..train.compatibility import grid_sample, affine_grid
from .affine import affine_transformation_translation, affine_transformation_scale
from .spatial_info import SpatialInfo
from typing_extensions import Literal


def mm_list(matrices: List[torch.Tensor]):
    """
    Multiply all matrices
    """
    m = matrices[0]
    for n in range(1, len(matrices)):
        m = torch.mm(m, matrices[n])
    return m


def resample_spatial_info(
        geometry_moving: SpatialInfo,
        moving_volume: TorchTensorNCX,
        geometry_fixed: SpatialInfo,
        tfm: torch.Tensor,
        interpolation: Literal['linear', 'nearest'] = 'linear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        align_corners: bool = False) -> TorchTensorNCX:
    """
    Apply an affine transformation to a given (moving) volume into a given geometry (fixed)

    Args:
        geometry_moving: Defines the geometric space of the moving volume
        moving_volume: the moving volume (2D or 3D)
        geometry_fixed: define the geometric space to be resampled
        tfm: an affine transformation matrix that moves the moving volume
        interpolation: how to interpolate the moving volume
        padding_mode: defines how to handle missing (moving) data
        align_corners: specifies how to align the voxel grids

    Returns:
        a volume with geometric space `geometry_fixed`. The content is the `moving_volume` moved by `tfm`

    Notes:
        the gradient will be propagated through the transform
    """

    # work in XYZ space, not ZYX!
    moving_shape = np.asarray(geometry_moving.shape)[::-1]
    moving_spacing = np.asarray(geometry_moving.spacing)[::-1]

    fixed_shape = np.asarray(geometry_fixed.shape)[::-1]
    fixed_spacing = np.asarray(geometry_fixed.spacing)[::-1]
    dim = len(moving_spacing)

    # move the DDF to max [-1, 1]
    center_moving_xyz = moving_shape * moving_spacing * 0.5
    assert isinstance(center_moving_xyz, np.ndarray)
    assert len(center_moving_xyz) == 3
    tfm_center_moving = affine_transformation_translation(-center_moving_xyz)

    # move the DDF to max [-1, 1]
    center_fixed_xyz = fixed_shape * fixed_spacing * 0.5
    assert isinstance(center_fixed_xyz, np.ndarray)
    assert len(center_fixed_xyz) == 3
    tfm_center_fixed = affine_transformation_translation(-center_fixed_xyz)

    pst_moving = geometry_moving.patient_scale_transform
    pst_fixed = geometry_fixed.patient_scale_transform

    space_transform_moving = affine_transformation_scale([s / 2 for s in moving_shape])
    space_transform_fixed = affine_transformation_scale([s / 2 for s in fixed_shape])

    # TODO check wether we need to shift by half voxed (i.e., `0` at the center or the corner?)
    #half_voxel = affine_transformation_translation([-0.5, -0.5, -0.5])

    # apply a series of coordinate transform to map the transformed moving geometry
    # to the fixed volume geometry
    tfm_torch3x4 = mm_list([
        space_transform_moving.inverse(),
        #half_voxel.inverse(),
        pst_moving.inverse(),
        tfm_center_moving,
        tfm,
        tfm_center_fixed.inverse(),
        pst_fixed,
        #half_voxel,
        space_transform_fixed,
    ])
    tfm_torch3x4 = tfm_torch3x4[:dim]

    if interpolation == 'linear':
        if dim == 2 or dim == 3:
            # pytorch is abusing the `bilinear` naming (it is actually trilinear for 3D)
            interpolation = 'bilinear'
        else:
            raise ValueError('expected only 2D/3D data!')
    elif interpolation == 'nearest':
        pass
    else:
        raise ValueError(f'not supported interpolation={interpolation}')

    target_shape = [1, 1] + list(geometry_fixed.shape)
    grid = affine_grid(tfm_torch3x4.unsqueeze(0), target_shape, align_corners).to(moving_volume.device)
    resampled_torch = grid_sample(
        moving_volume.type(grid.dtype),
        grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)

    return resampled_torch


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

    .. deprecated:: 0.0.2
       Use `resample_3d` instead! This is just for comparison
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
        interpolation_mode: Literal['linear', 'nearest'] = 'linear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        align_corners=False) -> TensorX:

    min_bb_mm = torch.tensor(min_bb_mm, dtype=torch.float32)
    max_bb_mm = torch.tensor(max_bb_mm, dtype=torch.float32)
    np_volume_origin = torch.tensor(np_volume_origin, dtype=torch.float32)
    np_volume_spacing = torch.tensor(np_volume_spacing, dtype=torch.float32)
    resampled_spacing = torch.tensor(resampled_spacing, dtype=torch.float32)

    was_numpy = False
    if isinstance(volume, np.ndarray):
        was_numpy = True
        volume = torch.from_numpy(volume)

    volume_ncx = volume.unsqueeze(0).unsqueeze(0)
    moving_geometry = SpatialInfo(shape=volume.shape, spacing=np_volume_spacing, origin=np_volume_origin)

    fixed_origin = min_bb_mm
    fixed_shape = ((max_bb_mm - min_bb_mm) / (resampled_spacing)).round().type(torch.long)
    fixed_geometry = SpatialInfo(shape=fixed_shape, spacing=resampled_spacing, origin=fixed_origin)

    resampled = resample_spatial_info(
        geometry_moving=moving_geometry,
        moving_volume=volume_ncx,
        geometry_fixed=fixed_geometry,
        tfm=torch.eye(4, dtype=torch.float32),
        interpolation=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    resampled = resampled[0, 0]
    if was_numpy:
        return resampled.detach().numpy()
    return resampled
