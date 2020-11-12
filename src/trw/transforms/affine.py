import math
from typing import Sequence

import torch
import torch.nn as nn
import trw.train
from trw.basic_typing import ShapeCX, TorchTensorNCX


def affine_transformation_translation(t: Sequence[float]) -> torch.Tensor:
    """
    Defines an affine translation for 2D or 3D data

    For a 3D transformation, returns:
           | 1 0 0 X |
       M = | 0 1 0 Y |
           | 0 0 1 Z |
           | 0 0 0 1 |
    Args:
        t: a (X, Y, Z) or (X, Y) tuple

    Returns:
        a transformation matrix
    """
    d = len(t)
    assert d == 2 or d == 3
    tfm = torch.eye(d + 1, dtype=torch.float32)
    tfm[0:d, d] = torch.FloatTensor(t)
    return tfm


def affine_transformation_scale(s: Sequence[float]) -> torch.Tensor:
    """
        Defines an affine scaling transformation

        For a 3D transformation, returns:
               | Sx 0  0  0 |
           M = | 0  Sy 0  0 |
               | 0  0  Sz 0 |
               | 0  0  0  1 |
        Args:
            s: a (Sx, Sy, Sz) or (Sx, Sy) tuple

        Returns:
            a transformation matrix
        """
    d = len(s)
    assert d == 2 or d == 3
    tfm = torch.zeros((d + 1, d + 1), dtype=torch.float32)
    for n in range(d):
        tfm[n, n] = s[n]
    tfm[d, d] = 1
    return tfm


def affine_transformation_rotation2d(angle_radian: float) -> torch.Tensor:
    """
    Defines a 2D rotation transform
    Args:
        angle_radian: the rotation angle in radian

    Returns:
        a transformation matrix
    """

    rotation = torch.tensor([
        [math.cos(angle_radian), math.sin(angle_radian), 0],
        [-math.sin(angle_radian), math.cos(angle_radian), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    return rotation


def affine_transformation_rotation_3d_x(angle_radian: float) -> torch.Tensor:
    """
    Rotation in 3D around the x axis

    See Also:
        https://en.wikipedia.org/wiki/Rotation_matrix

    Args:
        angle_radian:

    Returns:
        4x4 torch.Tensor
    """
    rotation = torch.tensor([
        [1, 0,                      0,                       0],
        [0, math.cos(angle_radian), -math.sin(angle_radian), 0],
        [0, math.sin(angle_radian), math.cos(angle_radian),  0],
        [0, 0,                      0,                       1]
    ], dtype=torch.float32)
    return rotation


def affine_transformation_rotation_3d_y(angle_radian: float) -> torch.Tensor:
    """
    Rotation in 3D around the y axis

    See Also:
        https://en.wikipedia.org/wiki/Rotation_matrix

    Args:
        angle_radian:

    Returns:
        4x4 torch.Tensor
    """
    rotation = torch.tensor([
        [math.cos(angle_radian),  0,     math.sin(angle_radian),     0],
        [0,                       1,     0,                          0],
        [-math.sin(angle_radian), 0,     math.cos(angle_radian),     0],
        [0,                       0,     0,                          1]
    ], dtype=torch.float32)
    return rotation


def affine_transformation_rotation_3d_z(angle_radian: float) -> torch.Tensor:
    """
    Rotation in 3D around the y axis

    See Also:
        https://en.wikipedia.org/wiki/Rotation_matrix

    Args:
        angle_radian:

    Returns:
        4x4 torch.Tensor
    """
    rotation = torch.tensor([
        [math.cos(angle_radian), -math.sin(angle_radian),        0,  0],
        [math.sin(angle_radian), math.cos(angle_radian),         0,  0],
        [0,                      0,                              1,  0],
        [0,                      0,                              0,  1]
    ], dtype=torch.float32)
    return rotation


def to_voxel_space_transform(matrix: torch.Tensor, image_shape: ShapeCX) -> torch.Tensor:
    """
    Express the affine transformation in image space coordinate

    Args:
        matrix: a transformation matrix for 2D or 3D transformation
        image_shape: the transformation matrix will be mapped to the image space coordinate system (i.e., the matrix
            is expressed as "voxel"). Should be [C, D, H, W] or [C, H, W] matrix (no `N` component)

    Returns:
        a 2x3 or 3x4 transform

    See:
        this is often used with :class:`trw.transforms.affine_transform` or :class:`torch.nn.functional.affine_grid`
    """
    assert isinstance(matrix, torch.Tensor)
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] == 3 or matrix.shape[0] == 4
    assert len(image_shape) == 3 or len(image_shape) == 4

    nb_rows = matrix.shape[0]

    # Notes:
    # - ``/ 2`` to account for the [-1, 1] range instead of [0, 1]
    # - ``[::-1]`` the transformation matrix is regular matrix (X, Y, Z) but the image is defined as (Z, Y, X) order
    # - ``[1:]`` discard the components of the image
    scale = [s / 2 for s in image_shape[1:][::-1]]
    space_transform = affine_transformation_scale(scale)  # remove the image component

    # the final ``.inverse()`` is used to that the transformation is defined from moving->resampled space
    tfm = torch.mm(torch.mm(space_transform.inverse(), matrix), space_transform).inverse()
    tfm = tfm[:nb_rows - 1]
    return tfm


def affine_transform(
        images: TorchTensorNCX,
        affine_matrices: torch.Tensor,
        interpolation: str = 'bilinear',
        padding_mode: str = 'border',
        align_corners: bool = None) -> TorchTensorNCX:
    """
    Transform a series of images with a series of affine transformations

    Args:
        images: 3D or 2D images with shape [N, C, D, H, W] or [N, C, H, W] respectively
        affine_matrices: a list of size N of 3x4 or 2x3 matrices (see :class:`trw.transforms.to_voxel_space_transform`
        interpolation: the interpolation method. Can be `nearest` or `bilinear`
        padding_mode: the padding to be used for resampled voxels outside the image. Can be ``'zeros'`` | ``'border'``
            | ``'reflection'``
        align_corners: Geometrically, we consider the pixels of the input  as squares rather than points.

    Returns:
        images transformed
    """
    assert isinstance(images, torch.Tensor)
    assert isinstance(affine_matrices, torch.Tensor)

    nb_images = images.shape[0]
    if len(affine_matrices.shape) == 2:
        affine_matrices = affine_matrices.repeat([nb_images, 1, 1])
    else:
        assert len(affine_matrices.shape) == 3
        assert len(affine_matrices) == nb_images

    dim = len(images.shape) - 2
    if dim == 2:
        assert affine_matrices.shape[1] == 2
        assert affine_matrices.shape[2] == 3
    elif dim == 3:
        assert affine_matrices.shape[1] == 3
        assert affine_matrices.shape[2] == 4
    else:
        raise NotImplementedError(f'dimension not supported! Must be 2 or 3, current={dim}')

    grid = trw.train.affine_grid(affine_matrices, list(images.shape), align_corners=align_corners)
    resampled_images = trw.train.grid_sample(
        images,
        grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)

    return resampled_images
