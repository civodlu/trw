from typing import Optional

import torch

from ..basic_typing import Length, ShapeX
import numpy as np
from .affine import affine_transformation_get_spacing, affine_transformation_get_origin,\
    apply_homogeneous_affine_transform


def make_aligned_patient_scale_transform(origin: Length, spacing: Length) -> torch.Tensor:
    """
    Make a patient scale transform from (origin, spacing, without rotation or shear component)

    Args:
        origin: (Z)YX origin of the geometric space (millimeter)
        spacing: (Z)YX spacing of the geometric space (millimeter)

    Returns:
        a patient scale transform, mapping a (Z)YX voxel to a position in (Z)YX defined in millimeter
    """
    origin = np.asarray(origin)[::-1]
    spacing = np.asarray(spacing)[::-1]

    dim = len(origin)
    pst = np.eye(dim + 1, dtype=np.float32)
    pst[:dim, dim] = origin
    for n in range(dim):
        pst[n, n] = spacing[n]
    return torch.from_numpy(pst)


def patient_scale_transform_get_spacing(pst: torch.Tensor) -> torch.Tensor:
    """
    Return the spacing (expansion factor) of the transformation per dimension ZYX

    Args:
        pst: a 3x3 or 4x4 transformation matrix

    Returns:
        [Z]YX spacing
    """
    spacing = affine_transformation_get_spacing(pst)
    return torch.flip(spacing, dims=(0,))


def patient_scale_transform_get_origin(pst: torch.Tensor) -> torch.Tensor:
    """
    Return the origin of the transformation per dimension ZYX

    Args:
        pst: a 3x3 or 4x4 transformation matrix

    Returns:
        [Z]YX spacing
    """
    o = affine_transformation_get_origin(pst)
    return torch.flip(o, dims=(0,))


class SpatialInfo:
    """
    Represent a the geometric space of a n-dimensional (2D or 3D) volume.

    Concepts: patient scale transform
        we often need to work with data in a given geometric space. This can be achieved
        by mapping voxel indices of a tensor to given geometric space by applying a linear
        transform on the location of the voxel to express.

        This patient transform can be decomposed as multiple linear transforms such as
        translation, rotation, zoom and shearing. SpatialInfo will encode its geometric space
        as PST = Translation * (RotationZ *) RotationY * RotationZ * Spacing

        The matrix is a homogeneous transformation matrix:

        ```
              | RXx RYx RZx Tx |
        PST = | RXy RYy RZy Ty |
              | RXz RYz RZz Tz |
              | 0   0   0   1  |
        ```

        with (RX, RY, RZ) the basis of the geometric space. The spacing is defined as (||RX||^2, ||RY||^2, ||RZ||^2).

    We use arbitrary unit `millimeter` unit for all the attributes.
    """
    def __init__(
            self,
            shape: ShapeX,
            patient_scale_transform: Optional[torch.Tensor] = None,
            origin: Optional[Length] = None,
            spacing: Optional[Length] = None,
        ):
        """
        Args:
            origin: a n-dimensional vector representing the distance between world origin (0, 0, 0) to the origin
                (top left voxel) of a volume defined in ZYX order
            spacing: the size in mm of a voxel in each dimension, in ZYX order
            shape: the shape of the volume in DHW order
            patient_scale_transform: an affine transformation matrix that maps the voxel coordinates
                to a world coordinates. if not None, `origin` and `spacing` should be `None` as the transform
                already defines the origin and spacing
        """
        assert (patient_scale_transform is not None and origin is None and spacing is None) or \
               (patient_scale_transform is None and (origin is not None or spacing is not None)), \
            'define only `patient_scale_transform` OR [`origin`, `spacing`]. PST already encodes `origin` and `spacing`'

        self.shape: Optional[ShapeX] = shape

        if patient_scale_transform is None:
            dim = len(shape)
            if origin is None:
                origin = [0] * dim
            if spacing is None:
                spacing = [1] * dim
            patient_scale_transform = make_aligned_patient_scale_transform(origin=origin, spacing=spacing)

        self.patient_scale_transform = None
        self.patient_scale_transform_inv = None
        self.set_patient_scale_transform(patient_scale_transform)

    def set_patient_scale_transform(self, patient_scale_transform: torch.Tensor) -> None:
        assert len(patient_scale_transform.shape) == 2, 'must be a 2D array'
        assert patient_scale_transform.shape[0] == patient_scale_transform.shape[1], 'must be square!'
        assert patient_scale_transform.shape[0] == len(self.shape) + 1, 'N-dimensional must have a (N+1) x (N+1) PST'

        self.patient_scale_transform = patient_scale_transform
        self.patient_scale_transform_inv = patient_scale_transform.inverse()

    @property
    def spacing(self) -> np.ndarray:
        """
        Calculate the spacing of the PST. Return the components as ZYX order.
        """
        return patient_scale_transform_get_spacing(self.patient_scale_transform).numpy()  # TODO refactor! this should be Tensor. Remove `resample.resample_3d`

    @property
    def origin(self) -> np.ndarray:
        """
        Return the origin expressed in world space (expressed as ZYX order).
        """
        return patient_scale_transform_get_origin(self.patient_scale_transform).numpy()  # TODO refactor! this should be Tensor. Remove `resample.resample_3d`

    def index_to_position(self, index_xyz: torch.Tensor) -> torch.Tensor:
        """
        Map an index to world space

        Args:
            index_xyz: coordinate in index space

        Returns:
            position in world space XY(Z)
        """
        return apply_homogeneous_affine_transform(self.patient_scale_transform, index_xyz)

    def position_to_index(self, position_xyz: torch.Tensor) -> torch.Tensor:
        """
        Map world space coordinate to an index

        Args:
            position_xyz: position in world space

        Returns:
            coordinate in index space XY(Z)
        """
        return apply_homogeneous_affine_transform(self.patient_scale_transform_inv, position_xyz)

