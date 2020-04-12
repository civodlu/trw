"""
This module is dedicated to data augmentations. In particular we strive to have a numpy and pytorch
implementation for each augmentation so that we could if perform it on GPU

Transforms are designed to work for n-dimensional data.
"""

from .crop import transform_batch_random_crop, batch_crop
from .pad import transform_batch_pad_numpy, transform_batch_pad_torch, transform_batch_pad
from .flip import flip
from .copy import copy
from .cutout_function import cutout
from .resize import resize
from .stack import stack
from .normalize import normalize
from .renormalize import renormalize
from .affine import affine_transformation_translation, affine_transformation_rotation2d, affine_transformation_scale, \
    affine_transform, to_voxel_space_transform

from .transforms import Transform, TransformBatchWithCriteria, criteria_feature_name, criteria_is_array_3_or_above
from .transforms_random_crop import TransformRandomCrop
from .transforms_random_flip import TransformRandomFlip
from .transforms_random_cutout import TransformRandomCutout
from .transforms_resize import TransformResize
from .transforms_normalize import TransformNormalize
from .transforms_compose import TransformCompose
from .transforms_affine import TransformAffine
