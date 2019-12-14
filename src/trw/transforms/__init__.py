"""
This module is dedicated to data augmentations. In particular we strive to have a numpy and pytorch
implementation for each augmentation so that we could if perform it on GPU

Transforms are designed to work for n-dimensional data.
"""

from .crop import transform_batch_random_crop
from .pad import transform_batch_pad_numpy, transform_batch_pad_torch, transform_batch_pad
from .flip import flip
from .copy import copy
from .cutout_function import cutout
from .resize import resize
from .stack import stack
from .normalize import normalize
from .renormalize import renormalize

from .transforms import Transform, TransformBatchWithCriteria, criteria_feature_name, criteria_is_array_3_or_above
from .transforms_random_crop import TransformRandomCrop, TransformRandomCropJoint
from .transforms_random_flip import TransformRandomFlip, TransformRandomFlipJoint
from .transforms_random_cutout import TransformRandomCutout
from .transforms_resize import TransformResize
from .transforms_normalize import TransformNormalize
from .transforms_compose import TransformCompose
