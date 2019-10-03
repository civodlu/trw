"""
This module is dedicated to data augmentations. In particular we strive to have a numpy and pytorch
implementation for each augmentation so that we could if perform it on GPU
"""

from .crop import transform_batch_random_crop
from .pad import transform_batch_pad_numpy, transform_batch_pad_torch, transform_batch_pad
from .transforms import Transform, TransformBatchWithCriteria, criteria_feature_name, criteria_is_array_3_or_above
from .transforms_random_crop import TransformRandomCrop
from .transforms_random_flip import TransformRandomFlip
from .transforms_random_cutout import TransformRandomCutout