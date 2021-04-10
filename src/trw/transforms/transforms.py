from typing import Sequence, Callable, List

import numpy as np
import torch
from ..basic_typing import Batch


class Transform:
    """
    Abstraction of a batch transform
    """
    def __call__(self, batch: Batch) -> Batch:
        raise NotImplementedError()


class TransformBatchWithCriteria(Transform):
    """
    Helper function to apply a given transform function on features that satisfy a criteria
    """
    def __init__(
            self,
            criteria_fn: Callable[[Batch], List[str]],
            transform_fn: Callable[[List[str], Batch], Batch]):
        """

        Args:
            criteria_fn: a function accepting as parameter `batch` and returning a
                list of features where the transform will be applied
            transform_fn: a function accepting parameters `list of feature_name, batch`
                and returning a transformed batch

        Returns:
            A transformed batch
        """
        self.criteria_fn = criteria_fn
        self.transform_fn = transform_fn

    def __call__(self, batch: Batch) -> Batch:
        features_to_transform = self.criteria_fn(batch)
        if len(features_to_transform) > 0:
            new_batch = self.transform_fn(features_to_transform, batch)
            return new_batch
        else:
            return batch


def criteria_is_array_4_or_above(batch: Batch) -> Sequence[str]:
    """
    Return `True` if the feature is a numpy or torch array dim >= 4, typically all
    n-d images, n >= 2
    """
    features = []
    for feature_name, feature_value in batch.items():
        if isinstance(feature_value, np.ndarray) and len(feature_value.shape) >= 4:
            features.append(feature_name)
        elif isinstance(feature_value, torch.Tensor) and len(feature_value.shape) >= 4:
            features.append(feature_name)
    return features


def criteria_feature_name(batch: Batch, feature_names: Sequence[str]) -> Sequence[str]:
    """
    Return `True` if the feature name belongs to a given set of names
    """
    return feature_names
