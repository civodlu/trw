import numpy as np
import torch


class Transform:
    """
    Abstraction of a batch transform
    """
    def __call__(self, batch):
        raise NotImplementedError()


class TransformBatchWithCriteria(Transform):
    """
    Helper function to apply a given transform function on features that satisfy a criteria
    """
    def __init__(self, criteria_fn, transform_fn):
        """

        Args:
            criteria_fn: a function accepting as parameter `batch` and returning a list of features where the
                transform will be applied
            transform_fn: a function accepting parameters `list of feature_name, batch` and returning a transformed batch

        Returns:
            A transformed batch
        """
        self.criteria_fn = criteria_fn
        self.transform_fn = transform_fn

    def __call__(self, batch):
        features_to_transform = self.criteria_fn(batch)
        if len(features_to_transform) > 0:
            new_batch = self.transform_fn(features_to_transform, batch)
            return new_batch
        else:
            return batch


def criteria_is_array_3_or_above(batch):
    """
    Return `True` if the feature is a numpy or torch array dim >= 3
    """
    features = []
    for feature_name, feature_value in batch.items():
        if isinstance(feature_value, np.ndarray) and len(feature_value.shape) >= 3:
            features.append(feature_name)
        elif isinstance(feature_value, torch.Tensor) and len(feature_value.shape) >= 3:
            features.append(feature_name)
    return features


def criteria_feature_name(batch, feature_names):
    """
    Return `True` if the feature name belongs to a given set of names
    """
    return feature_names

