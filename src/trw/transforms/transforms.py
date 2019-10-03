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
    Helper function to apply a given transform function on features that satisty a criteria
    """
    def __init__(self, criteria_fn, transform_fn):
        """

        Args:
            criteria_fn: a function accepting as parameter `feature_name, feature_value` and returning True to apply the transform
                or False to keep the original feature value
            transform_fn: a function accepting parameters `feature_name, feature_value` and returning a transformed feature_value
        """
        self.criteria_fn = criteria_fn
        self.transform_fn = transform_fn

    def __call__(self, batch):
        new_batch = {}
        for feature_name, feature_value in batch.items():
            if self.criteria_fn(feature_name, feature_value):
                transformed_feature_value = self.transform_fn(feature_name, feature_value)
                new_batch[feature_name] = transformed_feature_value
            else:
                new_batch[feature_name] = feature_value
        return new_batch


def criteria_is_array_3_or_above(feature_name, feature_value):
    """
    Return `True` if the feature is a numpy or torch array dim >= 3
    """
    if isinstance(feature_value, np.ndarray) and len(feature_value.shape) >= 3:
        return True
    if isinstance(feature_value, torch.Tensor) and len(feature_value.shape) >= 3:
        return True
    return False


def criteria_feature_name(feature_name, feature_value, feature_names):
    """
    Return `True` if the feature name belongs to a given set of names
    """
    return feature_name in feature_names

