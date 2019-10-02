import collections
import numpy as np
import torch
import functools
from trw.transforms import crop
from trw.transforms import pad


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
        for feature_name, feature_value in batch.items():
            if self.criteria_fn(feature_name, feature_value):
                transformed_feature_value = self.transform_fn(feature_name, feature_value)
                assert transformed_feature_value is not feature_value, '`transform_fn` should NOT perform in-place operations'
                batch[feature_name] = transformed_feature_value
        return batch


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


def transform_random_crop(feature_name, feature_value, padding, mode='edge', constant_value=0):
    """
    Add a specified padding to the image and randomly crop it so that we have the same size as the original
    image

    Args:
        feature_name: not used
        feature_value: the value of the feature
        padding: the padding to add to the feature value
        constant_value: a constant value, depending on the mode selected
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')

    Returns:
        a padded and cropped image to original size
    """
    padded = pad.transform_batch_pad(feature_value, padding=padding, mode=mode, constant_value=constant_value)
    cropped = crop.transform_batch_random_crop(padded, feature_value.shape[1:])
    return cropped


class TransformRandomCrop(TransformBatchWithCriteria):
    """
        Add padding on a numpy array of samples and random crop to original size

        Args:
            padding: a sequence of size `len(array.shape)-1` indicating the width of the
                padding to be added at the beginning and at the end of each dimension (except for dimension 0)
            feature_names: the name of the features to be padded. If `None`, a reasonable
                guess on the feature to transform will be made
            mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')

        Returns:
            a randomly cropped batch
        """
    def __init__(self, padding, feature_names=None, mode='edge', constant_value=0):
        if feature_names is None:
            criteria_fn = criteria_is_array_3_or_above
        else:
            criteria_fn = functools.partial(criteria_feature_name, feature_names=feature_names)

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(transform_random_crop, padding=padding, mode=mode, constant_value=constant_value)
         )
        self.criteria_fn = criteria_is_array_3_or_above
