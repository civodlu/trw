import collections

import functools
from trw.transforms import transforms
from trw.transforms.flip import transform_batch_random_flip_joint


def _transform_random_flip(feature_names, batch, axis, flip_probability):
    arrays = [batch[name] for name in feature_names]
    transformed_arrays = transform_batch_random_flip_joint(arrays, axis=axis, flip_probability=flip_probability)

    new_batch = collections.OrderedDict(zip(feature_names, transformed_arrays))
    for feature_name, feature_value in batch.items():
        if feature_name not in feature_names:
            # not in the transformed features, so copy the original value
            new_batch[feature_name] = feature_value
    return new_batch


class TransformRandomFlip(transforms.TransformBatchWithCriteria):
    """
    Randomly flip the axis of selected features
    """
    def __init__(self, axis, flip_probability=0.5, criteria_fn=None):
        """
        Args:
            axis: the axis to flip
            flip_probability: the probability that a sample is flipped
            criteria_fn: how to select the features to transform. If `None` transform all arrays with dim >= 3
        """
        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_4_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(_transform_random_flip, axis=axis, flip_probability=flip_probability))

