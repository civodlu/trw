import functools
from trw.transforms import transforms
from trw.transforms import flip


def _transform_random_flip(feature_name, feature_value, axis, flip_probability):
    return flip.transform_batch_random_flip(feature_value, axis=axis, flip_probability=flip_probability)


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
            criteria_fn = transforms.criteria_is_array_3_or_above

        super().__init__(criteria_fn=criteria_fn, transform_fn=functools.partial(_transform_random_flip, axis=axis, flip_probability=flip_probability))


