import functools
from trw.transforms import transforms
from trw.transforms.flip import transform_batch_random_flip_joint, transform_batch_random_flip


def _transform_random_flip(feature_name, feature_value, axis, flip_probability):
    is_joint = isinstance(feature_name, list) and isinstance(feature_value, list) and len(feature_name) == len(feature_value)
    if is_joint:
        flip_fn = transform_batch_random_flip_joint
    else:
        flip_fn = transform_batch_random_flip

    return flip_fn(feature_value, axis=axis, flip_probability=flip_probability)


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


class TransformRandomFlipJoint(transforms.TransformBatchJointWithCriteria):
    """
    Randomly flip the axis of selected features in a joint fashion (if a feature is selected and a sample is
        flipped, it will be flipped for all selected features)
    """
    def __init__(self, feature_names, axis, flip_probability=0.5):
        """
        Args:
            axis: the axis to flip
            flip_probability: the probability that a sample is flipped
            criteria_fn: how to select the features to transform. If `None` transform all arrays with dim >= 3
        """
        super().__init__(
            criteria_fn=functools.partial(transforms.criteria_feature_name, feature_names=feature_names),
            transform_fn=functools.partial(_transform_random_flip, axis=axis, flip_probability=flip_probability)
        )
