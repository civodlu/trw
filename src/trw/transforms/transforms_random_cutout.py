import functools
from trw.transforms import transforms
from trw.transforms import cutout_function
from trw.transforms.copy import copy


def _transform_random_cutout(feature_name, feature_value, cutout_size, cutout_value_fn):
    # make sure we do NOT modify the original images
    feature_value = copy(feature_value)
    for sample in feature_value:
        cutout_function.cutout(sample, cutout_size=cutout_size, cutout_value_fn=cutout_value_fn)
    return feature_value


class TransformRandomCutout(transforms.TransformBatchWithCriteria):
    """
    Randomly flip the axis of selected features
    """
    def __init__(self, cutout_size, criteria_fn=None, cutout_value_fn=functools.partial(cutout_function.cutout_value_fn_constant, value=0)):
        """
        Args:
            cutout_size: the size of the regions to occlude
            cutout_value_fn: a function to fill the cutout images. Should directly modify the image
            criteria_fn: how to select the features to transform. If `None` transform all arrays with dim >= 3
        """
        assert isinstance(cutout_size, tuple), 'must be a tuple!'
        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_3_or_above

        super().__init__(criteria_fn=criteria_fn, transform_fn=functools.partial(_transform_random_cutout, cutout_size=cutout_size, cutout_value_fn=cutout_value_fn))


