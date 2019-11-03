import functools

from trw.transforms import transforms
from trw.transforms.resize import resize


def _transform_resize(feature_name, feature_value, size, mode):
    assert len(feature_value.shape) == len(size) + 2, 'unexpected shape! `size` should not include samples or filter!'
    return resize(feature_value, size=size, mode=mode)


class TransformResize(transforms.TransformBatchWithCriteria):
    """
    Resize a tensor to a fixed size
    """
    def __init__(self, size, criteria_fn=None, mode='linear'):
        """
        Args:
            size: the size to reshape to. Excluing the sample and filter
            criteria_fn: how to select the features to transform. If `None` transform all arrays with dim >= 3
            mode: the resampling method. Can be `linear` or `nearest`
        """
        assert isinstance(size, list), 'must be a list!'

        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_3_or_above

        super().__init__(criteria_fn=criteria_fn, transform_fn=functools.partial(_transform_resize, size=size, mode=mode))
