import functools
from trw.transforms import transforms
from trw.transforms.normalize import normalize


def _transform_normalize(feature_name, feature_value, mean, std):
    return normalize(feature_value, mean=mean, std=std)


class TransformNormalize(transforms.TransformBatchWithCriteria):
    """
    Normalize a tensor image with mean and standard deviation.

    Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel
    of the input torch.Tensor, input[channel] = (input[channel] - mean[channel]) / std[channel]

    Args:
        array: the torch array to normalize. Expected layout is (sample, filter, d0, ... dN)
        mean: a N-dimensional sequence
        std: a N-dimensional sequence
        criteria_fn: function applied on each feature. If satisfied, the feature will be transformed, if not
            the original feature is returned

    Returns:
        A normalized batch such that the mean is 0 and std is 1 for the selected features
    """
    def __init__(self, mean, std, criteria_fn=None):
        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_3_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(_transform_normalize, mean=mean, std=std)
         )
        self.criteria_fn = criteria_fn
