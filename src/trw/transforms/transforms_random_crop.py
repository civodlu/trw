import functools
from trw.transforms import transforms
from trw.transforms import crop
from trw.transforms import pad


def _transform_random_crop(feature_name, feature_value, padding, mode='edge', constant_value=0):
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


class TransformRandomCrop(transforms.TransformBatchWithCriteria):
    """
        Add padding on a numpy array of samples and random crop to original size

        Args:
            padding: a sequence of size `len(array.shape)-1` indicating the width of the
                padding to be added at the beginning and at the end of each dimension (except for dimension 0)
            criteria_fn: function applied on each feature. If satisfied, the feature will be transformed, if not
                the original feature is returned
            mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')

        Returns:
            a randomly cropped batch
        """
    def __init__(self, padding, criteria_fn=None, mode='edge', constant_value=0):
        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_3_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(_transform_random_crop, padding=padding, mode=mode, constant_value=constant_value)
         )
        self.criteria_fn = transforms.criteria_is_array_3_or_above
