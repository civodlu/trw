import functools
from trw.transforms import transforms
from trw.transforms import crop
from trw.transforms import pad


def _transform_random_crop(feature_name, feature_value, padding, mode='edge', constant_value=0, size=None):
    """
    Add a specified padding to the image and randomly crop it so that we have the same size as the original
    image

    This support joint padding & cropping of multiple arrays (e.g., to support segmentation maps)

    Args:
        feature_name: a feature name or a list of feature names
        feature_value: the value of the feature or a list of feature values
        padding: the padding to add to the feature value. If `None`, no padding added
        constant_value: a constant value, depending on the mode selected
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')
        size: if `None`, the image will be cropped to the original size, else it must be a list of the size to crop for each dimension except for dimension 0

    Returns:
        a padded and cropped image to original size
    """

    # we have joint arrays, padding and cropping must be identical for all arrays
    is_joint = isinstance(feature_name, list) and isinstance(feature_value, list) and len(feature_name) == len(feature_value)
    if is_joint:
        crop_fn = crop.transform_batch_random_crop_joint
        pad_fn = pad.transform_batch_pad_joint

        if size is None:
            size = feature_value[0].shape[1:]

    else:
        crop_fn = crop.transform_batch_random_crop
        pad_fn = pad.transform_batch_pad

        if size is None:
            size = feature_value.shape[1:]

    if padding is not None:
        padded = pad_fn(feature_value, padding=padding, mode=mode, constant_value=constant_value)
    else:
        padded = feature_value

    cropped = crop_fn(padded, size)
    return cropped


class TransformRandomCrop(transforms.TransformBatchWithCriteria):
    """
    Add padding on a numpy array of samples and random crop to original size

    Args:
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0). If `None`, no padding added
        criteria_fn: function applied on each feature. If satisfied, the feature will be transformed, if not
            the original feature is returned
        mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')
        size: the size of the cropped image. If `None`, same size as input image

    Returns:
        a randomly cropped batch
    """
    def __init__(self, padding, criteria_fn=None, mode='edge', constant_value=0, size=None):
        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_3_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(_transform_random_crop, padding=padding, mode=mode, constant_value=constant_value, size=size)
         )
        self.criteria_fn = transforms.criteria_is_array_3_or_above


class TransformRandomCropJoint(transforms.TransformBatchJointWithCriteria):
    """
    Add random padding & cropping on a numpy or Torch arrays. The arrays are joints and the same padding/cropping applied on all the arrays

    Args:
        feature_names: these are the features that will be jointly padded and cropped
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0). If `None`, no padding added
        criteria_fn: function applied on each feature. If satisfied, the feature will be transformed, if not
            the original feature is returned
        mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')
        size: the size of the cropped image. If `None`, same size as input image

    Returns:
        a randomly cropped batch
    """
    def __init__(self, feature_names, padding, mode='edge', constant_value=0, size=None):
        super().__init__(
            criteria_fn=functools.partial(transforms.criteria_feature_name, feature_names=feature_names),
            transform_fn=functools.partial(_transform_random_crop, padding=padding, mode=mode, constant_value=constant_value, size=size)
        )
