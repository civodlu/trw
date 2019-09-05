import numpy as np
import torch


def _crop_5d(image, min, max):
    return image[min[0]:max[0], min[1]:max[1], min[2]:max[2], min[3]:max[3], min[4]:max[4]]


def _crop_4d(image, min, max):
    return image[min[0]:max[0], min[1]:max[1], min[2]:max[2], min[3]:max[3]]


def _crop_3d(image, min, max):
    return image[min[0]:max[0], min[1]:max[1], min[2]:max[2]]


def _crop_2d(image, min, max):
    return image[min[0]:max[0], min[1]:max[1]]


def _crop_1d(image, min, max):
    return image[min[0]:max[0]]


def transform_batch_random_crop(array, crop_shape):
    """
    Randomly crop a numpy array of samples given a target size. This works for an arbitrary number of dimensions

    Args:
        array: a numpy array. Samples are stored in the first dimension
        crop_shape: a sequence of size `len(array.shape)-1` indicating the shape of the crop

    Returns:
        a cropped array
    """
    is_numpy = isinstance(array, np.ndarray)
    is_torch = isinstance(array, torch.Tensor)

    nb_dims = len(array.shape) - 1
    assert is_numpy or is_torch, 'must be a numpy array or pytorch.Tensor!'
    assert len(crop_shape) == nb_dims, 'padding must have shape size of {}, got={}'.format(nb_dims, len(crop_shape))
    for index, size in enumerate(crop_shape):
        assert array.shape[index + 1] >= size, \
            'crop_size is larger than array size! shape={}, crop_size={}, index={}'.format(array.shape[1:], crop_shape, index)

    # calculate the maximum offset per dimension. We can then
    # use `max_offsets` and `crop_shape` to calculate the cropping
    max_offsets = []
    for size, crop_size in zip(array.shape[1:], crop_shape):
        max_offset = size - crop_size
        max_offsets.append(max_offset)

    nb_samples = array.shape[0]

    # calculate the offset per dimension
    offsets = []
    for max_offset in max_offsets:
        offset = np.random.random_integers(0, max_offset, nb_samples)
        offsets.append(offset)
    offsets = np.stack(offsets, axis=-1)

    # select the crop function according to dimension
    if nb_dims == 1:
        crop_fn = _crop_1d
    elif nb_dims == 2:
        crop_fn = _crop_2d
    elif nb_dims == 3:
        crop_fn = _crop_3d
    elif nb_dims == 4:
        crop_fn = _crop_4d
    elif nb_dims == 5:
        crop_fn = _crop_5d
    else:
        assert 0, 'TODO implement for generic dimension'

    cropped_array = []
    for n in range(nb_samples):
        min_corner = offsets[n]
        cropped_array.append(crop_fn(array[n], min_corner, min_corner + crop_shape))

    if is_numpy:
        return np.asarray(cropped_array)
    if is_torch:
        return torch.stack(cropped_array)
    assert 0, 'unreachable!'
