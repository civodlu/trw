import trw.train
import torch
import numpy as np
import skimage.transform
import trw.utils


def resize_torch(array, size, mode):
    return trw.utils.upsample(array, size, mode)


def resize_numpy(array, size, mode):
    if mode == 'nearest':
        order = 0
    elif mode == 'linear':
        order = 1
    else:
        raise NotImplemented('mode={} is not implemented!'.format(mode))

    resized_array = skimage.transform.resize(array, [array.shape[0], array.shape[1]] + size, order=order, anti_aliasing=False, preserve_range=True)
    if resized_array.dtype != array.dtype:
        resized_array = resized_array.astype(array.dtype)
    return resized_array


def resize(array, size, mode='linear'):
    """
    Resize the array

    Args:
        array: a N-dimensional tensor, representing 1D to 3D data (3 to 5 dimensional data with dim 0 for the samples and dim 1 for filters)
        size: a (N-2) list to which the `array` will be upsampled or downsampled
        mode: string among ('nearest', 'linear') specifying the resampling method

    Returns:
        a resized N-dimensional tensor
    """
    if isinstance(array, np.ndarray):
        return resize_numpy(array, size=size, mode=mode)
    elif isinstance(array, torch.Tensor):
        return resize_torch(array, size=size, mode=mode)
    else:
        raise NotImplemented()

