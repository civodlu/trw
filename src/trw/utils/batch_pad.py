import numpy as np
import torch
import torch.nn.functional as F


def batch_pad_numpy(array, padding, mode='edge', constant_value=0):
    """
    Add padding on a numpy array of samples. This works for an arbitrary number of dimensions

    Args:
        array: a numpy array. Samples are stored in the first dimension
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode

    Returns:
        a padded array
    """
    assert isinstance(array, np.ndarray), 'must be a numpy array!'
    assert len(padding) == len(array.shape) - 1, 'padding must have shape size of {}'.format(len(array.shape) - 1)
    full_padding = [0] + list(padding)
    full_padding = list(zip(full_padding, full_padding))

    if mode == 'constant':
        constant_values = [(constant_value, constant_value)] * len(array.shape)
        padded_array = np.pad(array, full_padding, mode='constant', constant_values=constant_values)
    else:
        padded_array = np.pad(array, full_padding, mode=mode)
    return padded_array


def batch_pad_torch(array, padding, mode='edge', constant_value=0):
    """
    Add padding on a numpy array of samples. This works for an arbitrary number of dimensions

    This function mimics the API of `transform_batch_pad_numpy` so they can be easily interchanged.

    Args:
        array: a Torch array. Samples are stored in the first dimension
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')

    Returns:
        a padded array
    """
    assert isinstance(padding, list), 'must be a list!'
    assert isinstance(array, torch.Tensor), 'must be a torch.Tensor!'
    assert len(padding) == len(array.shape) - 1, 'padding must have shape size of {}'.format(len(array.shape) - 1)

    full_padding = []
    for p in reversed(padding):  # pytorch start from last dimension to first dimension so reverse the padding
        full_padding.append(p)
        full_padding.append(p)

    if mode == 'edge':
        mode = 'replicate'
    if mode == 'symmetric':
        mode = 'reflect'

    if mode != 'constant':
        # for reflect and replicate we MUST remove the `component` padding
        full_padding = full_padding[:-2]

    if mode == 'constant':
        padded_array = F.pad(array, full_padding, mode='constant', value=constant_value)
    else:
        padded_array = F.pad(array, full_padding, mode=mode)
    return padded_array


def batch_pad(array, padding, mode='edge', constant_value=0):
    """
    Add padding on a numpy array of samples. This works for an arbitrary number of dimensions

    Args:
        array: a numpy array. Samples are stored in the first dimension
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode

    Returns:
        a padded array
    """
    if isinstance(array, np.ndarray):
        return batch_pad_numpy(array, padding, mode, constant_value)
    elif isinstance(array, torch.Tensor):
        return batch_pad_torch(array, padding, mode, constant_value)

    raise NotImplemented()


def batch_pad_joint(arrays, padding, mode='edge', constant_value=0):
    """
    Add padding on a list of numpy or tensor array of samples. Supports arbitrary number of dimensions

    Args:
        arrays: a numpy array. Samples are stored in the first dimension
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode

    Returns:
        a list of padded arrays
    """
    assert isinstance(arrays, list), 'must be a list of arrays'
    padded_arrays = [batch_pad(a, padding=padding, mode=mode, constant_value=constant_value) for a in arrays]
    return padded_arrays
