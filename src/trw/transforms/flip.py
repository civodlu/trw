import numpy as np
import torch
from trw.transforms.stack import stack


def flip(array, axis):
    """
    Flip an axis of an array
    Args:
        array: a :class:`numpy.ndarray` or :class:`torch.Tensor` n-dimensional array
        axis: the xis to flip

    Returns:
        an array with specified axis flipped
    """
    if isinstance(array, np.ndarray):
        return np.flip(array, axis=axis)
    elif isinstance(array, torch.Tensor):
        return torch.flip(array, [axis])
    else:
        raise NotImplemented()


def transform_batch_random_flip(array, axis, flip_probability=0.5):
    """
    Randomly flip an image with a given probability

    Args:
        array: a :class:`numpy.ndarray` or :class:`torch.Tensor` n-dimensional array. Samples are stored on axis 0
        axis: the axis to flip
        flip_probability: the probability that a sample is flipped

    Returns:
        an array
    """
    r = np.random.rand(array.shape[0])
    flip_choices = r <= flip_probability

    samples = []
    for flip_choice, sample in zip(flip_choices, array):
        if flip_choice:
            samples.append(flip(sample, axis=axis - 1))
        else:
            samples.append(sample)

    return stack(samples)
