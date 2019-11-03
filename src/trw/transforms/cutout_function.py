import numpy as np


def cutout_value_fn_constant(image, value):
    """
    Replace all image as a constant value
    """
    image[:] = value


def cutout(image, cutout_size, cutout_value_fn):
    """
    Remove a part of the image randomly

    Args:
        array: a :class:`numpy.ndarray` or :class:`torch.Tensor` n-dimensional array. Samples are stored on axis 0
        cutout_size: the cutout_size of the regions to be occluded
        cutout_value_fn: the function value used for occlusion. Must take as argument `image` and modify directly the image

    Returns:
        None
    """
    nb_dims = len(cutout_size)
    assert len(image.shape) == nb_dims
    offsets = [np.random.randint(0, image.shape[n] - cutout_size[n] + 1) for n in range(len(cutout_size))]

    if nb_dims == 1:
        cutout_value_fn(
            image[offsets[0]:offsets[0] + cutout_size[0]])

    elif nb_dims == 2:
        cutout_value_fn(
            image[
            offsets[0]:offsets[0] + cutout_size[0],
            offsets[1]:offsets[1] + cutout_size[1]])

    elif nb_dims == 3:
        cutout_value_fn(
            image[
            offsets[0]:offsets[0] + cutout_size[0],
            offsets[1]:offsets[1] + cutout_size[1],
            offsets[2]:offsets[2] + cutout_size[2]])

    elif nb_dims == 4:
        cutout_value_fn(
            image[
            offsets[0]:offsets[0] + cutout_size[0],
            offsets[1]:offsets[1] + cutout_size[1],
            offsets[2]:offsets[2] + cutout_size[2],
            offsets[3]:offsets[3] + cutout_size[3]])
    else:
        raise NotImplemented()
