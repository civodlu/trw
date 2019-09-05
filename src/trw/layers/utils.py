import torch


def div_shape(shape, div=2):
    """
    Divide the shape by a constant

    Args:
        shape: the shape
        div: a divisor

    Returns:
        a list
    """
    if isinstance(shape, (list, tuple, torch.Size)):
        return [s // div for s in shape]
    return shape // div
