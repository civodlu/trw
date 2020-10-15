from typing import Union, List, Tuple, Sequence

import torch


def div_shape(shape: Sequence[int], div: float = 2) -> Sequence[int]:
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
