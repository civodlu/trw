from typing import Sequence, Union, Dict, Any
import numpy as np
import torch

"""Generic numeric type"""
Numeric = Union[int, float]

"""Generic Shape"""
Shape = Sequence[int]

"""Shape expressed as [N, C, D, H, W, ...] components"""
ShapeNCX = Sequence[int]

"""Shape expressed as [C, D, H, W, ...] components"""
ShapeCX = Sequence[int]

"""Shape expressed as [D, H, W, ...] components"""
ShapeX = Sequence[int]

"""Generic Tensor as numpy or torch"""
Tensor = Union[np.ndarray, torch.Tensor]

"""Generic Tensor as numpy or torch. Must be shaped as [N, C, D, H, W, ...]"""
TensorNCX = Union[np.ndarray, torch.Tensor]

"""Generic Tensor as numpy or torch. Must be shaped as 2D array [N, X]"""
TensorNX = Union[np.ndarray, torch.Tensor]


"""Torch Tensor. Must be shaped as [N, C, D, H, W, ...]"""
TorchTensorNCX = torch.Tensor

"""Torch Tensor. Must be shaped as 2D array [N, X]"""
TorchTensorNX = torch.Tensor


"""Numpy Tensor. Must be shaped as [N, C, D, H, W, ...]"""
NumpyTensorNCX = np.ndarray

"""Numpy Tensor. Must be shaped as 2D array [N, X]"""
NumpyTensorNX = np.ndarray

"""Represent a dictionary of (key, value)"""
Batch = Dict[str, Any]

"""Length shaped as D, H, W, ..."""
Length = Sequence[float]
