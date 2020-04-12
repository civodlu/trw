import torch
from trw.train import sub_tensor


class SubTensor(torch.nn.Module):
    """
    Select a region of a tensor (without copy), excluded the first component (N)
    """
    def __init__(self, min_indices, max_indices_exclusive):
        """
        Args:
            min_indices: the minimum indices to select for each dimension, excluded the first component (N)
            max_indices_exclusive: the maximum indices (excluded) to select for each dimension, excluded the first component (N)
        """
        super().__init__()
        self.max_indices_exclusive = list(max_indices_exclusive)
        self.min_indices = [0] + list(min_indices)

    def forward(self, x):
        return sub_tensor(x, self.min_indices, [len(x)] + self.max_indices_exclusive)
