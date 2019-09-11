import torch.nn as nn


class ShiftScale(nn.Module):
    """
    Normalize a tensor with a mean and standard deviation

    The output tensor will be (x - mean) / standard_deviation

    This layer simplify the preprocessing for the `trw.simple_layers` package
    """
    
    def __init__(self, mean, standard_deviation):
        """

        Args:
            mean:
            standard_deviation:
        """
        super().__init__()
        self.mean = mean
        self.standard_deviation = standard_deviation
    
    def forward(self, x):
        """
        Args:
            x: a tensor

        Returns: return a flattened tensor
        """
        return (x - self.mean) / self.standard_deviation
