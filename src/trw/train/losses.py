import torch
import torch.nn as nn


class LossDiceMulticlass(nn.Module):
    """
    Implementation of the Dice Loss (multi-class) for N-d images
    
    If multi-class, compute the loss for each class then average the losses
    """
    def __init__(self, normalization_fn=nn.Sigmoid, eps=0.0001, return_dice_by_class=False):
        super().__init__()

        self.eps = eps
        self.normalization = None
        self.return_dice_by_class = return_dice_by_class

        if normalization_fn is not None:
            self.normalization = normalization_fn()
        
    def forward(self, output, target):
        """
        
        Args:
            output: must have W x C x d0 x ... x dn shape, where C is the total number of classes to predict
            target: must have W x d0 x ... x dn shape

        Returns:
            if return_dice_by_class is False, return 1 - dice score suitable for optimization.
            Else, return the average dice score by class
        """
        assert len(output.shape) > 2
        assert len(output.shape) == len(target.shape) + 1, 'output: must have W x C x d0 x ... x dn shape and target: must have W x d0 x ... x dn shape'
        
        if self.normalization is not None:
            output = self.normalization(output)

        # for each class (including background!), create a mask
        # so that class N is encoded as one hot at dimension 1
        encoded_target = torch.zeros_like(output)
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        
        intersection = output * encoded_target
        indices_to_sum = tuple(range(2, len(output.shape)))
        numerator = 2 * intersection.sum(indices_to_sum)
        denominator = output + encoded_target
        denominator = denominator.sum(indices_to_sum) + self.eps

        if not self.return_dice_by_class:
            loss_per_channerl = 1 - numerator / denominator
            return loss_per_channerl.sum(1) / output.shape[1]  # average over channels
        else:
            return (numerator / denominator).sum(0) / output.shape[0]  # average over samples
