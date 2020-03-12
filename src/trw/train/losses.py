import torch
import torch.nn as nn
import numpy as np


def one_hot(targets, num_classes, dtype=torch.float32):
    """
    Encode the targets (an tensor of integers representing a class)
    as one hot encoding.

    Support target as N-dimensional data (e.g., 3D segmentation map).

    Args:
        num_classes: the total number of classes
        targets: a N-dimensional integral tensor (e.g., 1D for classification, 2D for 2D segmentation map...)
        dtype: the type of the output tensor

    Returns:
        a one hot encoding of a N-dimentional integral tensor
    """
    nb_samples = len(targets)
    if len(targets.shape) == 2:
        # 2D target (e.g., classification)
        encoded_shape = (nb_samples, num_classes)
    else:
        # N-d target (e.g., segmentation map)
        encoded_shape = tuple([nb_samples, num_classes] + list(targets.shape[1:]))

    with torch.no_grad():
        encoded_target = torch.zeros(encoded_shape, dtype=dtype, device=targets.device)
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
    return encoded_target


class LossDiceMulticlass(nn.Module):
    """
    Implementation of the Dice Loss (multi-class) for N-d images
    
    If multi-class, compute the loss for each class then average the losses
    """
    def __init__(self, normalization_fn=nn.Sigmoid, eps=0.00001, return_dice_by_class=False):
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
        assert len(output.shape) == len(target.shape) + 1, 'output: must have W x C x d0 x ... x dn shape and ' \
                                                           'target: must have W x d0 x ... x dn shape'
        
        if self.normalization is not None:
            output = self.normalization(output)

        # for each class (including background!), create a mask
        # so that class N is encoded as one hot at dimension 1
        encoded_target = one_hot(target, output.shape[1], dtype=output.dtype)
        
        intersection = output * encoded_target
        indices_to_sum = tuple(range(2, len(output.shape)))
        numerator = 2 * intersection.sum(indices_to_sum)
        denominator = output + encoded_target
        denominator = denominator.sum(indices_to_sum) + self.eps

        if not self.return_dice_by_class:
            # average over classes (1 loss per sample)
            average_loss_per_channel = (1 - numerator / denominator).mean(dim=1)
            return average_loss_per_channel
        else:
            return (numerator / denominator).mean(dim=0)  # average over samples


class LossFocalMulticlass(nn.Module):
    r"""
        This criterion is a implementation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection, https://arxiv.org/pdf/1708.02002.pdf

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion. One weight factor for each class.
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
    """

    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                assert isinstance(alpha, (list, np.ndarray))
                self.alpha = torch.from_numpy(np.asarray(alpha))
            assert len(alpha.shape) == 1
            assert alpha.shape[0] > 1

        self.gamma = gamma

    def forward(self, outputs, targets):
        assert len(outputs.shape) == len(targets.shape) + 1, 'output: must have W x C x d0 x ... x dn shape and ' \
                                                            'target: must have W x d0 x ... x dn shape'

        if self.alpha is not None:
            assert len(self.alpha) == outputs.shape[1], 'there must be one alpha weight by class!'
            if self.alpha.device != outputs.device:
                self.alpha = self.alpha.to(outputs.device)

        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # for segmentation maps, make sure we average all values by sample
        nb_samples = len(outputs)
        return focal_loss.view((nb_samples, -1)).mean(dim=1)
