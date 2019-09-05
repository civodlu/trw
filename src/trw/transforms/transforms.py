import collections
import numpy as np
import torch
from . import crop
from . import pad


class Transform:
    def __call__(self, split):
        raise NotImplementedError()


class TransformRandomCrop(Transform):
    """
        Add padding on a numpy array of samples and random crop to original size

        Args:
            padding: a sequence of size `len(array.shape)-1` indicating the width of the
                padding to be added at the beginning and at the end of each dimension (except for dimension 0)
            feature_names: the name of the features to be padded. If `None`, a reasonable
                guess on the feature to transform will be made
            mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')

        Returns:
            a randomly cropped batch
        """
    def __init__(self, padding, feature_names=None, mode='edge', constant_value=0):
        self.padding = padding
        self.feature_names = feature_names
        self.mode = mode
        self.constant_value = constant_value

    def __call__(self, split):
        assert isinstance(split, collections.Mapping), 'must be a dictionary like object'

        if self.feature_names is None:
            # the features are not specified, so make a guess according to dimensions
            names = []
            for name, values in split.items():
                if isinstance(values, np.ndarray) and len(values.shape) >= 3:
                    names.append(name)
                elif isinstance(values, torch.Tensor) and len(values.shape) >= 3:
                    names.append(name)
        else:
            names = self.feature_names

        transformed = collections.OrderedDict()
        for name, values in split.items():
            if name in names:
                padded = pad.transform_batch_pad(values, padding=self.padding, mode=self.mode, constant_value=self.constant_value)
                cropped = crop.transform_batch_random_crop(padded, values.shape[1:])
                transformed[name] = cropped
            else:
                transformed[name] = values
        return transformed
