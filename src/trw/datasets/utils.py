import os
import numpy as np
import torch
from PIL import Image


def pic_to_tensor(pic):
    assert isinstance(pic, Image.Image), 'image must be a PIL Image'

    i = np.array(pic)
    if len(i.shape) == 2:
        i = np.reshape(i, [i.shape[0], i.shape[1], 1])
    i = i.transpose((2, 0, 1))
    return torch.from_numpy(i)


def pic_to_numpy(pic):
    assert isinstance(pic, Image.Image), 'image must be a PIL Image'

    i = np.array(pic)
    if len(i.shape) == 2:
        i = np.reshape(i, [i.shape[0], i.shape[1], 1])
    i = i.transpose((2, 0, 1))
    return i


class named_dataset(object):
    """
    Decorator to convert a dataset with unnamed features to a dataset where all the features are explicitly named
    """
    def __init__(self, names):
        """
        :param names: specifies the names of the `super().__getitem__`. All items must be named
        """
        self.output_names = names

    def __call__(self, cls):
        class DatasetWithNamedOutput(cls):
            output_names = self.output_names
            instance = cls

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __len__(self):
                return self.instance.__len__(self)

            def __getitem__(self, idx):
                r = self.instance.__getitem__(self, idx)
                assert len(self.output_names) == len(r)
                return dict(zip(self.output_names, r))

            @property
            def raw_folder(self):
                return os.path.join(self.root, self.instance.__name__, 'raw')

            @property
            def processed_folder(self):
                return os.path.join(self.root, self.instance.__name__, 'processed')

        return DatasetWithNamedOutput
