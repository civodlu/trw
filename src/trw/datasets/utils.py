import os
import numpy as np
import torch
from PIL import Image
import torchvision


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


def download_and_extract_archive(url, dataset_path):
    version = [int(x) for x in torchvision.__version__.split('.')]
    if version[0] == 0 and version[1] <= 2:
        raise NotImplementedError('Can\'t download dataset with torchvision <= 0.2')

    from torchvision.datasets.utils import download_and_extract_archive
    download_and_extract_archive(url, dataset_path)