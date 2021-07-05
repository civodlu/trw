import numpy as np
import torch
from PIL.Image import Image
from PIL import Image
import torchvision


def pic_to_tensor(pic: Image) -> torch.Tensor:
    assert isinstance(pic, Image), 'image must be a PIL Image'

    i = np.array(pic)
    if len(i.shape) == 2:
        i = np.reshape(i, [i.shape[0], i.shape[1], 1])
    i = i.transpose((2, 0, 1))
    return torch.from_numpy(i)


def pic_to_numpy(pic: Image) -> np.ndarray:
    assert isinstance(pic, Image), 'image must be a PIL Image'

    i = np.array(pic)
    if len(i.shape) == 2:
        i = np.reshape(i, [i.shape[0], i.shape[1], 1])
    i = i.transpose((2, 0, 1))
    return i


def download_and_extract_archive(url: str, dataset_path: str) -> None:
    from packaging.version import Version
    current_version = Version(torchvision.__version__)

    min_version = Version('0.2')
    if current_version < Version('0.2'):
        raise NotImplementedError(f'Can\'t download dataset with torchvision <= {min_version}')

    from torchvision.datasets.utils import download_and_extract_archive
    download_and_extract_archive(url, dataset_path)
