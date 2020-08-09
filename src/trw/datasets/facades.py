import collections
import functools
import os

import torchvision
import torch
from PIL import Image
import numpy as np
from trw.train import SamplerSequential, SamplerRandom, SequenceArray
from .utils import download_and_extract_archive


def create_facades_dataset(
        root=None,
        batch_size=32,
        normalize_0_1=True,
        transforms_train=None,
        url='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'):

    split_loader = functools.partial(image_directory_facades, normalize_0_1=normalize_0_1, extension='.jpg')

    return create_dataset_from_archive_url(
        root=root,
        split_loader=split_loader,
        batch_size=batch_size,
        transforms_train=transforms_train,
        url=url,
        dataset_name='facades',
        split_train_name='train',
        split_names=['train', 'val', 'test'])


def image_directory_facades(sampler, path, uid_prefix, extension, normalize_0_1):
    """

    Args:
        path:
        uid_prefix:
        extension:
        normalize_0_1:

    Returns:

    """
    files = torchvision.datasets.utils.list_files(path, extension, normalize_0_1)
    images = []
    segmentations = []
    names = []
    for file in files:
        path = os.path.join(path, file)
        i = Image.open(path)
        i = np.array(i).transpose((2, 0, 1))
        images.append(i[:, :, :i.shape[2] // 2])
        segmentations.append(i[:, :, i.shape[2] // 2:])
        names.append(uid_prefix + '_' + file)

    images = np.asarray(images, dtype=np.float32)
    segmentations = np.asarray(segmentations, dtype=np.float32)

    if normalize_0_1:
        images /= 255.0
        segmentations /= 255.0

    split = {
        'images': torch.from_numpy(images),
        'segmentations': torch.from_numpy(segmentations),
        'sample_uid': names
    }
    return SequenceArray(split, sampler=sampler)


def create_dataset_from_archive_url(
        dataset_name,
        split_loader,
        split_names,
        url,
        root=None,
        batch_size=32,
        split_train_name='train',
        transforms_train=None,
        ):

    if root is None:
        # first, check if we have some environment variables configured
        root = os.environ.get('TRW_DATA_ROOT')

    if root is None:
        # else default a standard folder
        root = './data'

    path = os.path.join(root, 'facades')
    torchvision.datasets.utils.makedir_exist_ok(path)

    exist = os.path.exists(os.path.join(path, 'facades'))
    if not exist:
        download_and_extract_archive(url, path)

    dataset = collections.OrderedDict()
    assert split_train_name in split_names, 'no training split!'
    for split_name in split_names:
        if split_name == split_train_name:
            sampler = SamplerRandom(batch_size=batch_size)
        else:
            sampler = SamplerSequential(batch_size=batch_size)

        root_split = os.path.join(path, dataset_name, split_name)
        split = split_loader(sampler, root_split, uid_prefix=split_name)
        if split_name == split_train_name and transforms_train is not None:
            split = split.map(transforms_train, nb_workers=0)

        dataset[split_name] = split

    return collections.OrderedDict([
        (dataset_name, dataset)
    ])
