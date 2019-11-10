import collections
import torchvision
import trw.train
import os
import numpy as np
import torch


def create_cifar10_dataset(batch_size=300, root=None, transform_train=None, transform_valid=None, nb_workers=2, data_processing_batch_size=None, normalize_0_1=True):
    if root is None:
        # first, check if we have some environment variables configured
        root = os.environ.get('TRW_DATA_ROOT')

    if root is None:
        # else default a standard folder
        root = './data'

    if nb_workers > 0 and data_processing_batch_size is None:
        data_processing_batch_size = batch_size // nb_workers
    else:
        data_processing_batch_size = batch_size

    cifar_path = os.path.join(root, 'cifar10')

    train_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=True, transform=None, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=False, transform=None, download=True)

    def convert_image(dataset):
        normalization_factor = 1.0
        if normalize_0_1:
            normalization_factor = 255.0

        d = np.transpose(dataset.data.astype(np.float32), (0, 3, 1, 2)) / normalization_factor
        return torch.from_numpy(d)

    ds = {'images': convert_image(train_dataset), 'targets': np.asarray(train_dataset.targets, dtype=np.int64)}
    
    def create_sequence(transforms, ds):
        if transforms is None:
            sequence = trw.train.SequenceArray(ds, trw.train.SamplerRandom(batch_size=batch_size))
        else:
            assert batch_size % data_processing_batch_size == 0
            sampler = trw.train.SamplerRandom(batch_size=data_processing_batch_size)
            sequence = trw.train.SequenceArray(ds, sampler=sampler).map(transforms, nb_workers=nb_workers, max_jobs_at_once=nb_workers * 2)
            sequence = sequence.batch(batch_size // data_processing_batch_size)
        return sequence

    splits = collections.OrderedDict()
    splits['train'] = create_sequence(transform_train, ds).collate()

    ds = {'images': convert_image(test_dataset), 'targets': np.asarray(test_dataset.targets, dtype=np.int64)}
    splits['test'] = create_sequence(transform_valid, ds).collate()
    return {
        'cifar10': splits
    }
