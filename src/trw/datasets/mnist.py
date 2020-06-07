import collections
import trw.train
import os
import torchvision
import numpy as np


def identity(batch):
    return batch


def create_mnist_dataset(
        batch_size=1000,
        root=None,
        transforms=None,
        nb_workers=5,
        data_processing_batch_size=200,
        normalize_0_1=False,
        select_classes_train=None,
        select_classes_test=None):
    """

    Args:
        batch_size:
        root:
        transforms:
        nb_workers:
        data_processing_batch_size:
        normalize_0_1:
        select_classes_train: a subset of classes to be selected for the
            training split
        select_classes_test: a subset of classes to be selected for the
            test split

    Returns:

    """
    if root is None:
        # first, check if we have some environment variables configured
        root = os.environ.get('TRW_DATA_ROOT')

    if root is None:
        # else default a standard folder
        root = './data'

    train_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True)

    test_dataset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=True)

    splits = collections.OrderedDict()
    normalization_factor = 1.0
    if normalize_0_1:
        normalization_factor = 255.0
    ds = {'images': train_dataset.data.view((-1, 1, 28, 28)).float().numpy() / normalization_factor, 'targets': train_dataset.targets}
    if select_classes_train is not None:
        indices = np.where(np.in1d(train_dataset.targets, np.asarray(select_classes_train)))
        ds = {'images': ds['images'][indices], 'targets': ds['targets'][indices]}

    if transforms is None:
        sequence = trw.train.SequenceArray(ds, trw.train.SamplerRandom(batch_size=batch_size))
    else:
        assert batch_size % data_processing_batch_size == 0
        sampler = trw.train.SamplerRandom(batch_size=data_processing_batch_size)
        sequence = trw.train.SequenceArray(ds, sampler=sampler).map(transforms, nb_workers=nb_workers, max_jobs_at_once=nb_workers * 2)
        sequence = sequence.batch(batch_size // data_processing_batch_size)

    splits['train'] = sequence.collate()

    ds = {'images': test_dataset.data.view((-1, 1, 28, 28)).float().numpy() / normalization_factor, 'targets': test_dataset.targets}
    if select_classes_test is not None:
        indices = np.where(np.in1d(test_dataset.targets, np.asarray(select_classes_test)))
        ds = {'images': ds['images'][indices], 'targets': ds['targets'][indices]}

    splits['test'] = trw.train.SequenceArray(ds, trw.train.SamplerRandom(batch_size=batch_size)).collate()

    # generate the class mapping
    mapping = dict()
    mappinginv = dict()
    for id, name in enumerate(torchvision.datasets.MNIST.classes):
        mapping[name] = id
        mappinginv[id] = name
    output_mappings = {'targets': {'mapping': mapping, 'mappinginv': mappinginv}}

    datasets_info = {
        'mnist': {
            'train': {'output_mappings': output_mappings},
            'test': {'output_mappings': output_mappings},
        }
    }

    datasets = {
        'mnist': splits
    }

    return datasets, datasets_info
