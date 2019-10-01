import collections
import trw.train
import os
import torchvision


def identity(batch):
    return batch


def create_mnist_datasset(batch_size=1000, root=None, transforms=None, nb_workers=5, data_processing_batch_size=200, normalize_0_1=False):
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

    test_dataset = torchvision.datasets.NIST(
        root=root,
        train=False,
        download=True)

    splits = collections.OrderedDict()
    normalization_factor = 1.0
    if normalize_0_1:
        normalization_factor = 255.0
    ds = {'images': train_dataset.data.view((-1, 1, 28, 28)).float().numpy() / normalization_factor, 'targets': train_dataset.targets}

    if transforms is None:
        sequence = trw.train.SequenceArray(ds, trw.train.SamplerRandom(batch_size=batch_size))
    else:
        assert batch_size % data_processing_batch_size == 0
        sampler = trw.train.SamplerRandom(batch_size=data_processing_batch_size)
        sequence = trw.train.SequenceArray(ds, sampler=sampler).map(transforms, nb_workers=nb_workers, max_jobs_at_once=nb_workers * 10)
        sequence = sequence.batch(batch_size // data_processing_batch_size)

    splits['train'] = sequence.collate()

    splits['test'] = trw.train.SequenceArray(
        {'images': test_dataset.data.view((-1, 1, 28, 28)).float().numpy() / normalization_factor, 'targets': test_dataset.targets},
        trw.train.SamplerRandom(batch_size=batch_size)).collate()

    # generate the class mapping
    mapping = dict()
    mappinginv = dict()
    for id, name in enumerate(torchvision.MNIST.classes):
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
