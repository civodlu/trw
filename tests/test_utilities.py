from unittest import TestCase
import trw.train
import collections
import torch
import numpy as np


class TestUtilities(TestCase):
    def test_find_default_dataset_and_split(self):
        datasets = {
            'dataset': {
                'split1': {
                    'feature1': []
                }
            }
        }

        dataset_name, split_name = trw.train.find_default_dataset_and_split_names(datasets)
        assert dataset_name == 'dataset'
        assert split_name == 'split1'

    def test_find_default_dataset_and_split_with_datasetname(self):
        datasets = {
            'dataset': {
                'split1': {
                    'feature1': []
                }
            }
        }

        dataset_name, split_name = trw.train.find_default_dataset_and_split_names(datasets, default_dataset_name='dataset')
        assert dataset_name == 'dataset'
        assert split_name == 'split1'

    def test_find_default_dataset_and_split_with_splitname(self):
        datasets = {
            'dataset': {
                'split1': {
                    'feature1': []
                }
            }
        }

        dataset_name, split_name = trw.train.find_default_dataset_and_split_names(datasets, default_split_name='split1')
        assert dataset_name == 'dataset'
        assert split_name == 'split1'

    def test_find_default_dataset_and_split_with_wrongdatasetname(self):
        datasets = {
            'dataset': {
                'split1': {
                    'feature1': []
                }
            }
        }

        dataset_name, split_name = trw.train.find_default_dataset_and_split_names(
            datasets,
            default_dataset_name='doesntexist')
        assert dataset_name is None
        assert split_name is None

    def test_find_default_dataset_and_split_with_wrongsplitname(self):
        datasets = {
            'dataset': {
                'split1': {
                    'feature1': []
                }
            }
        }

        dataset_name, split_name = trw.train.find_default_dataset_and_split_names(
            datasets,
            default_split_name='doesntexist')
        assert dataset_name is None
        assert split_name is None

    def test_classification_mapping(self):
        datasets_infos = {
            'dataset1': {
                'split1': {
                    'output_mappings': {
                        'output_name': {
                            'mappinginv': {
                                0: 'class_0'
                            }
                        }
                    }
                }
            }
        }

        mapping = trw.train.get_classification_mapping(datasets_infos, 'dataset1', 'split1', 'output_name')
        assert mapping is not None

        class_name = trw.train.get_class_name(mapping, 0)
        assert class_name == 'class_0'

    def test_classification_mapping_none(self):
        mapping = trw.train.get_classification_mapping(None, 'dataset1', 'split1', 'output_name')
        assert mapping is None

        mapping = trw.train.get_classification_mapping({}, 'dataset1', 'split1', 'output_name')
        assert mapping is None

        mapping = trw.train.get_classification_mapping({'dataset1': {}}, 'dataset1', 'split1', 'output_name')
        assert mapping is None

    def test_flatten_dictionaries(self):
        d = collections.OrderedDict()
        d['key1'] = 'v1'
        d['key2'] = {'key3': 'v2'}
        d['key4'] = {'key5': {'key6': 'v3'}}
        flattened_d = trw.train.flatten_nested_dictionaries(d)
        assert len(flattened_d) == 3
        assert flattened_d['key1'] == 'v1'
        assert flattened_d['key2-key3'] == 'v2'
        assert flattened_d['key4-key5-key6'] == 'v3'

    def test_clamp_n(self):
        t = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        min = torch.LongTensor([3, 2, 4])
        max = torch.LongTensor([3, 4, 8])
        clamped_t = trw.train.clamp_n(t, min, max)

        assert (clamped_t == torch.LongTensor([[3, 2, 4], [3, 4, 6]])).all()

    def test_triplets(self):
        targets = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5]

        anchors, positives, negatives = trw.train.make_triplet_indices(targets)

        assert len(anchors) == len(positives)
        assert len(anchors) == len(negatives)
        assert len(anchors) == len(targets)

        targets = np.asarray(targets)
        assert (targets[anchors] == targets[positives]).all()
        assert np.max(targets[anchors] == targets[negatives]) == 0

    def test_pairs(self):
        targets = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5]

        samples_0, samples_1, same_target = trw.train.make_pair_indices(targets)

        assert len(samples_0) == len(samples_1)
        assert len(samples_0) == len(same_target)

        targets = np.asarray(targets)

        same = targets[samples_0] == targets[samples_1]

        assert (same == same_target).all()
