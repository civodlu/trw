from unittest import TestCase
import trw.train


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