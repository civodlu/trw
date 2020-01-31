import unittest
import trw
import torch.nn as nn
import torch
import os
import numpy as np


class Criterion:
    def __call__(self, output, truth):
        return torch.zeros(len(output))


def identity(x):
    return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x = batch['x']
        return {
            'classification_output': trw.train.OutputClassification(
                x,
                classes_name='x_truth',
                criterion_fn=Criterion, output_postprocessing=identity)
        }


def create_datasets():
    x = torch.arange(0, 10)
    y = x.clone()
    y[0] = 4
    y[1] = 4

    batch = {
        'x': x,
        'x_truth': y,
        'image_rgb': torch.randn([10, 3, 32, 32], dtype=torch.float32),
        'image_g1': torch.randn([10, 1, 32, 32], dtype=torch.float32),
        'image_g0': torch.randn([10, 32, 32], dtype=torch.float32),
    }
    sampler = trw.train.SamplerSequential(batch_size=10)
    split = trw.train.SequenceArray(batch, sampler=sampler)
    return {
        'dataset1': {
            'split1': split
        }
    }


class TestCallbackExportClassificationErrors(unittest.TestCase):
    def test_basic(self):
        callback = trw.train.CallbackExportClassificationErrors()
        options = trw.train.create_default_options(device=torch.device('cpu'))

        model = Model()
        datasets = create_datasets()
        losses = {
            'dataset1': lambda a, b, c: 0
        }

        output_mappings = {
            'x': {
                'mappinginv': {
                    0: 'str_0',
                    1: 'str_1',
                    2: 'str_2',
                    3: 'str_3',
                    4: 'str_4',
                    5: 'str_5',
                    6: 'str_6',
                    7: 'str_7',
                    8: 'str_8',
                    9: 'str_9',
                }
            }
        }

        datasets_infos = {
            'dataset1': {
                'split1': {
                    'output_mappings': output_mappings
                }
            }
        }

        callback(options, None, model, losses, None, datasets, datasets_infos, None)

        expected_root = os.path.join(options['workflow_options']['logging_directory'], 'errors', 'dataset1')
        expected_files = ['classification_output_split1_s0', 'classification_output_split1_s1']
        for expected_file in expected_files:
            path_txt = os.path.join(expected_root, expected_file + '.txt')
            assert os.path.exists(path_txt)

            path_image_rgb = os.path.join(expected_root, expected_file + '_image_rgb.png')
            assert os.path.exists(path_image_rgb)

            path_image_g1 = os.path.join(expected_root, expected_file + '_image_g1.png')
            assert os.path.exists(path_image_g1)

            path_image_g0 = os.path.join(expected_root, expected_file + '_image_g0.npy')
            assert os.path.exists(path_image_g0)

            with open(path_txt, 'r') as f:
                lines = f.readlines()
            assert 'x_str=str_' in lines[-2]
            assert 'x_truth_str=str_' in lines[-1]


    def test_classification_report(self):
        output_mappings = {
            'good': {
                'mappinginv': {
                    0: 'str_0',
                    1: 'str_1',
                }
            }
        }

        datasets_infos = {
            'dataset1': {
                'split1': {
                    'output_mappings': output_mappings
                }
            }
        }

        output_raw = np.random.randn(10, 2)
        truth = np.random.randint(0, 2, 10)
        outputs = {
            'dataset1': {
                'split1': {
                    'output1': {
                        'output_ref': trw.train.OutputClassification(None, 'good'),
                        'output_raw': output_raw,
                        'output': np.argmax(output_raw, axis=1),
                        'output_truth': truth
                    }
                }
            }
        }

        callback = trw.train.CallbackExportClassificationReport()
        options = trw.train.create_default_options(device=torch.device('cpu'))
        options['workflow_options']['current_logging_directory'] = os.path.join(
            options['workflow_options']['logging_directory'],
            'test_classification_report')
        root_output = options['workflow_options']['current_logging_directory']
        trw.train.create_or_recreate_folder(options['workflow_options']['current_logging_directory'])
        callback(options, None, None, None, outputs, None, datasets_infos, None)

        path_report = os.path.join(root_output, 'output1-dataset1-split1-report.txt')
        path_roc = os.path.join(root_output, 'output1-dataset1-split1-ROC.png')
        path_cm = os.path.join(root_output, 'output1-dataset1-split1-cm.png')
        assert os.path.exists(path_report)
        assert os.path.exists(path_roc)
        assert os.path.exists(path_cm)

        with open(path_report, 'r') as f:
            lines = ''.join(f.readlines())

        # make sure the class mapping was correct
        assert 'str_0' in lines
        assert 'str_1' in lines


if __name__ == '__main__':
    unittest.main()
