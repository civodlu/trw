import unittest
import trw.train
import torch.nn as nn
import torch
import numpy as np


class ModelEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x = batch['x']
        y = batch['y']

        return {
            'embedding_x': trw.train.OutputEmbedding(x),
            'embedding_y': trw.train.OutputEmbedding(y),
        }


class TestCallbackEmbeddingStatistics(unittest.TestCase):
    def test_basic(self):
        callback = trw.train.CallbackEmbeddingStatistics(embedding_names=None, split_name=None)

        options = trw.train.create_default_options(device=torch.device('cpu'))
        history = [{
            'dataset_1': {
                'split_1': {

                }
            }
        }]
        model = ModelEmbedding()

        x = torch.randn(size=[20, 3])
        y = torch.randn(size=[20, 1])
        sampler = trw.train.SamplerSequential(batch_size=20)

        datasets = {
            'dataset_1': {
                'split_1': trw.train.SequenceArray({'x': x, 'y': y}, sampler=sampler)
            }
        }

        losses = {
            'dataset_1': lambda dataset_name, batch, loss_terms: 0.0
        }

        callback(options, history, model, losses, None, datasets, None, None)

        history_step = history[0]['dataset_1']['split_1']
        assert len(history_step) == 2
        assert history_step['embedding_x']['max'] == float(torch.max(x))
        assert history_step['embedding_x']['min'] == float(torch.min(x))
        assert abs(history_step['embedding_x']['mean'] - float(torch.mean(x))) < 1e-5
        assert abs(history_step['embedding_x']['std'] - float(np.std(x.numpy()))) < 1e-5

        assert history_step['embedding_y']['max'] == float(torch.max(y))
        assert history_step['embedding_y']['min'] == float(torch.min(y))
        assert abs(history_step['embedding_y']['mean'] - float(torch.mean(y))) < 1e-5
        assert abs(history_step['embedding_y']['std'] - float(np.std(y.numpy()))) < 1e-5


if __name__ == '__main__':
    unittest.main()
