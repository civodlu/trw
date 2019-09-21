from unittest import TestCase
import trw.train
import torch
import torch.nn as nn


class ModelDense(nn.Module):
    def __init__(self, n_output=5):
        super().__init__()

        self.d1 = nn.Linear(10, 100)
        self.d2 = nn.Linear(100, n_output)

    def forward(self, batch):
        return self.d2(self.d1(batch['input']))


class ModelDenseSequential(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = nn.Sequential(
            nn.Linear(10, 100),
            nn.Linear(100, 5)
        )

    def forward(self, batch):
        return self.d1(batch['input'])


class ModelNested(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = ModelDense()

    def forward(self, batch):
        return self.d1(batch)


class Recorder:
    def __init__(self):
        self.values = []

    def __call__(self, s):
        self.values.append(s)


class TestCallbackModelSummary(TestCase):
    def test_simple_model(self):
        model = ModelDense()
        batch = {
            'input': torch.zeros([11, 10])
        }

        logger = Recorder()
        r = trw.train.model_summary(model, batch, logger)

        # 2 denses + 2 biases
        expected_trainables = 10 * 100 + 100 * 5 + 100 + 5
        assert expected_trainables == r['trainable_params']
        assert r['total_params'] == r['trainable_params']

    def test_simple_model_internal_sequential(self):
        model = ModelDenseSequential()
        batch = {
            'input': torch.zeros([11, 10])
        }

        logger = Recorder()
        r = trw.train.model_summary(model, batch, logger)

        # 2 denses + 2 biases
        expected_trainables = 10 * 100 + 100 * 5 + 100 + 5
        assert expected_trainables == r['trainable_params']
        assert r['total_params'] == r['trainable_params']

    def test_simple_model_sequential(self):
        model = nn.Sequential(
            nn.Linear(10, 100),
            nn.Linear(100, 5)
        )

        batch = torch.zeros([11, 10])

        logger = Recorder()
        r = trw.train.model_summary(model, batch, logger)

        # 2 denses + 2 biases
        expected_trainables = 10 * 100 + 100 * 5 + 100 + 5
        assert expected_trainables == r['trainable_params']
        assert r['total_params'] == r['trainable_params']

    def test_simple_model_nested(self):
        model = ModelNested()
        batch = {
            'input': torch.zeros([11, 10])
        }

        logger = Recorder()
        r = trw.train.model_summary(model, batch, logger)

        # 2 denses + 2 biases
        expected_trainables = 10 * 100 + 100 * 5 + 100 + 5
        assert expected_trainables == r['trainable_params']
        assert r['total_params'] == r['trainable_params']