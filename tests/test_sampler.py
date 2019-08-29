from unittest import TestCase
import trw.train
import numpy as np


class TestSampler(TestCase):
    def test_sequential_batch_1(self):
        split = {
            'values': np.arange(100)
        }

        sampler = trw.train.SamplerSequential()
        sampler.initializer(split)
        for index, indices in enumerate(sampler):
            assert index == indices

    def test_sequential_batch_10(self):
        split = {
            'values': np.arange(103)
        }

        sampler = trw.train.SamplerSequential(batch_size=10)
        sampler.initializer(split)
        for index, indices in enumerate(sampler):
            if index < 10:
                assert len(indices) == 10
                for d_i, i in enumerate(indices):
                    assert index * 10 + d_i == i
            else:
                assert len(indices) == 3
                assert indices[0] == 100
                assert indices[1] == 101
                assert indices[2] == 102

    def test_sequential_batch_10_smaller(self):
        split = {
            'values': np.arange(5)
        }

        sampler = trw.train.SamplerSequential(batch_size=10)
        sampler.initializer(split)

        indices_list = []
        for index, indices in enumerate(sampler):
            indices_list.append(indices)
            assert len(indices) == 5
            for d_i, i in enumerate(indices):
                assert index * 10 + d_i == i
        assert len(indices_list) == 1

