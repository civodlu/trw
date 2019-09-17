from unittest import TestCase
import trw.train
import numpy as np
import collections


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

    def test_split_numpy_sequential(self):
        split = {'values': np.arange(0, 1000)}

        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential())
        index = 0
        for index, batch in enumerate(split_np):
            assert len(batch) == 2
            assert batch['values'][0] == index
        assert index == 999

    def test_split_numpy_random(self):
        split = {'values': np.arange(0, 1000)}
        values = set()

        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerRandom())
        for batch in split_np:
            assert len(batch) == 2
            values.add(batch['values'][0])

        assert len(values) == 1000

    def test_split_numpy_random_batch(self):
        split = {'values': np.arange(0, 1000)}
        values = set()

        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerRandom(batch_size=10))
        for batch in split_np:
            assert len(batch) == 2
            assert len(batch['values']) == 10
            for v in batch['values']:
                values.add(v)

        assert len(values) == 1000

    def test_split_numpy_random_with_replacement(self):
        nb_classes = 10
        nb_samples = 10000
        split = {'values': np.arange(0, nb_classes)}

        values = []
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerRandom(replacement=True, nb_samples_to_generate=nb_samples))
        for batch in split_np:
            assert len(batch) == 2
            values.append(batch['values'][0])

        assert len(values) == 10000
        values_counts = collections.Counter(values)
        for value, count in values_counts.items():
            assert value >= 0 and value <= 9
            assert abs(count - nb_samples / nb_classes) < 0.1 * nb_samples / nb_classes

    def test_split_numpy_random_with_replacement_batch(self):
        nb_classes = 10
        nb_samples = 10000
        split = {'values': np.arange(0, nb_classes)}

        values = []
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerRandom(replacement=True, nb_samples_to_generate=nb_samples, batch_size=10))
        for batch in split_np:
            assert len(batch) == 2
            assert len(batch['values']) == 10
            for v in batch['values']:
                values.append(v)

        assert len(values) == 10000
        values_counts = collections.Counter(values)
        for value, count in values_counts.items():
            assert value >= 0 and value <= 9
            assert abs(count - nb_samples / nb_classes) < 0.1 * nb_samples / nb_classes

    def test_split_numpy_random_subset(self):
        nb_samples = 10
        split = {'values': np.arange(0, nb_samples)}

        values = set()
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerSubsetRandom(indices=range(5, 10)))
        for batch in split_np:
            assert len(batch) == 2
            value = batch['values'][0]
            values.add(value)
            assert value >= 5 and value < 10
        assert len(values) == 5

    def test_split_class_resample(self):
        nb_samples = 9999  # must be divisible by the number of classes, else it will be rounded

        split = {
            'classes': np.asarray([0, 0, 0, 1, 1, 2]),
            'indices': np.asarray([0, 1, 2, 3, 4, 5]),
        }

        classes = []
        indices = set()
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerClassResampling(class_name='classes', nb_samples_to_generate=nb_samples, batch_size=10))
        for batch in split_np:
            assert len(batch) == 3
            assert 'sample_uid' in batch
            for v in batch['classes']:
                classes.append(v)
            for v in batch['indices']:
                indices.add(v)

        assert len(classes) == nb_samples
        assert len(indices) == 6

        values_counts = collections.Counter(classes)
        for value, count in values_counts.items():
            assert value >= 0 and value <= 2
            assert count == nb_samples // 3

