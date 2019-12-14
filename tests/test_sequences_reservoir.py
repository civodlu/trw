from unittest import TestCase
import trw.train
import numpy as np
import time
import collections
import torch
import functools


def function_to_run(batch):
    return batch


def function_to_run2(batch):
    print('JOB starte', batch['path'])
    time.sleep(0.1)
    return batch

def worker_with_error(batch):
    if batch['sample_uid'][0] == 10:
        raise IndexError('This is an expected exception to test worker recovery from failure!')
    return batch


def make_volume_torch(batch):
    print('JOB starte', batch['path'])
    batch['volume'] = torch.zeros([1, 10, 11, 12], dtype=torch.float)
    time.sleep(0.1)
    return batch


def make_list_dicts(batch, wait_time=None):
    samples = []
    for n in range(10):
        sub_batch = {
            'sample_uid': batch['sample_uid'],
            'volume': torch.zeros([1, 42])
        }
        samples.append(sub_batch)

    if wait_time is not None:
        time.sleep(wait_time)
    return samples


class TestSequenceReservoir(TestCase):
    def test_reservoir_basics2(self):
        # Test the sequence satisfies statistical properties:
        # - items from a sequence must be in equal proportion (they are randomly sampled)
        nb_indices = 10
        paths = [[i, 42] for i in range(nb_indices)]
        split = {'path': np.asarray(paths)}
        max_reservoir_samples = 5

        sampler = trw.train.SamplerRandom()
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(
            numpy_sequence,
            max_reservoir_samples=max_reservoir_samples,
            min_reservoir_samples=max_reservoir_samples,
            function_to_run=function_to_run).collate()

        time.sleep(2)

        time_start = time.time()
        samples = collections.defaultdict(lambda: 0)
        nb_epochs = 10000
        for i in range(1, nb_epochs):
            batches = []
            for batch in sequence:
                nb_samples = trw.train.len_batch(batch)
                # we requested a single sample at a time
                self.assertTrue(nb_samples == 1)

                p = trw.train.to_value(batch['path'])
                self.assertTrue(p.shape == (1, 2))
                batches.append(batch)

                value = int(trw.train.to_value(batch['sample_uid'])[0])
                samples[value] += 1
            self.assertTrue(len(batches) <= max_reservoir_samples)
        time_end = time.time()
        print('TIME=', time_end - time_start)

        expected_counts = nb_epochs / nb_indices * max_reservoir_samples
        for c, counts in samples.items():
            error_percent = abs(counts - expected_counts) / expected_counts
            print(f'c={c}, counts={counts}, expected_counts={expected_counts}, error={error_percent}')
            self.assertTrue(error_percent < 0.1)

    def test_subsample_uid(self):
        """
        Make sure we can resample the sequence with UID
        """
        nb_indices = 800
        paths = [[i, 42] for i in range(nb_indices)]
        split = {'path': np.asarray(paths)}
        max_reservoir_samples = 200

        sampler = trw.train.SamplerSequential()
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(
            numpy_sequence,
            max_reservoir_samples=max_reservoir_samples,
            min_reservoir_samples=max_reservoir_samples,
            function_to_run=function_to_run).collate().batch(40)
        
        subsampled_sequence = sequence.subsample_uids(uids=np.arange(200, 300), uids_name=trw.train.default_sample_uid_name)
        
        nb_samples = 0
        values = set()
        for batch in subsampled_sequence:
            batch_set = set(trw.train.to_value(batch['sample_uid']))
            nb_samples += trw.train.len_batch(batch)
            values = values.union(batch_set)

        assert len(values) == 100
        assert np.min(list(values)) == 200
        assert np.max(list(values)) == 299

    def test_reservoir_batch(self):
        """
        Test that we can easily combine SequenceArray -> SequenceAsyncReservoir -> SequenceBatch
        """
        nb_indices = 20
        split = {'path': np.asarray(np.arange(nb_indices))}
        sampler = trw.train.SamplerSequential(batch_size=1)
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(
            numpy_sequence, min_reservoir_samples=10, max_reservoir_samples=10, function_to_run=make_list_dicts).batch(5)

        for batch in sequence:
            assert trw.train.len_batch(batch) == 5 * 10, 'found={}, expected={}'.format(trw.train.len_batch(batch), 5 * 10)

    def test_fill_reservoir_every_epoch(self):
        """
        The reservoir will start tasks and retrieve results every epoch
        """
        max_jobs_at_once = 5
        nb_indices = 30
        nb_epochs = 7
        min_reservoir_samples = 5
        max_reservoir_samples = 20

        split = {'path': np.asarray(np.arange(nb_indices))}
        sampler = trw.train.SamplerSequential(batch_size=1)
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(
            numpy_sequence,
            min_reservoir_samples=min_reservoir_samples,
            max_reservoir_samples=max_reservoir_samples,
            max_jobs_at_once=max_jobs_at_once,
            function_to_run=functools.partial(make_list_dicts, wait_time=0.02))

        for epoch in range(nb_epochs):
            time.sleep(0.5)
            expected_reservoir_size = min(epoch * max_jobs_at_once, max_reservoir_samples)
            assert sequence.reservoir_size() >= expected_reservoir_size, 'found={}, expected={}'.format(sequence.reservoir_size(), expected_reservoir_size)
            assert sequence.reservoir_size() <= expected_reservoir_size + max_jobs_at_once, 'found={}, expected={}'.format(sequence.reservoir_size(), expected_reservoir_size)
            print('found={}, expected={}'.format(sequence.reservoir_size(), expected_reservoir_size))
            for batch_id, batch in enumerate(sequence):
                if batch_id == 0 and epoch == 0:
                    assert sequence.reservoir_size() >= min_reservoir_samples and \
                           sequence.reservoir_size() <= min_reservoir_samples + max_jobs_at_once

    def test_worker_error(self):
        nb_indices = 20

        split = {'path': np.asarray(np.arange(nb_indices))}
        sampler = trw.train.SamplerSequential(batch_size=1)
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)

        sequence = trw.train.SequenceAsyncReservoir(
            numpy_sequence,
            min_reservoir_samples=5,
            max_reservoir_samples=10,
            max_jobs_at_once=1,
            function_to_run=worker_with_error).collate()

        for n in range(100):
            batches = []
            for batch in sequence:
                assert batch['sample_uid'][0] != 10, 'this job should have failed!'
                batches.append(batch)
            assert len(batches) >= 5