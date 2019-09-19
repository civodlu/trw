from unittest import TestCase
import trw.train
import numpy as np
import time
import collections
import torch

def function_to_run(batch):
    #print('JOB starte', batch['path'])
    #time.sleep(0.1)
    #time.sleep(0.5)
    return batch

def function_to_run2(batch):
    print('JOB starte', batch['path'])
    time.sleep(0.5)
    return batch

def make_volume_torch(batch):
    print('JOB starte', batch['path'])
    batch['volume'] = torch.zeros([1, 91, 110, 91], dtype=torch.float)
    time.sleep(2)
    return batch


class TestSampler(TestCase):
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
        nb_epochs = 2000
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
            self.assertTrue(error_percent < 0.1)

    def test_reservoir_performance(self):
        # basic test to measure performance: we expect the sampling time to be negligible
        # and this whatever time it takes for the worker job
        nb_indices = 800
        paths = [[i, 42] for i in range(nb_indices)]
        split = {'path': np.asarray(paths)}
        max_reservoir_samples = 5

        sampler = trw.train.SamplerRandom()
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(
            numpy_sequence,
            max_reservoir_samples=max_reservoir_samples,
            min_reservoir_samples=max_reservoir_samples,
            function_to_run=make_volume_torch).collate().batch(40)

        print('Iter start')
        time_start = time.time()
        iter(sequence)
        time_end = time.time()
        print('Iter end', time_end - time_start)

        time_ref = (time_end - time_start) / max_reservoir_samples
        print('ref=', time_ref)

        nb_epochs = 10
        for i in range(1, nb_epochs):
            time_start = time.time()
            batches = []
            for batch in sequence:
                batches.append(batch)
            time_end = time.time()
            time_epoch = time_end - time_start
            print('TIME=', time_end - time_start)
            self.assertTrue(time_epoch * 10 < time_ref)

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

    def test_uniform_sampling(self):
        """
        Make sure we sample uniformly the samples
        """
        nb_indices = 20
        nb_reservoir_samples = 10
        maximum_number_of_samples_per_epoch = 5
        nb_epochs = 5000

        sampler = trw.train.SamplerRandom()
        split = {'path': np.asarray(np.arange(nb_indices))}
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(
            numpy_sequence,
            max_reservoir_samples=nb_reservoir_samples,
            min_reservoir_samples=nb_reservoir_samples,
            function_to_run=function_to_run,
            maximum_number_of_samples_per_epoch=maximum_number_of_samples_per_epoch).collate()

        frequencies = collections.defaultdict(lambda: 0)
        nb_samples = 0

        for epoch in range(nb_epochs):
            for batch in sequence:
                nb_batch_samples = trw.train.len_batch(batch)
                for n in range(nb_batch_samples):
                    nb_samples += 1
                    uid = int(trw.train.to_value(batch[trw.train.default_sample_uid_name][n]))
                    frequencies[uid] += 1

        expected_sampling = nb_samples / nb_indices
        tolerance = 0.05
        for uid, sampling in frequencies.items():
            assert abs(expected_sampling - sampling) < tolerance * expected_sampling, 'expected={}, found={}'.format(expected_sampling, sampling)
