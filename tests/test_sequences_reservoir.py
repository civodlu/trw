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
