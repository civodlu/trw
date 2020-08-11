import trw
import numpy as np
import torch
from unittest import TestCase
import collections

import trw.utils


def make_data(size):
    d = collections.OrderedDict()
    d['string_list'] = [f'string_{v}' for v in range(size)]
    d['np_array'] = np.zeros([size, 2, 3])
    d['torch_array'] = torch.arange(size)
    return d


class TestSequenceReBatch(TestCase):
    def test_simple_no_overflow_and_full_batch(self):
        d = make_data(60)
        nb_batches = 0
        total_samples = 0
        sequence = trw.train.SequenceArray(d, sampler=trw.train.SamplerSequential(batch_size=10)).rebatch(20)
        for batch_id, batch in enumerate(sequence):
            nb_samples = trw.utils.len_batch(batch)
            nb_batches += 1
            total_samples += nb_samples
            self.assertTrue(nb_samples == 20)

        self.assertTrue(nb_batches == 3)
        self.assertTrue(total_samples == 60)

    def test_simple_no_overflow_discard_not_full_batch(self):
        """
        The last batch of size 10 must be discarded
        """
        d = make_data(70)
        nb_batches = 0
        total_samples = 0
        sequence = \
            trw.train.SequenceArray(d, sampler=trw.train.SamplerSequential(batch_size=10)).\
            rebatch(20, discard_batch_not_full=True)

        for batch_id, batch in enumerate(sequence):
            nb_samples = trw.utils.len_batch(batch)
            nb_batches += 1
            total_samples += nb_samples
            self.assertTrue(nb_samples == 20)

        self.assertTrue(nb_batches == 3)
        self.assertTrue(total_samples == 60)

    def test_with_overflow_full_batch(self):
        """
        Split the overflow. All batches are full
        """
        d = make_data(80)
        nb_batches = 0
        total_samples = 0
        sequence = \
            trw.train.SequenceArray(d, sampler=trw.train.SamplerSequential(batch_size=40)). \
            rebatch(10, discard_batch_not_full=True)

        for batch_id, batch in enumerate(sequence):
            nb_samples = trw.utils.len_batch(batch)
            nb_batches += 1
            total_samples += nb_samples
            self.assertTrue(nb_samples == 10)

        self.assertTrue(nb_batches == 8)
        self.assertTrue(total_samples == 80)

    def test_with_overflow_not_batch(self):
        """
        Make sure the last batch is not discarded
        """
        d = make_data(85)
        nb_batches = 0
        total_samples = 0
        sequence = \
            trw.train.SequenceArray(d, sampler=trw.train.SamplerSequential(batch_size=40)). \
                rebatch(10, discard_batch_not_full=False)

        for batch_id, batch in enumerate(sequence):
            nb_samples = trw.utils.len_batch(batch)
            nb_batches += 1
            total_samples += nb_samples
            if batch_id == 8:
                self.assertTrue(nb_samples == 5)
            else:
                self.assertTrue(nb_samples == 10)

        self.assertTrue(nb_batches == 9)
        self.assertTrue(total_samples == 85)

    def test_with_overflow_not_batch_irregular(self):
        """
        Make sure the last batch is not discarded
        """
        d = make_data(85)
        nb_batches = 0
        total_samples = 0
        sequence = \
            trw.train.SequenceArray(d, sampler=trw.train.SamplerSequential(batch_size=12)). \
                rebatch(10, discard_batch_not_full=False)

        for batch_id, batch in enumerate(sequence):
            nb_samples = trw.utils.len_batch(batch)
            nb_batches += 1
            total_samples += nb_samples
            if batch_id == 8:
                self.assertTrue(nb_samples == 5)
            else:
                self.assertTrue(nb_samples == 10)

        self.assertTrue(nb_batches == 9)
        self.assertTrue(total_samples == 85)
        print('DONE')