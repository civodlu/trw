import warnings

import trw
import trw.utils
from trw.train import sequence
from trw.train import sampler as sampler_trw
import numpy as np
import collections
import copy
from trw.utils import get_batch_n
from trw.train import SequenceArray


# this the name used for the sample UID
sample_uid_name = 'sample_uid'


class SequenceArrayFixedSamplesPerEpoch(SequenceArray):
    """
    Create a sequence of batches from numpy arrays, lists and :class:`torch.Tensor`
    If number_of_samples_per_epoch is specified, only iterate trough chunks of samples at each iterator call.
    If number_of_samples_per_epoch is None, behave as SequenceArray.
    Note: the sampler iterator is not copied; it is necessary to create multiple instances of
    SequenceArrayFixedSamplesPerEpoch and its Sampler to have multiple independent iterators.
    """
    def __init__(self, split, sampler=None, transforms=None, use_advanced_indexing=True, sample_uid_name=sample_uid_name, number_of_samples_per_epoch=None):
        """

        Args:
            split: a dictionary of tensors. Tensors may be `numpy.ndarray`, `torch.Tensor`, numeric
            sampler: the sampler to be used to iterate through the sequence
            transforms: a transform or list of transforms to be applied on each batch of data
            use_advanced_indexing:
            sample_uid_name: if not `None`, create a unique UID per sample so that it is easy to track
                particular samples (e.g., during data augmentation)
        """
        if sampler is not None:
            super().__init__(split, sampler=sampler, transforms=transforms, use_advanced_indexing=use_advanced_indexing, sample_uid_name=sample_uid_name)
        else:
            super().__init__(split, sampler=sampler_trw.SamplerRandom(), transforms=transforms, use_advanced_indexing=use_advanced_indexing, sample_uid_name=sample_uid_name)
        self.number_of_samples_per_epoch = number_of_samples_per_epoch
        self.sampler.initializer(self.split)
        self.sampler_iterator = iter(self.sampler)

    def subsample(self, nb_samples):
        raise NotImplementedError()

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        raise NotImplementedError()

    def __iter__(self):
        return SequenceArrayFixedSamplesPerEpochIterator(self)


class SequenceArrayFixedSamplesPerEpochIterator(sequence.SequenceIterator):
    """
    Iterate the elements of an :class:`trw.train.SequenceArray` sequence

    Assumptions:
        - underlying `base_sequence` doesn't change sizes while iterating
    """
    def __init__(self, base_sequence):
        super().__init__()
        self.base_sequence = base_sequence
        self.nb_samples = trw.utils.len_batch(self.base_sequence.split)
        self.number_samples_generated = 0

    def __next__(self):
        if self.base_sequence.number_of_samples_per_epoch is not None and \
                self.number_samples_generated >= self.base_sequence.number_of_samples_per_epoch:
            # we have reached the maximum number of samples, stop the sequence
            raise StopIteration()

        try:
            indices = self.base_sequence.sampler_iterator.__next__()
        except StopIteration:
            self.base_sequence.sampler.initializer(self.base_sequence.split)
            self.base_sequence.sampler_iterator = iter(self.base_sequence.sampler)
            if self.base_sequence.number_of_samples_per_epoch is not None:
                indices = self.base_sequence.sampler_iterator.__next__()
            else:
                raise StopIteration()

        if not isinstance(indices, (np.ndarray, collections.Sequence)):
            indices = [indices]

        self.number_samples_generated += len(indices)

        return get_batch_n(
            self.base_sequence.split,
            self.nb_samples,
            indices,
            self.base_sequence.transforms,
            self.base_sequence.use_advanced_indexing)
