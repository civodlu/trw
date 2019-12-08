from trw.train import sequence
from trw.train import utilities
import numpy as np
import torch
import collections


def split_in_2_batches(batch: collections.Mapping, first_batch_size: int):
    """
    Split a single batch into 2 batches. The first batch will have a fixed size.

    If there is not enough sample to split the batch, return (batch, None)

    Args:
        batch: the batch to split
        first_batch_size: the batch size of the first batch. The remaining samples will be in the second batch

    Returns:
        a tuple (first batch, second batch)
    """
    batch_size = utilities.len_batch(batch)
    if batch_size <= first_batch_size:
        return batch, None

    # actually, split!
    batch_1 = type(batch)()
    batch_2 = type(batch)()

    for name, value in batch.items():
        if isinstance(value, (np.ndarray, torch.Tensor, list)):
            # split the array/list
            batch_1[name] = value[:first_batch_size]
            batch_2[name] = value[first_batch_size:]
        else:
            # for other types, simply duplicate
            batch_1[name] = value
            batch_2[name] = value

    return batch_1, batch_2


class SequenceReBatch(sequence.Sequence):
    """
    This sequence will normalize the batch size of an underlying sequence

    If the underlying sequence batch is too large, it will be split in multiple cases. Conversely,
    if the size of the batch is too small, it several batches will be merged until we reach the expected batch size.
    """
    def __init__(self, source_split, batch_size, discard_batch_not_full=False, collate_fn=sequence.default_collate_list_of_dicts):
        """
        Normalize a sequence to identical batch size given an input sequence with varying batch size

        Args:
            source_split: the underlying sequence to normalize
            batch_size: the size of the batches created by this sequence
            discard_batch_not_full: if True, the last batch will be discarded if not full
            collate_fn: function to merge multiple batches
        """
        super().__init__(source_split)

        assert batch_size > 0
        assert isinstance(source_split, sequence.Sequence), '`source_split` must be a `Sequence`'
        self.source_split = source_split
        self.batch_size = batch_size
        self.discard_batch_not_full = discard_batch_not_full
        self.iter_source = None
        self.iter_overflow = None
        self.collate_fn = collate_fn

    def subsample(self, nb_samples):
        subsampled_source = self.source_split.subsample(nb_samples)
        return SequenceReBatch(subsampled_source, batch_size=self.batch_size, discard_batch_not_full=self.discard_batch_not_full, collate_fn=self.collate_fn)

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        subsampled_source = self.source_split.subsample_uids(uids, uids_name, new_sampler)
        return SequenceReBatch(subsampled_source, batch_size=self.batch_size, discard_batch_not_full=self.discard_batch_not_full, collate_fn=self.collate_fn)

    def __next__(self):
        batches = []
        total_nb_samples = 0
        try:
            while True:
                if self.iter_overflow is not None:
                    # handles the samples that had previously overflown
                    batch = self.iter_overflow
                    self.iter_overflow = None
                else:
                    # if not, get the next batch
                    batch = self.iter_source.__next__()

                nb_samples = utilities.len_batch(batch)
                if total_nb_samples + nb_samples == self.batch_size:
                    # here we are good!
                    batches.append(batch)
                    break

                if total_nb_samples + nb_samples > self.batch_size:
                    # too many samples, split the batch and keep the extra samples in the overflow
                    first_batch_size = self.batch_size - total_nb_samples
                    first_batch, overflow_batch = split_in_2_batches(batch, first_batch_size)
                    batches.append(first_batch)
                    self.iter_overflow = overflow_batch
                    break

                # else keep accumulating until we have enough samples
                total_nb_samples += nb_samples
                batches.append(batch)

        except StopIteration:
            if len(batches) == 0 or (len(batches) != self.batch_size and self.discard_batch_not_full):
                # end the sequence
                raise StopIteration()

        if self.collate_fn is not None:
            # finally make a batch
            return self.collate_fn(batches)
        return batches

    def __iter__(self):
        self.iter_source = self.source_split.__iter__()
        return self
