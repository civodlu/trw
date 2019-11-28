from trw.train import sequence
from trw.train import sampler
from trw.train import utilities
import numpy as np
import copy
import torch
import collections


# this the name used for the sample UID
sample_uid_name = 'sample_uid'


class SequenceArray(sequence.Sequence):
    """
    Create a sequence of batches from numpy arrays, lists and :class:`torch.Tensor`
    """
    def __init__(self, split, sampler=sampler.SamplerRandom(), transforms=None, use_advanced_indexing=True, sample_uid_name=sample_uid_name):
        """

        Args:
            split: a dictionary of tensors. Tensors may be `numpy.ndarray`, `torch.Tensor`, numeric
            sampler: the sampler to be used to iterate through the sequence
            transforms: a transform or list of transforms to be applied on each batch of data
            use_advanced_indexing:
            sample_uid_name: if not `None`, create a unique UID per sample so that it is easy to track
                particular samples (e.g., during data augmentation)
        """
        super().__init__(None)  # there is no source sequence for this as we get our input from a numpy split
        self.split = split
        self.nb_samples = None
        self.sampler = sampler
        self.sampler_iterator = None
        self.transforms = transforms
        self.use_advanced_indexing = use_advanced_indexing

        # create a unique UID
        if sample_uid_name is not None and sample_uid_name not in split:
            split[sample_uid_name] = np.asarray(np.arange(utilities.len_batch(split)))

    def subsample(self, nb_samples):
        # get random indices
        subsample_sample = sampler.SamplerRandom(batch_size=nb_samples)
        subsample_sample.initializer(self.split)

        # extract the indices
        indices = next(iter(subsample_sample))
        subsampled_split = SequenceArray.get(
            self.split,
            utilities.len_batch(self.split),
            indices,
            self.transforms,
            use_advanced_indexing=True  # use `use_advanced_indexing` so that we keep the types as close as possible to original
        )
        return SequenceArray(subsampled_split, copy.deepcopy(self.sampler), transforms=self.transforms, use_advanced_indexing=self.use_advanced_indexing)

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        uid_values = self.split.get(uids_name)
        assert uid_values is not None, 'no UIDs with name={}'.format(uids_name)

        # find the samples that are in `uids`
        indices_to_keep = []
        uids_set = set(uids)
        for index, uid in enumerate(uid_values):
            if uid in uids_set:
                indices_to_keep.append(index)

        # reorder the `indices_to_keep` following the `uids` ordering
        uids_ordering = {uid: index for index, uid in enumerate(uids)}
        kvp_index_ordering = []
        for index in indices_to_keep:
            uid = uid_values[index]
            ordering = uids_ordering[uid]
            kvp_index_ordering.append((index, ordering))
        kvp_uids_ordering = sorted(kvp_index_ordering, key=lambda value: value[1])
        indices_to_keep = [index for index, ordering in kvp_uids_ordering]

        # extract the samples
        subsampled_split = SequenceArray.get(
            self.split,
            utilities.len_batch(self.split),
            indices_to_keep,
            self.transforms,
            use_advanced_indexing=True  # use `use_advanced_indexing` so that we keep the types as close as possible to original
        )

        if new_sampler is None:
            new_sampler = copy.deepcopy(self.sampler)
        else:
            new_sampler = copy.deepcopy(new_sampler)

        return SequenceArray(subsampled_split, new_sampler, transforms=self.transforms, use_advanced_indexing=self.use_advanced_indexing)

    def initializer(self):
        self.nb_samples = utilities.len_batch(self.split)
        self.sampler.initializer(self.split)
        self.sampler_iterator = iter(self.sampler)

    @staticmethod
    def get(split, nb_samples, indices, transforms, use_advanced_indexing):
        """
        Collect the split indices given and apply a series of transformations

        Args:
            nb_samples: the total number of samples of split
            split: a mapping of `np.ndarray` or `torch.Tensor`
            indices: a list of indices as numpy array
            transforms: a transformation or list of transformations or None
            use_advanced_indexing: if True, use the advanced indexing mechanism else use a simple list (original data is referenced)
                advanced indexing is typically faster for small objects, however for large objects (e.g., 3D data)
                the advanced indexing makes a copy of the data making it very slow.

        Returns:
            a split with the indices provided
        """
        data = {}
        for split_name, split_data in split.items():
            if isinstance(split_data, (torch.Tensor, np.ndarray)) and len(split_data) == nb_samples:
                # here we prefer [split_data[i] for i in indices] over split_data[indices]
                # this is because split_data[indices] will make a deep copy of the data which may be time consuming
                # for large data

                # TODO for small data batch: prefer the indexing, for large data, prefer the referencing
                if use_advanced_indexing:
                    split_data = split_data[indices]
                else:
                    split_data = [split_data[i] for i in indices]
            if isinstance(split_data, list) and len(split_data) == nb_samples:
                split_data = [split_data[i] for i in indices]

            data[split_name] = split_data

        if transforms is None:
            # do nothing: there is no transform
            pass
        elif isinstance(transforms, collections.Sequence):
            # we have a list of transforms, apply each one of them
            for transform in transforms:
                data = transform(data)
        else:
            # anything else should be a functor
            data = transforms(data)

        return data

    def get_next(self):
        indices = self.sampler_iterator.__next__()
        if not isinstance(indices, (np.ndarray, collections.Sequence)):
            indices = [indices]
        return SequenceArray.get(self.split, self.nb_samples, indices, self.transforms, self.use_advanced_indexing)

    def __next__(self):
        return self.get_next()

    def __iter__(self):
        self.initializer()
        return self

