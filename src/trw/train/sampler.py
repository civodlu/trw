from trw.train import utilities
import torch
import collections
import numpy as np


class Sampler(object):
    """
    Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self):
        self.data_source = None

    def initializer(self, data_source):
        """
        Initialize the sequence iteration

        Args:
            data_source: the data source to iterate
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        Returns: an iterator the return indices of the original data source
        """
        raise NotImplementedError()

    def __len__(self):
        """
        Returns: the number of elements the sampler will return in a single iteration
        """
        raise NotImplementedError()
    
    def get_batch_size(self):
        """
        Returns:
           the size of the batch
        """
        raise NotImplementedError()


class _SamplerSequentialIter:
    """
    Lazily iterate the indices of a sequential batch
    """
    def __init__(self, nb_samples, batch_size):
        self.nb_samples = nb_samples
        self.batch_size = batch_size
        self.current = 0

    def __next__(self):
        if self.current >= self.nb_samples:
            raise StopIteration()

        indices = np.arange(self.current, min(self.current + self.batch_size, self.nb_samples))
        self.current += self.batch_size
        return indices


class SamplerSequential(Sampler):
    """
    Samples elements sequentially, always in the same order.
    """
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def initializer(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        if self.batch_size == 1:
            return iter(range(utilities.len_batch(self.data_source)))
        else:
            return _SamplerSequentialIter(utilities.len_batch(self.data_source), self.batch_size)

    def __len__(self):
        return len(self.data_source)

    def get_batch_size(self):
        return self.batch_size


class SamplerRandom(Sampler):
    """
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    """

    def __init__(self, replacement=False, nb_samples_to_generate=None, batch_size=1):
        """

        Args:
            replacement: samples are drawn with replacement if ``True``, default=``False``
            nb_samples_to_generate: number of samples to draw, default=`len(dataset)`. This argument
                is supposed to be specified only when `replacement` is ``True``.
            batch_size: the number of samples returned by each batch. If possible, use this instead of ``SequenceBatch`` for performance reasons
        """
        super().__init__()
        
        if nb_samples_to_generate is not None:
            assert replacement, 'can only specified `nb_samples_to_generate` when we sample with replacement'
            
        self.replacement = replacement
        self.nb_samples_to_generate = nb_samples_to_generate
        self.indices = None
        self.last_index = None
        self.num_samples = None
        self.batch_size = batch_size

    def initializer(self, data_source):
        self.data_source = data_source
        self.indices = None
        self.last_index = 0

        self.num_samples = utilities.len_batch(self.data_source)
        if not self.replacement and self.nb_samples_to_generate is None:
            self.nb_samples_to_generate = self.num_samples
        
    def __iter__(self):
        if self.replacement:
            self.indices = np.random.randint(0, self.num_samples, size=self.nb_samples_to_generate, dtype=np.int64)
        else:
            self.indices = np.arange(0, self.num_samples)
        np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.last_index >= len(self.indices):
            raise StopIteration
        
        next_indices = self.indices[self.last_index:self.last_index + self.batch_size]
        self.last_index += self.batch_size
        return next_indices

    def __len__(self):
        return self.num_samples
    
    def get_batch_size(self):
        return self.batch_size


class SamplerSubsetRandom(Sampler):
    """
    Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def initializer(self, data_source):
        pass

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    def get_batch_size(self):
        return 1


class SamplerClassResampling(Sampler):
    """
    Resample the samples so that `class_name` classes have equal probably of being sampled.
    
    Classification problems rarely have balanced classes so it is often required to super-sample the minority class to avoid
    penalizing the under represented classes and help the classifier to learn good features (as opposed to learn the class
    distribution).
    """

    def __init__(self, class_name, nb_samples_to_generate, reuse_class_frequencies_across_epochs=True, batch_size=1):
        """
        :param class_name: the class to be resampled. Classes must be integers
        :param nb_samples_to_generate: the number of samples to generate
        :param reuse_class_frequencies_across_epochs: if True, the class frequencies will be calculated only once then reused from epoch to epoch. This is
            because iterating through the samples to calculate the class frequencies may be time consuming and it should not change over the epochs.
        """
        super().__init__()
        self.class_name = class_name
        self.nb_samples_to_generate = nb_samples_to_generate
        self.reuse_class_frequencies_across_epochs = reuse_class_frequencies_across_epochs
        self.batch_size = batch_size
        
        self.samples_index_by_classes = None
        self.indices = None
        self.current_index = None

        self.last_data_source_samples = 0

    def initializer(self, data_source):
        assert self.class_name in data_source, 'can\'t find {} in data!'.format(self.class_name)
        self.data_source = data_source

        data_source_samples = utilities.len_batch(data_source)

        classes = utilities.to_value(data_source[self.class_name])  # we want numpy values here!
        assert len(classes.shape) == 1, 'must be a 1D vector representing a class'
        if self.samples_index_by_classes is None or \
           not self.reuse_class_frequencies_across_epochs or \
           data_source_samples != self.last_data_source_samples:  # if we don't have the same size, rebuild the cache
                self._fit(classes)

        self.last_data_source_samples = data_source_samples

        nb_classes = len(self.samples_index_by_classes)
        nb_samples_per_class = self.nb_samples_to_generate // nb_classes

        indices_by_class = []
        for class_name, indices in self.samples_index_by_classes.items():
            indices_of_indices = np.random.randint(0, len(indices), nb_samples_per_class)
            indices_by_class.append(indices[indices_of_indices])

        # concatenate the indices by class, then shuffle them
        # to make sure we don't have batches with only the same class!
        self.indices = np.concatenate(indices_by_class)
        np.random.shuffle(self.indices)

        self.current_index = 0

    def _fit(self, classes):
        d = collections.defaultdict(lambda: [])
        for index, c in enumerate(classes):
            d[c].append(index)

        self.samples_index_by_classes = {
            c: np.asarray(indexes) for c, indexes in d.items()
        }

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        next_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return next_indices

    def __iter__(self):
        return self

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)

        # this is just an estimate
        return self.nb_samples_to_generate
    
    def get_batch_size(self):
        return self.batch_size
