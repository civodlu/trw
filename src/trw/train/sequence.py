import logging
import collections
from trw.train import utilities
import torch
import torch.utils.data.dataloader
from trw.train import sampler
import functools
import weakref


# specific windows platform debugging
import platform
is_windows_platform = platform.system() == 'Windows'

logger = logging.getLogger(__name__)


def collate_dicts_pytorch(list_of_dicts):
    """
    Collate a list of dictionaries into a batch (i.e., a dictionary of values by feature name)

    Args:
        list_of_dicts: a list of dictionaries

    Returns:
        a batch
    """
    
    #start = time.time()
    assert isinstance(list_of_dicts, collections.Sequence), 'must be a list!'
    assert isinstance(list_of_dicts[0], collections.Mapping), 'must be a dictionary!'
    if len(list_of_dicts) == 0:
        return {}

    # re-use the default pytorch collate BUT concatenate the batch on axis 0 instead
    d = torch.utils.data.dataloader.default_collate(list_of_dicts)
    r = collections.OrderedDict()
    for name, value in d.items():
        if isinstance(value, (torch.Tensor)):
            r[name] = value.view([-1] + list(value.shape)[2:])
        else:
            r[name] = value

    #end = time.time()
    #print('COLLATE=', end - start)
    return r


def remove_nested_list(items):
    """
    Remove 2 nested list where items is just a list (one element) of list
    """
    if isinstance(items, list) and len(items) == 1 and isinstance(items[0], list):
        # we have a list of list, remove one level of list!
        return items[0]

    return items


# one problem with the default pytorch collate is that it is creating a new dimension
# for the samples instead of concatenating the samples
default_collate_list_of_dicts = functools.partial(utilities.collate_list_of_dicts, device=None)


class Sequence:
    """
    A `Sequence` defines how to iterate the data as a sequence of small batches of data.

    To train a deep learning model, it is often necessary to split our original data into small
    chunks. This is because storing all at once the forward pass of our model is memory
    hungry, instead, we calculate the forward and backward pass on a small chunk of data.
    This is the interface for batching a dataset.

    Examples::

        data = list(range(100))
        sequence = SequenceArray({'data': data}).batch(10)
        for batch in sequence:
            # do something with our batch

    """
    def __init__(self, source_split):
        self.source_split = source_split
        self.next_split = None

        if source_split is not None:
            # use a weak reference to avoid circular references
            source_split.next_split = weakref.proxy(self)
    
    def __iter__(self):
        """

        Returns:
            An iterator of batches
        """
        raise NotImplementedError()
    
    def collate(self, collate_fn=utilities.default_collate_fn, device=None):
        """
        Aggregate the input batch as a dictionary of torch.Tensor and move the data to the appropriate device
        
        Args:
            collate_fn: the function to collate the input batch
            device: the device where to send the samples. If None, the default device is CPU
            
        Returns:
            a collated sequence of batches
        """
        from . import sequence_collate
        return sequence_collate.SequenceCollate(self, collate_fn=collate_fn, device=device)

    def map(self, function_to_run, nb_workers=0, max_jobs_at_once=None, worker_post_process_results_fun=None, queue_timeout=0.1, preprocess_fn=None, collate_fn=None):
        """
        Transform a sequence using a given function.

        .. note:: The map may create more samples than the original sequence.

        :param function_to_run: the mapping function
        :param nb_workers: the number of workers that will process the split. If 0, no workers will be created.
        :param max_jobs_at_once: the maximum number of results that can be pushed in the result queue at once. If 0, no limit.
            If None, it will be set equal to the number of workers
        :param worker_post_process_results_fun: a function used to post-process the worker results (executed by the worker)
        :param queue_timeout: the timeout used to pull results from the output queue
        :param preprocess_fn: a function that will preprocess the batch just prior to sending it to the other processes
        :param collate_fn: a function to collate each batch of data
        :return: a sequence of batches
        """
        from . import sequence_map
        return sequence_map.SequenceMap(
            self,
            function_to_run=function_to_run,
            nb_workers=nb_workers,
            max_jobs_at_once=max_jobs_at_once,
            worker_post_process_results_fun=worker_post_process_results_fun,
            queue_timeout=queue_timeout,
            preprocess_fn=preprocess_fn,
            collate_fn=collate_fn)
    
    def batch(self, batch_size, discard_batch_not_full=False, collate_fn=default_collate_list_of_dicts):
        """
        Group several batches of samples into a single batch
        
        :param batch_size: the number of samples of the batch
        :param discard_batch_not_full: if True and if a batch is not full, discard these
        :param collate_fn: a function to collate the batches. If None, no collation performed
        :return: a sequence of batches
        """
        from . import sequence_batch
        return sequence_batch.SequenceBatch(
            source_split=self,
            batch_size=batch_size,
            discard_batch_not_full=discard_batch_not_full,
            collate_fn=collate_fn,
        )

    def rebatch(self, batch_size, discard_batch_not_full=False, collate_fn=default_collate_list_of_dicts):
        """
        Normalize a sequence to identical batch size given an input sequence with varying batch size

        Args:
            batch_size: the size of the batches created by this sequence
            discard_batch_not_full: if True, the last batch will be discarded if not full
            collate_fn: function to merge multiple batches
        """
        from . import sequence_rebatch
        return sequence_rebatch.SequenceReBatch(
            source_split=self,
            batch_size=batch_size,
            discard_batch_not_full=discard_batch_not_full,
            collate_fn=collate_fn,
        )

    def async_reservoir(
            self,
            max_reservoir_samples,
            function_to_run,
            min_reservoir_samples=1,
            nb_workers=1,
            max_jobs_at_once=None,
            reservoir_sampler=sampler.SamplerSequential(),
            collate_fn=remove_nested_list,
            maximum_number_of_samples_per_epoch=None):
        """
        Create a sequence created from a reservoir. The purpose of this sequence is to maximize the GPU for batches of data
        at the expense of recycling previously processed samples.

        Args:
            max_reservoir_samples: the maximum number of samples of the reservoir
            function_to_run: the function to run asynchronously
            min_reservoir_samples: the minimum of samples of the reservoir needed before an output sequence can be created
            nb_workers: the number of workers that will process `function_to_run`
            max_jobs_at_once: the maximum number of jobs that can be pushed in the result list at once. If 0, no limit. If None: set to the number of workers
            reservoir_sampler: a sampler that will be used to sample the reservoir or None if no sampling needed
            collate_fn: a function to post-process the samples into a single batch. If None, return the items as they were in `source_split`
            maximum_number_of_samples_per_epoch: the maximum number of samples per epoch to generate.
                If we reach this maximum, this will not empty the reservoir but simply interrupt the sequence so
                that we can restart.
        """
        from . import sequence_async_reservoir
        return sequence_async_reservoir.SequenceAsyncReservoir(
            source_split=self,
            max_reservoir_samples=max_reservoir_samples,
            function_to_run=function_to_run,
            min_reservoir_samples=min_reservoir_samples,
            nb_workers=nb_workers,
            max_jobs_at_once=max_jobs_at_once,
            reservoir_sampler=reservoir_sampler,
            collate_fn=collate_fn,
            maximum_number_of_samples_per_epoch=maximum_number_of_samples_per_epoch
        )

    def fill_queue(self):
        """
        Fill the queue jobs of the current sequence
        """
        pass

    def fill_queue_all_sequences(self):
        """
        Go through all the sequences and fill their input queue
        """
        sequences_filled = set()

        sequences_to_examine = [self]
        while len(sequences_to_examine) > 0:
            current = sequences_to_examine.pop()
            sequences_filled.add(current)
            current.fill_queue()

            if current.source_split is not None and current.source_split not in sequences_filled:
                sequences_to_examine.append(current.source_split)
            #if current.next_split is not None and current.next_split not in sequences_filled:
            #    sequences_to_examine.append(current.next_split)

    def __next__(self):
        """

        Returns:
            The next batch of data
        """
        raise NotImplemented()

    def next_item(self, blocking):
        """

        Args:
            blocking: if True, the next elements will block the current thread if not ready

        Returns:
            The next batch of data
        """
        return self.__next__()

    def has_background_jobs(self):
        """
        Returns:
            True if this sequence has a background job to create the next element
        """
        return False

    def has_background_jobs_previous_sequences(self):
        """
        Returns:
            the number of sequences that have background jobs currently running to create the next element
        """
        nb_jobs = 0
        sequences_filled = set()

        sequences_to_examine = [self]
        while len(sequences_to_examine) > 0:
            current = sequences_to_examine.pop()
            if current.has_background_jobs():
                nb_jobs += 1

            sequences_filled.add(current)

            if current.source_split is not None and current.source_split not in sequences_filled:
                sequences_to_examine.append(current.source_split)

        return nb_jobs

    def subsample(self, nb_samples):
        """
        Sub-sample a sequence to a fixed number of samples.

        The purpose is to obtain a smaller sequence, this is particularly useful for the export of augmentations, samples.

        Args:
            nb_samples: the number of samples desired in the original sequence

        Returns:
            a subsampled `Sequence`
        """
        raise NotImplemented()

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        """
        Sub-sample a sequence to samples with specified UIDs.

        Args:
            uids (list): the uids. If `new_sampler` keeps the ordering, then the samples of the resampled sequence should follow `uids` ordering
            uids_name (str): the name of the UIDs
            new_sampler (Sampler): the sampler to be used for the subsampler sequence. If `None`, re-use the existing

        Returns:
            a subsampled `Sequence`
        """
        raise NotImplemented()












