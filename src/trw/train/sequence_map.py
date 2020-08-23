import logging
import collections
from queue import Empty
import functools
import traceback
from trw.train import sequence
from trw.train.job_executor import JobExecutor, default_queue_timeout


logger = logging.getLogger(__name__)


def single_function_to_run(batch, function_to_run):
    """
    apply a list of functions on a batch of data
    """
    for fn in function_to_run:
        batch = fn(batch)
    return batch


class SequenceMap(sequence.Sequence):
    def __init__(self, source_split, nb_workers, function_to_run, max_jobs_at_once=None, worker_post_process_results_fun=None, queue_timeout=default_queue_timeout, preprocess_fn=None, collate_fn=None, ):
        """
        Transform a sequence using a given function.

        .. note:: The map may create more samples than the original sequence.

        :param source_split: the input sequence
        :param function_to_run: the mapping function
        :param nb_workers: the number of workers that will process the split. If 0, no workers will be created.
        :param max_jobs_at_once: the maximum number of results that can be pushed in the result queue at once. If 0, no limit.
            If None, it will be set equal to the number of workers
        :param worker_post_process_results_fun: a function used to post-process the worker results
        :param queue_timeout: the timeout used to pull results from the output queue
        :param preprocess_fn: a function that will preprocess the batch just prior to sending it to the other processes. If `None`,
            no preprocessing performed
        :param collate_fn: a function to collate the batch of data or `None`
        """
        super().__init__(source_split)

        assert isinstance(source_split, sequence.Sequence), '`source_split` must be a `Sequence`'

        if isinstance(function_to_run, collections.Sequence):
            # if we have a list of transforms, wrap them in a single function
            self.function_to_run = functools.partial(single_function_to_run, function_to_run=function_to_run)
        else:
            self.function_to_run = function_to_run
        self.queue_timeout = queue_timeout
        self.preprocess_fn = preprocess_fn
        self.collate_fn = collate_fn

        logger.info('SequenceMap created={}, nb_workers={}, max_jobs_at_once={}'.format(self, nb_workers, max_jobs_at_once))

        self.job_executer = None
        if nb_workers != 0:
            if max_jobs_at_once is None:
                # default: each worker can push at least one item
                # before blocking
                max_jobs_at_once = nb_workers

            self.job_executer = JobExecutor(
                nb_workers=nb_workers,
                function_to_run=self.function_to_run,
                max_jobs_at_once=max_jobs_at_once,
                worker_post_process_results_fun=worker_post_process_results_fun)

        self.iter_source = None
        self.jobs_processed = None
        self.jobs_queued = None
        # self.time_spent_in_blocked_state = 0.0

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        subsampled_source = self.source_split.subsample_uids(uids, uids_name, new_sampler)

        # do not use worker processes: we probably just want a much smaller sequence
        return SequenceMap(
            subsampled_source,
            nb_workers=0,
            function_to_run=self.function_to_run,
            preprocess_fn=self.preprocess_fn,
            collate_fn=self.collate_fn)

    def subsample(self, nb_samples):
        subsampled_source = self.source_split.subsample(nb_samples)

        # do not use worker processes: we probably just want a much smaller sequence
        return SequenceMap(
            subsampled_source,
            nb_workers=0,
            function_to_run=self.function_to_run,
            preprocess_fn=self.preprocess_fn,
            collate_fn=self.collate_fn)

    def fill_queue(self):
        try:
            while not self.job_executer.input_queue.full():
                i = self.iter_source.next_item(blocking=False)
                if self.preprocess_fn is not None:
                    i = self.preprocess_fn(i, source=self)

                self.jobs_queued += 1
                self.job_executer.input_queue.put(i)
        except StopIteration:
            # we are done!
            pass

    def initializer(self):
        """
        Initialize the sequence to iterate through batches
        """
        if self.job_executer is not None:
            self.job_executer.reset()

        self.jobs_processed = 0
        self.jobs_queued = 0

        self.main_thread_list = None
        self.main_thread_index = None

    def __next_local(self, next_fn):
        """
        Get the next elements

        Handles single item or list of items returned by next_fn
        :param next_fn: return the next elements
        """
        if self.main_thread_list is None:
            items = None
            while items is None:
                items = next_fn()

            is_sequence = isinstance(items, collections.Sequence) and not isinstance(items, collections.Mapping)
            if is_sequence:
                # sequence: we need to locally store the sequence and iterate it
                self.main_thread_list = items
                self.main_thread_index = 0
            else:
                # not a sequence: we can safely return the item
                return items

        # manage the iteration of an existing sequence
        if self.main_thread_index >= len(self.main_thread_list):
            raise IndexError(f'BUG! list size={len(self.main_thread_list)}, index={self.main_thread_index}')
        item = self.main_thread_list[self.main_thread_index]
        self.main_thread_index += 1
        if self.main_thread_index == len(self.main_thread_list):
            # we have exhausted our current list of items, resume the `function_to_run` calls
            self.main_thread_list = None
        return item

    def __next__(self):
        return self.next_item(blocking=True)

    def has_background_jobs(self):
        return self.jobs_queued != self.jobs_processed

    def next_item(self, blocking):
        def single_process_next():
            while True:
                i = self.iter_source.__next__()
                if self.preprocess_fn is not None:
                    i = self.preprocess_fn(i, source=self)

                try:
                    items = self.function_to_run(i)

                except Exception as e:
                    # case where we have a job that failed: discard
                    print('-------------- ERROR in worker function --------------')
                    print(e)
                    print('-------------- first job will be aborted --------------')
                    traceback.print_exc()
                    print('-------------------------------------------------------')
                    continue

                return items

        def multiple_process_next():
            nb_background_jobs = self.has_background_jobs_previous_sequences()
            if nb_background_jobs == 0:
                # we are only done once all the jobs have been completed!
                raise StopIteration()

            while True:
                try:
                    items = self.job_executer.output_queue.get(True, timeout=self.queue_timeout)
                    self.jobs_processed += 1
                    if items is None:
                        continue  # the job has failed, get the next item!
                    # ok, we are nood now!
                    break
                except (Empty, KeyboardInterrupt):
                    # no job available, make sure the worker of the other pools are not starved
                    self.fill_queue_all_sequences()
                    if not blocking:
                        raise StopIteration()

            return items

        if self.job_executer is None:
            # use the main thread for the processing. In this case we need to mimic the behaviour
            # of the pool (i.e., if the `function_to_run` returns a list, we need to process one
            # item at a time
            items = self.__next_local(single_process_next)

        else:
            self.fill_queue()
            items = self.__next_local(multiple_process_next)

        if self.collate_fn is not None:
            items = self.collate_fn(items, source=self)
        return items

    def __iter__(self):
        self.initializer()
        self.iter_source = self.source_split.__iter__()
        return self

    def close(self):
        """
        Finish and join the existing pool processes
        """
        if self.job_executer is not None:
            self.job_executer.close()

