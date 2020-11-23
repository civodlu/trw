import logging
import collections
import time
from queue import Empty
import functools
import traceback
from trw.train import sequence
from trw.train.job_executor2 import JobExecutor2, default_queue_timeout


logger = logging.getLogger(__name__)


def single_function_to_run(batch, function_to_run):
    """
    apply a list of functions on a batch of data
    """
    for fn in function_to_run:
        batch = fn(batch)
    return batch


class SequenceMap(sequence.Sequence):
    def __init__(
            self,
            source_split,
            nb_workers,
            function_to_run,
            max_jobs_at_once=None,
            nb_pin_threads=None,
            queue_timeout=default_queue_timeout,
            debug_job_report_timeout=30.0,
            collate_fn=None):
        """
        Transform a sequence using a given function.

        .. note:: The map may create more samples than the original sequence.

        :param source_split: the input sequence
        :param function_to_run: the mapping function
        :param nb_workers: the number of workers that will process the split. If 0, no workers will be created.
        :param max_jobs_at_once: the maximum number of results that can be pushed in the result queue per process
            at once. If 0, no limit. If None, it will be set equal to the number of workers
        :param queue_timeout: the timeout used to pull results from the output queue
        :param collate_fn: a function to collate the batch of data or `None`
        :param nb_pin_threads: the number of threads to be used to collect the data from the worker process
            to the main process
        :param debug_job_report_timeout: timeout after which if no job were processed a job report will be
            printed (e.g., to debug pipeline stalling)
        """
        super().__init__(source_split)

        assert isinstance(source_split, sequence.Sequence), '`source_split` must be a `Sequence`'

        if isinstance(function_to_run, collections.Sequence):
            # if we have a list of transforms, wrap them in a single function
            self.function_to_run = functools.partial(single_function_to_run, function_to_run=function_to_run)
        else:
            self.function_to_run = function_to_run
        self.queue_timeout = queue_timeout
        self.collate_fn = collate_fn
        self.debug_job_report_timeout = debug_job_report_timeout

        logger.info(f'SequenceMap created={self}, nb_workers={nb_workers}, max_jobs_at_once={max_jobs_at_once}')

        self.job_executor = None
        if nb_workers != 0:
            if max_jobs_at_once is None:
                # default: each worker can push at least one item
                # before blocking
                max_jobs_at_once = nb_workers

            self.job_executor = JobExecutor2(
                nb_workers=nb_workers,
                function_to_run=self.function_to_run,
                nb_pin_threads=nb_pin_threads,
                max_queue_size_per_worker=max_jobs_at_once)

        self.iter_source = None
        self.jobs_processed = None
        self.jobs_queued = None
        self.sequence_iteration_finished = None
        self.main_thread_list = None
        self.main_thread_index = None

        self.debug_time_to_get_next_item = 0
        self.debug_nb_items = 0

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        subsampled_source = self.source_split.subsample_uids(uids, uids_name, new_sampler)

        # do not use worker processes: we probably just want a much smaller sequence
        return SequenceMap(
            subsampled_source,
            nb_workers=0,
            function_to_run=self.function_to_run,
            collate_fn=self.collate_fn)

    def subsample(self, nb_samples):
        subsampled_source = self.source_split.subsample(nb_samples)

        # do not use worker processes: we probably just want a much smaller sequence
        return SequenceMap(
            subsampled_source,
            nb_workers=0,
            function_to_run=self.function_to_run,
            collate_fn=self.collate_fn)

    def fill_queue(self):
        try:
            while not self.job_executor.is_full():
                i = self.iter_source.next_item(blocking=False)
                self.jobs_queued += 1
                self.job_executor.put(i)
        except StopIteration:
            # we are done!
            self.sequence_iteration_finished = True

    def initializer(self):
        """
        Initialize the sequence to iterate through batches
        """
        if self.job_executor is not None:
            self.job_executor.reset()

        self.jobs_processed = 0
        self.jobs_queued = 0

        self.main_thread_list = None
        self.main_thread_index = None
        self.sequence_iteration_finished = False

    def __next_local(self, next_fn):
        """
        Get the next elements

        Handles single item or list of items returned by next_fn
        :param next_fn: return the next elements
        """
        if self.main_thread_list is None:
            items = None
            while items is None or len(items) == 0:
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
        return not self.job_executor.is_idle()

    def next_item(self, blocking):
        def single_process_next():
            while True:
                i = self.iter_source.__next__()

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
            assert self.main_thread_list is None
            nb_background_jobs = self.has_background_jobs_previous_sequences()

            # we are only done once all the jobs have been completed!
            if self.sequence_iteration_finished and \
                    nb_background_jobs == 0 and \
                    self.job_executor.pin_memory_queue.empty():

                # collect some useful statistics
                if self.debug_nb_items != 0:
                    logger.debug(f'SequenceMap={self}, nb_items_processed={self.debug_nb_items},'
                                 f'item_overhead_sequence_time_average='
                                 f'{self.debug_time_to_get_next_item / self.debug_nb_items}')

                # stop the sequence
                raise StopIteration()

            report_timeout_start = time.time()
            next_item_start = time.perf_counter()
            while True:
                try:
                    items = self.job_executor.pin_memory_queue.get(True, timeout=self.queue_timeout)
                    if items is None:
                        continue  # the job has failed, get the next item!
                    self.jobs_processed += 1
                    # ok, we are good now!
                    break
                except Empty:
                    # no job available, make sure the worker of the other pools are not starved
                    self.fill_queue_all_sequences()
                    if not blocking:
                        raise StopIteration()

                    if time.time() - report_timeout_start > self.debug_job_report_timeout:
                        print('------------------- STALLING -------------------')
                        report_timeout_start = time.time()
                        self.job_executor.job_report()

                    nb_background_jobs = self.has_background_jobs_previous_sequences()

                    # we are only done once all the jobs have been completed!
                    if self.sequence_iteration_finished and \
                            nb_background_jobs == 0 and \
                            self.job_executor.pin_memory_queue.empty():
                        print('--------------- IDLE STOP ----------------')
                        raise StopIteration()


            next_item_end = time.perf_counter()
            self.debug_time_to_get_next_item += next_item_end - next_item_start
            self.debug_nb_items += 1
            return items

        if self.job_executor is None:
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
        if self.job_executor is not None:
            self.job_executor.close()

