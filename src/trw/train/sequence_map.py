import os
import datetime
import logging
import collections
from torch import multiprocessing as mp
from queue import Empty
import functools

from trw.train import sequence


logger = logging.getLogger(__name__)

# timeout used for the queues
default_queue_timeout = 0.1


def single_function_to_run(batch, function_to_run):
    """
    apply a list of functions on a batch of data
    """
    for fn in function_to_run:
        batch = fn(batch)
    return batch


class JobExecutor:
    """
    Simple job executor using queues as communication channels for input and output

    Feed jobs using `JobExecutor.input_queue.put(argument)`. `function_to_run` will be called
    with `argument` and the output will be pushed to `JobExecutor.output_queue`


    .. todo:: On windows there are many issues related to shared memory: we would like to let the worker to create large memory arrays
        for example, when the datasets are extremely large, we often want to asynchronously load
        these datasets BUT in python < 3.8 (multiprocessing) and also in pytorch <= 1.1 (torch.multiprocessing), there doesn't seem to have standard
        and portable memory sharing facilities and will revert to file based sharing
    """
    def __init__(self, nb_workers, function_to_run, max_jobs_at_once=0, worker_post_process_results_fun=None, output_queue_size=0):
        """
        :param nb_workers: the number of workers to process the jobs
        :param max_jobs_at_once: the maximum number of jobs active (but not necessarily run) at once. If `JobExecutor.output_queue` is larger than `max_jobs_at_once`, the queue will block
        until more jobs are completed downstream. If None, there is no limit
        :param function_to_run: a function with argument `item` to be processed. The return values will be pushed to `JobExecutor.output_queue`
        :param worker_post_process_results_fun: a function used to post-process the worker results. It takes as input `results, channel_main_to_worker, channel_worker_to_main`
        :param output_queue_size: the output queue size. If `0`, no maximum size. If `None`, same as `max_jobs_at_once`
        """
        self.input_queue = mp.Queue(maxsize=max_jobs_at_once)
        if output_queue_size is None:
            self.output_queue = mp.Queue(maxsize=max_jobs_at_once)
        else:
            self.output_queue = mp.Queue(maxsize=output_queue_size)
        self.nb_workers = nb_workers

        # we can't cancel jobs, so instead record a session ID. If session of the worker and current session ID do not match
        # it means the results of these tasks should be discarded
        self.job_session_id = mp.Value('i', 0)

        self.channel_worker_to_main, self.channel_main_to_worker = mp.Pipe()

        self.pool = mp.Pool(
            processes=nb_workers,
            initializer=JobExecutor.worker,
            initargs=(self.input_queue, self.output_queue, function_to_run, worker_post_process_results_fun, self.job_session_id, self.channel_worker_to_main, self.channel_main_to_worker)
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.input_queue.empty():
            logging.warning('JobExecutor terminated. Not all jobs may not have been completed!')

        self.close()

    def __enter__(self):
        return self

    def __del__(self):
        self.close()

    def close(self):
        """
        Terminate all jobs
        """
        self.pool.terminate()
        self.pool.join()

    def reset(self):
        """
        Reset the input and output queues as well as job session IDs.

        The results of the jobs that have not yet been calculated will be discarded
        """
        while not self.input_queue.empty():
            self.input_queue.get()

        while not self.output_queue.empty():
            self.output_queue.get()

        # discard the results of the jobs that will not have the
        # current `job_session_id`
        with self.job_session_id.get_lock():
            self.job_session_id.value += 1

    @staticmethod
    def worker(input_queue, output_queue, func, post_process_results_fun, job_session_id, channel_worker_to_main, channel_main_to_worker):
        print('Started worker=', os.getpid(), datetime.datetime.now().time())
        while True:
            # TODO manage exceptions
            # blocking call to get the next work item

            #print('Worker | ', os.getpid(), '| retrieving job | ', datetime.datetime.now().time())
            item = input_queue.get(True)
            #print('Worker | ', os.getpid(), '| job retrieved | ', datetime.datetime.now().time())

            with job_session_id.get_lock():
                started_session_id = job_session_id.value

            results = func(item)
            assert results is not None, '`func`={} must result a result!'.format(func)

            if post_process_results_fun is not None:
                # channels are reversed for the communication worker->main thread
                post_process_results_fun(results, channel_main_to_worker=channel_worker_to_main, channel_worker_to_main=channel_main_to_worker)
                #post_process_results_fun(results, channel_main_to_worker=channel_main_to_worker, channel_worker_to_main=channel_worker_to_main)

            with job_session_id.get_lock():
                current_session_id = job_session_id.value

                # if the session ID has changed (before and after running the job), then we want to discard
                # the result of these jobs
                if current_session_id == started_session_id:
                    #print('Worker | ', os.getpid(), '| job finished. Posting | ', datetime.datetime.now().time())
                    output_queue.put(results)
                    #print('Worker | ', os.getpid(), '| job finished. Posted | ', datetime.datetime.now().time())
                else:
                    #print('Worker | ', os.getpid(), '| job DISCARDED. Posted | ', datetime.datetime.now().time())
                    pass


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
        # print('fill_queue', os.getpid(), 'NOW=', datetime.datetime.now().time(), self.function_to_run)
        try:
            while not self.job_executer.input_queue.full():
                i = self.iter_source.next_item(blocking=False)
                if self.preprocess_fn is not None:
                    i = self.preprocess_fn(i, source=self)

                # print('QUEUING', os.getpid(), 'NOW=', datetime.datetime.now().time(), 'QUEUE_SIZE=', self.job_executer.output_queue.qsize())
                self.jobs_queued += 1
                self.job_executer.input_queue.put(i)
                # print('QUEUED', os.getpid())
        except StopIteration:
            # we are done!
            pass
            # print('queue_filled', os.getpid(), ' jobs=', self.job_executer.input_queue.qsize(), 'NOW=', datetime.datetime.now().time())

    def initializer(self):
        """
        Initialize the sequence to iterate through batches
        """
        # print('TIME BLOCKED=', self.time_spent_in_blocked_state)
        if self.job_executer is not None:
            self.job_executer.reset()

        self.jobs_processed = 0
        self.jobs_queued = 0

        self.main_thread_list = None
        self.main_thread_index = None
        # self.time_spent_in_blocked_state = 0.0

    def __next_local(self, next_fn):
        """
        Get the next elements

        Handles single item or list of items returned by next_fn
        :param next_fn: return the next elements
        """
        if self.main_thread_list is None:
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
            i = self.iter_source.__next__()
            if self.preprocess_fn is not None:
                i = self.preprocess_fn(i, source=self)
            items = self.function_to_run(i)
            return items

        def multiple_process_next():
            nb_background_jobs = self.has_background_jobs_previous_sequences()
            # queue_size = self.job_executer.input_queue.qsize() + self.job_executer.output_queue.qsize()
            if nb_background_jobs == 0:
                # we are only done once all the jobs have been completed!
                # print('STOP', self, queue_size)
                raise StopIteration()

            while True:
                try:
                    # time_blocked_start = time.time()
                    items = self.job_executer.output_queue.get(True, timeout=self.queue_timeout)
                    # time_blocked_end = time.time()
                    # self.time_spent_in_blocked_state += time_blocked_end - time_blocked_start
                    break
                except Empty:
                    # no job available, make sure the worker of the other pools are not starved
                    self.fill_queue_all_sequences()
                    # self.fill_queue()
                    if not blocking:
                        raise StopIteration()

            # print('ITEM', os.getpid(), 'BEING_PROCESSED=', self.jobs_queued - self.jobs_processed)
            self.jobs_processed += 1
            return items

        if self.job_executer is None:
            # use the main thread for the processing. In this case we need to mimic the behaviour
            # of the pool (i.e., if the `function_to_run` returns a list, we need to process one
            # item at a time
            items = self.__next_local(single_process_next)

        else:
            # output_queue_size = -1
            # if is_windows_platform:
            #    output_queue_size = self.job_executer.output_queue.qsize()
            # logger.info('SequenceMap={}, job result ready={}'.format(self, output_queue_size))

            # self.fill_queue_all_sequences()  # TODO assess performance degradation. For some pipelines it may be advantageous to have `fill_queue_all_sequences`?
            self.fill_queue()
            items = self.__next_local(multiple_process_next)

        if self.collate_fn is not None:
            items = self.collate_fn(items, source=self)
        return items

    def __iter__(self):
        self.initializer()
        self.iter_source = self.source_split.__iter__()
        return self

