import datetime
import os
import traceback
from queue import Empty
import multiprocessing
import logging


logger = logging.getLogger(__name__)

# make sure we start a new process in an empty state
mp = multiprocessing.get_context("spawn")
#mp = multiprocessing.get_context("fork")

# timeout used for the queues
default_queue_timeout = 0.1


class JobExecutor:
    """
    Simple job executor using queues as communication channels for input and output

    Feed jobs using `JobExecutor.input_queue.put(argument)`. `function_to_run` will be called
    with `argument` and the output will be pushed to `JobExecutor.output_queue`

    Jobs that failed will have `None` pushed to the output queue.
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
        logger.info(f'JobExecutor created={self}, nb_workers={nb_workers}, '
                    f'max_jobs_at_once={max_jobs_at_once}, output_queue_size={output_queue_size}')

        self.input_queue = mp.Queue(maxsize=max_jobs_at_once)
        if output_queue_size is None:
            self.output_queue = mp.Queue(maxsize=max_jobs_at_once)
        else:
            self.output_queue = mp.Queue(maxsize=output_queue_size)
        self.nb_workers = nb_workers

        # we can't cancel jobs, so instead record a session ID. If session of
        # the worker and current session ID do not match
        # it means the results of these tasks should be discarded
        self.job_session_id = mp.Value('i', 0)

        self.channel_worker_to_main, self.channel_main_to_worker = mp.Pipe()
        self.must_finish_processes = mp.Value('i', 0)

        self.pool = None
        self.pool = mp.Pool(
            processes=nb_workers,
            initializer=JobExecutor.worker,
            initargs=(self.input_queue, self.output_queue, function_to_run, worker_post_process_results_fun, self.job_session_id, self.channel_worker_to_main, self.channel_main_to_worker, self.must_finish_processes)
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
        if self.pool is not None:
            logger.info(f'closing JobExecutor={self}')

            # clear all the queues so that the jobs are started

            self.reset()  # clear everything
            logger.info(f'JobExecutor queue emptied!')

            # notify all workers that they need to stop
            with self.must_finish_processes.get_lock():
                self.must_finish_processes.value = 1

            self.pool.close()
            logger.info(f'JobExecutor pool closed!')

            # using `torch.multiprocessing.set_sharing_strategy('file_system')`, a deadlock could occur
            # when the workers are terminated. To avoid this, simply terminate the process. At this point
            # we are not interested in the results anyway.
            self.pool.terminate()
            self.pool.join()
            logger.info(f'terminated JobExecutor={self}!')
            self.pool = None

    def reset(self):
        """
        Reset the input and output queues as well as job session IDs.

        The results of the jobs that have not yet been calculated will be discarded
        """
        try:
            while not self.input_queue.empty():
                self.input_queue.get()
        except EOFError:  # in case the other process was already terminated
            pass

        try:
            while not self.output_queue.empty():
                self.output_queue.get()
        except EOFError:  # in case the other process was already terminated
            pass

        # discard the results of the jobs that will not have the
        # current `job_session_id`
        with self.job_session_id.get_lock():
            self.job_session_id.value += 1

    @staticmethod
    def worker(
            input_queue,
            output_queue,
            func,
            post_process_results_fun,
            job_session_id,
            channel_worker_to_main,
            channel_main_to_worker,
            must_finish):
        print('Started worker=', os.getpid(), datetime.datetime.now().time())
        while True:
            item = None
            while True:
                if must_finish.value > 0:
                    # the process was notified to stop, so exit now
                    print('Finished worker=', os.getpid(), datetime.datetime.now().time())
                    return None
                try:
                    # To handle the pool shutdown, we MUST have
                    # a timeout parameter
                    item = input_queue.get(True, timeout=default_queue_timeout)
                except (Empty, KeyboardInterrupt):
                    # The queue is empty, so we have to wait more.
                    continue

                # we have a value, exit the loop to proceed
                break
            # print('Worker | ', os.getpid(), '| job retrieved | ', datetime.datetime.now().time())

            with job_session_id.get_lock():
                started_session_id = job_session_id.value

            try:
                results = func(item)
                # results may be `None`, in that case we simply fetch the next result

                if post_process_results_fun is not None and results:
                    # channels are reversed for the communication worker->main thread
                    post_process_results_fun(
                        results,
                        channel_main_to_worker=channel_worker_to_main,
                        channel_worker_to_main=channel_main_to_worker)
            except Exception as e:
                print('-------------- ERROR in worker function --------------')
                print(e)
                print('-------------- first job will be aborted --------------')
                traceback.print_exc()
                print('-------------------------------------------------------')

                results = None

            try:
                with job_session_id.get_lock():
                    current_session_id = job_session_id.value

                    # if the session ID has changed (before and after running the job), then we want to discard
                    # the result of these jobs
                    if current_session_id == started_session_id:
                        output_queue.put(results)
                    else:
                        pass
            except Exception as e:
                print('-------------- ERROR in worker function: could not push on the queue --------------')
                print(e)
                print('-------------- first job will be aborted --------------')
