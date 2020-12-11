import copy
import io
import os
import sys
import threading
import time
import traceback

from time import sleep, perf_counter

from threadpoolctl import threadpool_limits
from typing import Callable, List, Optional

from trw.basic_typing import Batch
import logging
import numpy as np
from queue import Queue as ThreadQueue, Empty

# Make sure we start a new process in an empty state so
# that Windows/Linux environment behave the similarly
import multiprocessing

from trw.utils.graceful_killer import GracefulKiller

multiprocessing = multiprocessing.get_context("spawn")
#multiprocessing = multiprocessing.get_context("fork")
from multiprocessing import Event, Process, Queue, Value

# timeout used for the queues
default_queue_timeout = 0.1


class JobMetadata:
    def __init__(self, job_session_id):
        self.job_created = time.perf_counter()
        self.job_processing_finished = None
        self.job_results_queued = None

        # pinning thread
        self.job_pin_thread_received = None
        self.job_pin_thread_queued = None

        self.job_session_id = job_session_id


def worker(
        input_queue: Queue,
        output_queue: Queue,
        transform: Callable[[Batch], Batch],
        global_abort_event: Event,
        local_abort_event: Event,
        wait_time: float,
        seed: int) -> None:
    """
    Worker that will execute a transform on a process.

    Args:
        input_queue: the queue to listen to
        output_queue:  the queue to output the results
        transform: the transform to be applied on each data queued
        global_abort_event: specify when the jobs need to shutdown
        local_abort_event: specify when the jobs need to shutdown but only for a given job executor
        wait_time: process will sleep this amount of time when input queue is empty
        seed: an int to seed random generators

    Returns:
        None
    """

    np.random.seed(seed)
    item = None
    job_session_id = None
    job_metadata = None
    #print(f'Worker={os.getpid()} Started!!')
    while True:
        try:
            #print('Worker: Retrieving job')
            if not global_abort_event.is_set() and not local_abort_event.is_set():
                if item is None:
                    if not input_queue.empty():
                        try:
                            job_session_id, item = input_queue.get()
                            job_metadata = JobMetadata(job_session_id=job_session_id)
                        except Exception as e:
                            # possible exception:  `unable to open shared memory object </torch_XXX_YYYYY>
                            # we MUST queue a `None` to specify that we received something but there was an error
                            print(f'Exception <input_queue.get> in background worker PID={os.getpid()}, E={e}')
                            item = None
                            # DO continue: we want to push `None`

                    else:
                        sleep(wait_time)
                        continue

                    if transform is not None and item is not None:
                        try:
                            item = transform(item)
                            job_metadata.job_processing_finished = time.perf_counter()
                            #print('Worker: processing=', item)
                        except Exception as e:
                            # exception is intercepted and skip to next job
                            # here we send the `None` result anyway to specify the
                            # job failed. we MUST send the `None` so that jobs queued
                            # and jobs processed match.
                            print('-------------- ERROR in worker function --------------')
                            print(f'Exception in background worker PID={os.getpid()}, E={e}')
                            print('-------------- first job will be aborted --------------')
                            string_io = io.StringIO()
                            traceback.print_exc(file=string_io)
                            print(string_io.getvalue())
                            print('-------------------------------------------------------')
                            item = None

                while True:
                    try:
                        job_metadata.job_results_queued = time.perf_counter()
                        output_queue.put((job_metadata, item))
                        item = None
                        break  # success, get ready to get a new item from the queue

                    except Exception as e:
                        # exception is intercepted and skip to next job
                        print(f'Exception <output_queue.put> in background worker '
                              f'thread_id={os.getpid()}, E={e}, ITEM={item}, id={job_session_id}')

                        # re-try to push on the queue!
                        sleep(wait_time)
                        continue

            else:
                print(f'Worker={os.getpid()} Stopped (abort_event SET)!!')
                return

        except KeyboardInterrupt:
            global_abort_event.set()
            print(f'Worker={os.getpid()} Stopped (KeyboardInterrupt)!!')
            return

        except Exception as e:
            # exception is intercepted and skip to next job
            print(f'Exception in background worker thread_id={os.getpid()}, E={e}, ITEM={item}, id={job_session_id}')
            continue


def collect_results_to_main_process(
        job_session_id: Value,
        jobs_queued: Value,
        worker_output_queue: Queue,
        output_queue: ThreadQueue,
        global_abort_event: Event,
        local_abort_event: Event,
        wait_time: float) -> None:

    assert output_queue is not None
    item = None
    item_job_session_id = None
    while True:
        try:
            if global_abort_event.is_set() or local_abort_event.is_set():
                print(f'Thread={threading.get_ident()}, (abort_event set) shutdown!')
                return

            # if we don't have an item we need to fetch it first. If the queue we want to get it from it empty, try
            # again later
            if item is None and item_job_session_id is None:
                if not worker_output_queue.empty():
                    try:
                        #time_queue_start = perf_counter()
                        job_metadata, item = worker_output_queue.get(timeout=wait_time)
                        item_job_session_id = job_metadata.job_session_id
                        job_metadata.job_pin_thread_received = time.perf_counter()
                        #time_queue_end = perf_counter()

                    except Empty:
                        # even if the `current_queue` was not empty, another thread might have stolen
                        # the job result already. Just continue to the next queue
                        continue

                    except RuntimeError as e:
                        print(f'collect_results_to_main_process Queue={threading.get_ident()} GET error={e}')
                        # the queue was sending something but failed
                        # discard this data and continue
                        item = None

                    #print('PINNING item_job_session_id=', item_job_session_id, ' current=', job_session_id.value, 'ITEM=', item)

                    if item is None:
                        # this job FAILED so there is no result to queue. Yet, we increment the
                        # job counter since this is used to monitor if the executor is
                        # idle
                        with jobs_queued.get_lock():
                            jobs_queued.value += 1
                            #print(f'PUSH NONE ---- jobs_queued={jobs_queued.value}')

                        # fetch a new job result!
                        item_job_session_id = None
                        continue

                else:
                    sleep(wait_time)
                    continue

            if item is None and item_job_session_id is None:
                continue

            if item_job_session_id != job_session_id.value:
                # this is an old result belonging to the previous
                # job session. Discard it and process a new one
                item = None
                item_job_session_id = None
                with jobs_queued.get_lock():
                    jobs_queued.value += 1
                continue

            if not output_queue.full():
                #print(f'Pinning thread output queue filled! item={item}')
                job_metadata.job_pin_thread_queued = time.perf_counter()
                output_queue.put((job_metadata, item))

                item_job_session_id = None
                with jobs_queued.get_lock():
                    jobs_queued.value += 1
                    #print(f'PUSH JOB={item} ---- jobs_queued={jobs_queued.value}')

                item = None
            else:
                sleep(wait_time)
                continue
        except KeyboardInterrupt:
            print(f'Thread={threading.get_ident()}, thread shut down (KeyboardInterrupt)')
            global_abort_event.set()
            raise KeyboardInterrupt
        except Exception as e:
            print(f'Thread={threading.get_ident()}, thread shut down (Exception)')
            local_abort_event.set()
            raise e


class JobExecutor2:
    """
    Execute jobs on multiple processes.

    At a high level, we have worker executing on multiple processes. Each worker will be fed
    by an input queue and results of the processing pushed to an output queue.

    Pushing data on a queue is very fast BUT retrieving it from a different process takes time.
    Even if PyTorch claims to have memory shared arrays, retrieving a large array from a queue
    has a linear runtime complexity. To limit this copy penalty, we can use threads that copy
    from the worker process to the main process (`pinning` threads. Here, sharing data between
    threads is almost free).

    Notes:
        This class was designed for maximum speed and not reproducibility in mind.
        The processed of jobs will not keep their ordering.
    """
    def __init__(
            self,
            nb_workers: int,
            function_to_run: Callable[[Batch], Batch],
            max_queue_size_per_worker: int = 2,
            max_queue_size_pin_thread_per_worker: int = 3,
            wait_time: float = 0.01,
            wait_until_processes_start: bool = True):
        """

        Args:
            nb_workers: the number of workers (processes) to execute the jobs
            function_to_run: the job to be executed
            max_queue_size_per_worker: define the maximum number of job results that can be stored
                before a process is blocked (i.e., larger queue size will improve performance but will
                require more memory to store the results). the pin_thread need to process the result before
                the blocked process can continue its processing.
            max_queue_size_pin_thread_per_worker: define the maximum number of results available on the main
                process (i.e., larger queue size will improve performance but will require more memory
                to store the results).
            wait_time: the default wait time for a process or thread to sleep if no job is available
            wait_until_processes_start: if True, the main process will wait until the worker processes and
                pin threads are fully running
        """
        self.wait_until_processes_start = wait_until_processes_start
        self.wait_time = wait_time
        self.max_queue_size_pin_thread_per_worker = max_queue_size_pin_thread_per_worker
        self.max_queue_size_per_worker = max_queue_size_per_worker
        self.function_to_run = function_to_run
        self.nb_workers = nb_workers

        self.global_abort_event = GracefulKiller.abort_event
        self.local_abort_event = Event()

        self.main_process = os.getpid()

        self.worker_control = 0
        self.worker_input_queues = []
        self.worker_output_queues = []
        self.processes = []

        self.jobs_processed = Value('i', 0)
        self.jobs_queued = 0

        self.pin_memory_threads = []
        self.pin_memory_queue = None

        # we can't cancel jobs, so instead record a session ID. If session of
        # the worker and current session ID do not match
        # it means the results of these tasks should be discarded
        self.job_session_id = Value('i', 0)

        self.start()

    def start(self, timeout: float = 10.0) -> None:
        """
        Start the processes and queues.

        Args:
            timeout:

        Returns:

        """
        if self.pin_memory_queue is None:
            self.pin_memory_queue = ThreadQueue(self.max_queue_size_pin_thread_per_worker * self.nb_workers)

        if self.nb_workers == 0:
            # nothing to do, this will be executed synchronously on
            # the main process
            return

        if len(self.processes) != self.nb_workers:
            print(f'Starting jobExecutor={self}, on process={os.getpid()} nb_workers={self.nb_workers}')
            logging.debug(f'Starting jobExecutor={self}, on process={os.getpid()} nb_workers={self.nb_workers}')
            if len(self.processes) > 0 or len(self.pin_memory_threads) > 0:
                self.close()
            self.local_abort_event.clear()

            with threadpool_limits(limits=1, user_api='blas'):
                for i in range(self.nb_workers): #maxsize = 0
                    self.worker_input_queues.append(Queue(maxsize=self.max_queue_size_per_worker))
                    self.worker_output_queues.append(Queue(self.max_queue_size_per_worker))

                    p = Process(
                        target=worker,
                        name=f'JobExecutorWorker-{i}',
                        args=(
                            self.worker_input_queues[i],
                            self.worker_output_queues[i],
                            self.function_to_run,
                            self.global_abort_event,
                            self.local_abort_event,
                            self.wait_time, i
                        ))
                    #p.daemon = True
                    p.start()
                    self.processes.append(p)
                    print(f'Worker={p.pid} started!')
                    logging.debug(f'Child process={p.pid} for jobExecutor={self}')

            # allocate one thread per process to move the data from the process memory space
            # to the main process memory
            self.pin_memory_threads = []
            for i in range(self.nb_workers):
                pin_memory_thread = threading.Thread(
                    name=f'JobExecutorThreadResultCollector-{i}',
                    target=collect_results_to_main_process,
                    args=(
                        self.job_session_id,
                        self.jobs_processed,
                        self.worker_output_queues[i],
                        self.pin_memory_queue,
                        self.global_abort_event,
                        self.local_abort_event,
                        self.wait_time
                    ))
                self.pin_memory_threads.append(pin_memory_thread)
                #pin_memory_thread.daemon = True
                pin_memory_thread.start()
                print(f'Thread={pin_memory_thread.ident}, thread started')

            self.worker_control = 0

        if self.wait_until_processes_start:
            # wait until all the processes and threads are alive
            waiting_started = perf_counter()
            while True:
                wait_more = False
                for p in self.processes:
                    if not p.is_alive():
                        wait_more = True
                        continue
                for t in self.pin_memory_threads:
                    if not t.is_alive():
                        wait_more = True
                        continue

                if wait_more:
                    waiting_time = perf_counter() - waiting_started
                    if waiting_time < timeout:
                        sleep(self.wait_time)
                    else:
                        logging.error('the worker processes/pin threads were too slow to start!')

                break
        logging.debug(f'jobExecutor ready={self}')

    def close(self, timeout: float = 10) -> None:
        """
        Stops the processes and threads.

        Args:
            timeout: time allowed for the threads and processes to shutdown cleanly
                before using `terminate()`

        """

        if os.getpid() != self.main_process:
            logging.error(f'attempting to close the executor from a '
                          f'process={os.getpid()} that did not create it! ({self.main_process})')
            return

        # notify all the threads and processes to be shut down
        print('Setting `abort_event` to interrupt Processes and threads! (JobExecutor)')
        self.local_abort_event.set()

        # give some time to the threads/processes to shutdown normally
        shutdown_time_start = perf_counter()
        while True:
            wait_more = False
            if len(self.processes) != 0:
                for p in self.processes:
                    if p.is_alive():
                        wait_more = True
                        break
            if len(self.pin_memory_threads):
                for t in self.pin_memory_threads:
                    if t.is_alive():
                        wait_more = True
                        break

            shutdown_time = perf_counter() - shutdown_time_start
            if wait_more:
                if shutdown_time < timeout:
                    sleep(0.1)
                    continue
                else:
                    logging.error('a job did not respond to the shutdown request in the allotted time. '
                                  'It could be that it needs a longer timeout or a deadlock. The processes'
                                  'will now be forced to shutdown!')

            # done normal shutdown or timeout
            break

        if len(self.processes) != 0:
            logging.debug(f'JobExecutor={self}: shutting down workers...')
            [i.terminate() for i in self.processes]

            for i, p in enumerate(self.processes):
                self.worker_input_queues[i].close()
                self.worker_input_queues[i].join_thread()

                self.worker_output_queues[i].close()
                self.worker_output_queues[i].join_thread()

            self.worker_input_queues = []
            self.worker_output_queues = []
            self.processes = []

        if len(self.pin_memory_threads) > 0:
            for thread in self.pin_memory_threads:
                thread.join()
                del thread
            self.pin_memory_threads = []

            del self.pin_memory_queue
            self.pin_memory_queue = None

    def is_full(self) -> bool:
        """
        Check if the worker input queues are full.

        Returns:
            True if full, False otherwise
        """
        if self.nb_workers == 0:
            return False

        for i in range(self.nb_workers):
            queue = self.worker_input_queues[self.worker_control]
            if not queue.full():
                return False
            self.worker_control = (self.worker_control + 1) % self.nb_workers

        return True

    def put(self, data: Batch) -> bool:
        """
        Queue a batch of data to be processed.

        Warnings:
            if the queues are full, the batch will NOT be appended

        Args:
            data: a batch of data to process

        Returns:
            True if the batch was successfully appended, False otherwise.
        """
        if self.nb_workers == 0:
            # if no asynchronous worker used, put the result
            # directly on the pin queue
            batch_in = copy.deepcopy(data)
            job_metadata = JobMetadata(job_session_id=0)
            batch_out = self.function_to_run(batch_in)
            job_metadata.job_processing_finished = time.perf_counter()
            job_metadata.job_results_queued = job_metadata.job_processing_finished
            job_metadata.job_pin_thread_received = job_metadata.job_processing_finished
            job_metadata.job_pin_thread_queued = job_metadata.job_processing_finished
            self.pin_memory_queue.put((job_metadata, batch_out))
            self.jobs_queued += 1
            return True
        else:
            for i in range(self.nb_workers):
                queue = self.worker_input_queues[self.worker_control]
                if not queue.full():
                    queue.put((self.job_session_id.value, data))
                    self.worker_control = (self.worker_control + 1) % self.nb_workers
                    self.jobs_queued += 1
                    return True

            # all queues are full, we have to wait
            return False

    def is_idle(self) -> bool:
        """
        Returns:
            True if the executor is not currently processing jobs
        """
        with self.jobs_processed.get_lock():
            return self.jobs_processed.value == self.jobs_queued

    def job_report(self, f=sys.stdout):
        """
        Summary of the executor state. Useful for debugging.
        """
        f.write(f'JobExecutor={self}, Main process={os.getpid()}, main thread={threading.get_ident()}\n')
        f.write(f'NbProcesses={len(self.processes)}, NbThreads={len(self.pin_memory_threads)}\n')
        for p in self.processes:
            f.write(f'  worker PID={p.pid}, is_alive={p.is_alive()}\n')

        for i, q in enumerate(self.worker_input_queues):
            f.write(f'  worker_input_queue {i} is_empty={q.empty()}, is_full={q.full()}\n')

        for i, q in enumerate(self.worker_output_queues):
            f.write(f'  worker_output_queue {i} is_empty={q.empty()}, is_full={q.full()}\n')

        q = self.pin_memory_queue
        f.write(f'  pin_memory_queue is_empty={q.empty()}, is_full={q.full()}\n')

        for t in self.pin_memory_threads:
            f.write(f'  thread IDENT={t.ident}, is_alive={t.is_alive()}\n')

        f.write(f'nb_jobs_received={self.jobs_queued}, nb_jobs_processed={self.jobs_processed.value}, job_session_id={self.job_session_id.value}\n')

    def reset(self):
        """
        Reset the input and output queues as well as job session IDs.

        The results of the jobs that have not yet been calculated will be discarded
        """

        # here we could clear the queues for a faster implementation.
        # Unfortunately, this is not an easy task to properly
        # counts all the jobs processed or discarded due to the
        # multi-threading. Instead, all tasks queued are executed
        # and we use a `job_session_id` to figure out the jobs to be
        # discarded
        """
        # empty the various queues
        try:
            for input_queue in self.worker_input_queues:
                while not input_queue.empty():
                    input_queue.get()
        except EOFError:  # in case the other process was already terminated
            pass

        try:
            for output_queue in self.worker_output_queues:
                while not output_queue.empty():
                    output_queue.get()
        except EOFError:  # in case the other process was already terminated
            pass
            
        with self.jobs_processed.get_lock():
            self.jobs_processed.value = 0
        self.jobs_queued = 0
        """

        # empty the current queue results, they are not valid anymore!
        try:
            while not self.pin_memory_queue.empty():
                self.pin_memory_queue.get()
        except EOFError:  # in case the other process was already terminated
            pass
        # discard the results of the jobs that will not have the
        # current `job_session_id`
        with self.job_session_id.get_lock():
            self.job_session_id.value += 1

    def __del__(self):
        #logging.debug(f'JobExecutor={self}: destructor called')
        self.close()
