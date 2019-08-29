import trw
from unittest import TestCase


keep_ids = set()


def fn_to_run_identity(data):
    return data


def fn_to_run_add_msg(data):
    data['message'] = 'done'
    return data


def post_process_results_fun(data, channel_main_to_worker, channel_worker_to_main):
    keep_ids.add(data)

    while channel_main_to_worker.poll():
        message = channel_main_to_worker.recv()
        assert message == 'test'
    assert len(keep_ids) == 1

    channel_worker_to_main.send('added test')


class TestJobExecutor(TestCase):
    def test_job_executor_simple_processing(self):
        """
        Process a series of jobs. Make sure all jobs are processed exactly once
        """
        executor = trw.train.JobExecutor(nb_workers=2, function_to_run=fn_to_run_add_msg, max_jobs_at_once=0)

        # post jobs
        for i in range(10):
            executor.input_queue.put({'job': i})

        jobs_processes = set()
        nb_samples = 0
        while nb_samples < 10:
            r = executor.output_queue.get()
            jobs_processes.add(r['job'])
            print(r)
            assert r['message'] == 'done'
            nb_samples += 1
        assert len(jobs_processes) == 10

    def test_job_executor_with_postprocessing(self):
        """
        Make sure we can send and receive message from and to the worker directly
        """
        executor = trw.train.JobExecutor(nb_workers=1, function_to_run=fn_to_run_identity, worker_post_process_results_fun=post_process_results_fun)

        # send a direct message to the worker
        executor.channel_main_to_worker.send('test')

        # post a job
        executor.input_queue.put('test')

        # wait for the worker and retrieve result
        result = executor.output_queue.get()
        self.assertTrue(result == 'test')

        worker_msg = executor.channel_worker_to_main.recv()
        self.assertTrue(worker_msg == 'added test')

        assert len(keep_ids) == 0  # the set is not shared between children and parent
