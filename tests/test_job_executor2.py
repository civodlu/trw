import time
import unittest
import torch
import trw
import numpy as np


def create_large_data(data):
    data['data'] = torch.zeros([512, 512, 512], dtype=torch.float32)
    #data['data'] = np.zeros([512, 512, 512], dtype=np.float32)
    return data


class TestJobExecutor2(unittest.TestCase):
    def test_put_get(self):
        timeout = 10.0

        executor = trw.train.JobExecutor2(5, create_large_data, nb_pin_threads=5)

        print('STARTED')

        nb_jobs = 10
        for i in range(nb_jobs):
            executor.put({'job_id': i})

        jobs_performed = []
        time_start = time.perf_counter()
        while True:
            if not executor.pin_memory_queue.empty():
                r = executor.pin_memory_queue.get()
                jobs_performed.append(r)
                print('JOB DONE')

            time_end = time.perf_counter()
            if time_end - time_start >= timeout:
                assert False, f'jobs were not processed within timeout (deadlock?)! ' \
                              f'Got={len(jobs_performed)}, expected={nb_jobs}'

            if len(jobs_performed) == nb_jobs:
                break

        assert executor.jobs_queued == nb_jobs
        assert executor.jobs_processed.value == nb_jobs

        print('DONE, time=', time_end - time_start)
        executor.close()
        print('COMPLETELY DONE')

if __name__ == '__main__':
    unittest.main()
