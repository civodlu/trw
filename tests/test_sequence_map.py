import numpy as np
import time
import datetime
import trw.train
import torch
from unittest import TestCase
import collections
import os
import copy


def load_fake_volumes_npy(item):
    v = np.ndarray([512, 512, 500], dtype=np.float32)
    v.fill(float(item['values']))

    r = {
        'volume': v
    }
    return r


def load_fake_volumes_torch(item):
    time_start = time.time()
    v = torch.ones([512, 512, 500], dtype=torch.float32)
    v.fill_(float(item['values']))
    time_end = time.time()
    print('torch_construction_time=', time_end - time_start)
    r = {
        'volume': v
    }
    return r


def load_data(item):
    print('job | ', os.getpid(), ' | loading data |', item['indices'], datetime.datetime.now().time())
    item['time_created'] = time.time()
    time.sleep(2)
    item['time_loaded'] = time.time()
    return item


def run_augmentations(item):
    print('job | ', os.getpid(), ' | augmentation data', item['indices'], datetime.datetime.now().time())
    time.sleep(0.1)
    item['time_augmented'] = time.time()
    items = []
    for i in range(10):
        item_copy = copy.deepcopy(item)
        item_copy['augmentation'] = i
        items.append(item_copy)
    return items


def load_5_items(item):
    items = []
    for i in range(5):
        v = torch.ones([512, 512, 1], dtype=torch.float32)
        v.fill_(float(item['values']))
        items.append({'volume': v})
    return items


def double_values(item):
    item['values'] = item['values'] * 2.0
    return item


class TestSequenceMap(TestCase):
    def test_map_async_20(self):
        # create very large numpy arrays and send it through multiprocessing.Queue: this is expected to be slow!
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_fake_volumes_npy, nb_workers=2)

        time_start = time.time()
        vs = []
        for v in split:
            vs.append(v['volume'].shape)
            print(v['volume'].shape)
        time_end = time.time()
        split.close()
        print('test_map_async_1.TIME=', time_end - time_start)
        self.assertTrue(len(vs) == 20)

    def test_map_sync_multiple_items(self):
        # make sure we have iterate through each item of the returned items
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_5_items, nb_workers=0)

        vs = []
        for v in split:
            vs.append(v['volume'].shape)

        self.assertTrue(len(vs) == 20 * 5)
        split.close()
        print('DONE')

    def test_map_async_multiple_items(self):
        # make sure we have iterate through each item of the returned items
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_5_items, nb_workers=3)

        vs = []
        for v in split:
            vs.append(v['volume'].shape)

        self.assertTrue(len(vs) == 20 * 5)
        split.close()
        print('DONE')

    def test_map_async_20_pytorch(self):
        # large pytorch arrays, this should be much faster to share compare to numpy arrays as
        # only descriptors are effectively sent through the queue
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_fake_volumes_torch, nb_workers=2)

        time_start = time.time()
        vs = []
        for v in split:
            vs.append(v['volume'].shape)
            print(v['volume'].shape)
        time_end = time.time()
        print('test_map_async_1.TIME=', time_end - time_start)

        self.assertTrue(len(vs) == 20)
        split.close()
        print('DONE')

    @staticmethod
    def run_complex_2_map(nb_workers, nb_indices, with_wait):
        # the purpose of this test is to combine 2 maps: one executing slow calls
        # (e.g., IO limited) with another one to do augmentation (e.g., CPU limited)

        print('run_complex_2_map START', datetime.datetime.now().time())

        indices = np.asarray(list(range(nb_indices)))
        split = {
            'indices': indices,
        }

        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).map(load_data, nb_workers=nb_workers).map(run_augmentations, nb_workers=1, max_jobs_at_once=None)
        # process creation is quite slow on windows (>0.7s), so create the processes first
        # so that creation time is not included in processing time
        for batch in split:
            break

        if with_wait:
            # if we wait, it means we won't have jobs to be cancelled (i.e., we will have more accurate timing)
            # else the time of the cancelled jobs will be added to the current time (i.e., `for batch in split` started
            # many jobs)
            time.sleep(3.0)

        print('STARTED', datetime.datetime.now().time())
        time_start = time.time()
        batches = []
        ids = set()
        for batch in split:
            # time_processed = time.time()
            # processed_time = time_processed - batch['time_augmented']
            # loaded_time = time_processed - batch['time_loaded']
            # created_time = time_processed - batch['time_created']
            # batch['time_processed'] = time_processed
            # print(batch)
            # print('TIME----', processed_time, loaded_time, created_time, 'NOW=', datetime.datetime.now().time())
            batches.append(batch)
            ids.add(str(batch['indices'][0]) + '_' + str(batch['augmentation']))

        print('ENDED', datetime.datetime.now().time())

        expected_time = nb_indices * 2.0 / nb_workers + nb_indices * 0.1
        time_end = time.time()
        total_time = time_end - time_start
        print('total_time', total_time, 'Target time=', expected_time, 'nb_jobs=', len(ids), 'nb_jobs_expected=', len(indices) * 10)

        assert len(batches) == len(indices) * 10, 'nb={}'.format(len(batches))
        assert len(ids) == len(indices) * 10

        split.close()

        print('run_complex_2_map END', datetime.datetime.now().time())


    def test_complex_2_map__single_worker(self):
        TestSequenceMap.run_complex_2_map(1, nb_indices=10, with_wait=True)

    def test_complex_2_map__5_worker(self):
        TestSequenceMap.run_complex_2_map(5, nb_indices=10, with_wait=True)

    def test_complex_2_map__5_worker_no_wait(self):
        TestSequenceMap.run_complex_2_map(5, nb_indices=10, with_wait=False)

    def test_complex_2_map__5_worker_40(self):
        TestSequenceMap.run_complex_2_map(5, nb_indices=21, with_wait=True)

    def test_split_closing(self):
        # make sure we can close the processes gracefully
        nb_indices = 40
        indices = np.asarray(list(range(nb_indices)))
        split = {
            'indices': indices,
        }

        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).map(load_data, nb_workers=4).map(run_augmentations, nb_workers=1, max_jobs_at_once=None)
        for batch in split:
            break
        split.job_executer.close()
        print('DONE')
