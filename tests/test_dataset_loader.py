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


class TestDatasetLoader(TestCase):
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

    def test_map_async_multiple_items(self):
        # make sure we have iterate through each item of the returned items
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_5_items, nb_workers=3)
    
        vs = []
        for v in split:
            vs.append(v['volume'].shape)
    
        self.assertTrue(len(vs) == 20 * 5)

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

    def test_split_numpy_sequential(self):
        split = {'values': np.arange(0, 1000)}

        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential())
        index = 0
        for index, batch in enumerate(split_np):
            assert len(batch) == 2
            assert batch['values'][0] == index
        assert index == 999

    def test_split_numpy_random(self):
        split = {'values': np.arange(0, 1000)}
        values = set()

        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerRandom())
        for batch in split_np:
            assert len(batch) == 2
            values.add(batch['values'][0])

        assert len(values) == 1000

    def test_split_numpy_random_batch(self):
        split = {'values': np.arange(0, 1000)}
        values = set()

        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerRandom(batch_size=10))
        for batch in split_np:
            assert len(batch) == 2
            assert len(batch['values']) == 10
            for v in batch['values']:
                values.add(v)

        assert len(values) == 1000

    def test_split_numpy_random_with_replacement(self):
        nb_classes = 10
        nb_samples = 10000
        split = {'values': np.arange(0, nb_classes)}

        values = []
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerRandom(replacement=True, nb_samples_to_generate=nb_samples))
        for batch in split_np:
            assert len(batch) == 2
            values.append(batch['values'][0])

        assert len(values) == 10000
        values_counts = collections.Counter(values)
        for value, count in values_counts.items():
            assert value >= 0 and value <= 9
            assert abs(count - nb_samples / nb_classes) < 0.1 * nb_samples / nb_classes

    def test_split_numpy_random_with_replacement_batch(self):
        nb_classes = 10
        nb_samples = 10000
        split = {'values': np.arange(0, nb_classes)}

        values = []
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerRandom(replacement=True, nb_samples_to_generate=nb_samples, batch_size=10))
        for batch in split_np:
            assert len(batch) == 2
            assert len(batch['values']) == 10
            for v in batch['values']:
                values.append(v)

        assert len(values) == 10000
        values_counts = collections.Counter(values)
        for value, count in values_counts.items():
            assert value >= 0 and value <= 9
            assert abs(count - nb_samples / nb_classes) < 0.1 * nb_samples / nb_classes

    def test_split_numpy_random_subset(self):
        nb_samples = 10
        split = {'values': np.arange(0, nb_samples)}

        values = set()
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerSubsetRandom(indices=range(5, 10)))
        for batch in split_np:
            assert len(batch) == 2
            value = batch['values'][0]
            values.add(value)
            assert value >= 5 and value < 10
        assert len(values) == 5

    def test_split_class_resample(self):
        nb_samples = 9999  # must be divisible by the number of classes, else it will be rounded

        split = {
            'classes': np.asarray([0, 0, 0, 1, 1, 2]),
            'indices': np.asarray([0, 1, 2, 3, 4, 5]),
        }

        classes = []
        indices = set()
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerClassResampling(class_name='classes', nb_samples_to_generate=nb_samples, batch_size=10))
        for batch in split_np:
            assert len(batch) == 3
            assert 'sample_uid' in batch
            for v in batch['classes']:
                classes.append(v)
            for v in batch['indices']:
                indices.add(v)

        assert len(classes) == nb_samples
        assert len(indices) == 6

        values_counts = collections.Counter(classes)
        for value, count in values_counts.items():
            assert value >= 0 and value <= 2
            assert count == nb_samples // 3

    def test_split_batcher(self):
        split = {
            'classes': np.asarray([0, 1, 2, 3, 4, 5]),
            'indices': np.asarray([0, 1, 2, 3, 4, 5]),
        }
        
        indices = []
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).batch(2)
        for batch in split_np:
            batch = trw.train.default_collate_fn(batch, device=None)
            assert isinstance(batch, collections.Mapping)
            assert len(batch['classes']) == 2
            assert len(batch['indices']) == 2
            indices.append(batch['indices'])
            
        assert len(indices) == 3
        assert torch.sum(torch.stack(indices)) == 1 + 2 + 3 + 4 + 5

    def test_split_batcher_batch_not_full(self):
        split = {
            'classes': np.asarray([0, 1, 2, 3, 4]),
            'indices': np.asarray([0, 1, 2, 3, 4]),
        }
    
        indices = []
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).batch(2)
        for batch in split_np:
            batch = trw.train.default_collate_fn(batch, device=None)
            indices.append(batch['indices'])
    
        assert len(indices) == 3
        assert len(indices[-1]) == 1

    def test_split_batcher_batch_not_full_discarded(self):
        split = {
            'classes': np.asarray([0, 1, 2, 3, 4]),
            'indices': np.asarray([0, 1, 2, 3, 4]),
        }
    
        indices = []
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).batch(2, discard_batch_not_full=True)
        for batch in split_np:
            batch = trw.train.default_collate_fn(batch, device=None)
            indices.append(batch['indices'])
    
        assert len(indices) == 2

    @staticmethod
    def run_complex_2_map(nb_workers, nb_indices, with_wait):
        # the purpose of this test is to combine 2 maps: one executing slow calls
        # (e.g., IO limited) with another one to do augmentation (e.g., CPU limited)

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
            #time_processed = time.time()
            #processed_time = time_processed - batch['time_augmented']
            #loaded_time = time_processed - batch['time_loaded']
            #created_time = time_processed - batch['time_created']
            #batch['time_processed'] = time_processed
            #print(batch)
            #print('TIME----', processed_time, loaded_time, created_time, 'NOW=', datetime.datetime.now().time())
            batches.append(batch)
            ids.add(str(batch['indices'][0]) + '_' + str(batch['augmentation']))

        print('ENDED', datetime.datetime.now().time())

        expected_time = nb_indices * 2.0 / nb_workers  + nb_indices * 0.1
        time_end = time.time()
        total_time = time_end - time_start
        print('total_time', total_time, 'Target time=', expected_time, 'nb_jobs=', len(ids), 'nb_jobs_expected=', len(indices) * 10)

        assert len(batches) == len(indices) * 10, 'nb={}'.format(len(batches))
        assert len(ids) == len(indices) * 10

        if with_wait:
            assert total_time < expected_time + 0.2

    def test_complex_2_map__single_worker(self):
        TestDatasetLoader.run_complex_2_map(1, nb_indices=10, with_wait=True)

    def test_complex_2_map__5_worker(self):
        TestDatasetLoader.run_complex_2_map(5, nb_indices=10, with_wait=True)

    def test_complex_2_map__5_worker_no_wait(self):
        TestDatasetLoader.run_complex_2_map(5, nb_indices=10, with_wait=False)

    def test_complex_2_map__5_worker_40(self):
        TestDatasetLoader.run_complex_2_map(5, nb_indices=21, with_wait=True)

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

    def test_split_subsample(self):
        nb_indices = 40
        split = {
            'uid': np.asarray(list(range(nb_indices))),
            'values': np.asarray(list(range(nb_indices))),
        }

        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).map(double_values, nb_workers=1).batch(5).subsample(10)

        batches = []
        for i in split:
            i = trw.train.default_collate_fn(i, device=None)
            for n in range(trw.train.len_batch(i)):
                self.assertTrue(i['uid'].data.numpy()[n] * 2.0 == i['values'][n])
                self.assertTrue(np.max(i['uid'].data.numpy()) < 40)
            batches.append(i)
            print(i)

        # we resampled the sequence to 10 samples with a batch size of 5
        # so we expect 2 batches
        self.assertTrue(len(batches) == 2)
