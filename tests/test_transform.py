from unittest import TestCase
import trw.train
import trw.transforms
import numpy as np
import time
import torch
import functools


class TestTransform(TestCase):
    def test_batch_pad_constant_numpy(self):
        d = np.asarray([[4], [5], [6]], dtype=int)
        d_transformed = trw.transforms.transform_batch_pad_numpy(d, [2], mode='constant', constant_value=9)
        self.assertTrue(d_transformed.shape == (3, 5))
        assert (d_transformed[0] == [9, 9, 4, 9, 9]).all()
        assert (d_transformed[1] == [9, 9, 5, 9, 9]).all()
        assert (d_transformed[2] == [9, 9, 6, 9, 9]).all()

    def test_batch_pad_constant_torch(self):
        d = np.asarray([[4], [5], [6]], dtype=int)
        d = torch.from_numpy(d)
        d_transformed = trw.transforms.transform_batch_pad_torch(d, [2], mode='constant', constant_value=9)
        d_transformed = d_transformed.data.numpy()
        self.assertTrue(d_transformed.shape == (3, 5))
        assert (d_transformed[0] == [9, 9, 4, 9, 9]).all()
        assert (d_transformed[1] == [9, 9, 5, 9, 9]).all()
        assert (d_transformed[2] == [9, 9, 6, 9, 9]).all()

    def test_batch_pad_symmetric_numpy(self):
        d = np.asarray([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=int)
        d_transformed = trw.transforms.transform_batch_pad_numpy(d, [2], mode='symmetric')
        self.assertTrue(d_transformed.shape == (3, 7))

    def test_batch_pad_edge_torch(self):
        i1 = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        i2 = [[40, 41, 42], [50, 51, 52], [60, 61, 62]]
        d = np.asarray([i1, i2], dtype=float)
        d = d.reshape((2, 1, 3, 3))
        d = torch.from_numpy(d)
        d_transformed = trw.transforms.transform_batch_pad_torch(d, [0, 2, 3], mode='edge')
        d_transformed = d_transformed.data.numpy()
        self.assertTrue(d_transformed.shape == (2, 1, 7, 9))

    def test_batch_pad_replicate_numpy(self):
        i1 = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        i2 = [[40, 41, 42], [50, 51, 52], [60, 61, 62]]
        d = np.asarray([i1, i2], dtype=float)
        d = d.reshape((2, 1, 3, 3))
        d_transformed = trw.transforms.transform_batch_pad_numpy(d, [0, 2, 3], mode='edge')
        self.assertTrue(d_transformed.shape == (2, 1, 7, 9))

    def test_batch_pad_constant_2d_numpy(self):
        i1 = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        i2 = [[40, 41, 42], [50, 51, 52], [60, 61, 62]]

        d = np.asarray([i1, i2], dtype=int)
        d_transformed = trw.transforms.transform_batch_pad_numpy(d, [2, 3], mode='constant')
        self.assertTrue(d_transformed.shape == (2, 7, 9))

    def test_batch_pad_constant_2d_torch(self):
        i1 = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        i2 = [[40, 41, 42], [50, 51, 52], [60, 61, 62]]
        d = np.asarray([i1, i2], dtype=int)
        d = torch.from_numpy(d)

        d_transformed = trw.transforms.transform_batch_pad_torch(d, [2, 3], mode='constant')
        d_transformed = d_transformed.data.numpy()
        self.assertTrue(d_transformed.shape == (2, 7, 9))

    def test_random_crop_numpy(self):
        d = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)
        d_transformed = trw.transforms.transform_batch_random_crop(d, [2])
        self.assertTrue((d_transformed[0] == [1, 2]).all() or (d_transformed[0] == [2, 3]).all())
        self.assertTrue((d_transformed[1] == [4, 5]).all() or (d_transformed[1] == [5, 6]).all())
        self.assertTrue((d_transformed[2] == [7, 8]).all() or (d_transformed[2] == [8, 9]).all())

    def test_random_crop_torch(self):
        d = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)
        d = torch.from_numpy(d)
        d_transformed = trw.transforms.transform_batch_random_crop(d, [2])
        d_transformed = d_transformed.data.numpy()
        self.assertTrue((d_transformed[0] == [1, 2]).all() or (d_transformed[0] == [2, 3]).all())
        self.assertTrue((d_transformed[1] == [4, 5]).all() or (d_transformed[1] == [5, 6]).all())
        self.assertTrue((d_transformed[2] == [7, 8]).all() or (d_transformed[2] == [8, 9]).all())

    def test_random_crop_padd_transform_numpy(self):
        size = [1, 31, 63]
        d = np.zeros([60000] + size, dtype=np.float)
        d[:, size[0] // 2, size[1] // 2, size[2] // 2] = 1

        transform = trw.transforms.TransformRandomCrop(padding=[0, 8, 8])
        batch = transform({'d': d})

        assert batch['d'].shape == (60000, 1, 31, 63)
        d_summed = np.sum(batch['d'], axis=0).squeeze()
        ys, xs = np.where(d_summed > 0)

        # we have set one's at the center of a 2D image, test the maximum and
        # minimum displacement
        self.assertTrue(min(ys) == size[1] // 2 - 8)
        self.assertTrue(max(ys) == size[1] // 2 + 8)

        self.assertTrue(min(xs) == size[2] // 2 - 8)
        self.assertTrue(max(xs) == size[2] // 2 + 8)

    def test_random_crop_padd_transform_torch(self):
        size = [1, 31, 63]
        d = np.zeros([60000] + size, dtype=np.float)
        d[:, size[0] // 2, size[1] // 2, size[2] // 2] = 1.0
        d = torch.from_numpy(d)

        transform = trw.transforms.TransformRandomCrop(padding=[0, 8, 8])
        batch = transform({'d': d})

        d_transformed = batch['d'].data.numpy()

        assert d_transformed.shape == (60000, 1, 31, 63)
        d_summed = np.sum(d_transformed, axis=0).squeeze()
        ys, xs = np.where(d_summed > 0)

        # we have set one's at the center of a 2D image, test the maximum and
        # minimum displacement
        self.assertTrue(min(ys) == size[1] // 2 - 8)
        self.assertTrue(max(ys) == size[1] // 2 + 8)

        self.assertTrue(min(xs) == size[2] // 2 - 8)
        self.assertTrue(max(xs) == size[2] // 2 + 8)


    def test_augmented_split_performance(self):
        # mimic MNSIT shapes
        N = 60000
        data = np.random.rand(N, 1, 28, 28).astype(np.float)
        c = np.zeros([N], dtype=int)
        split_np = {'images': data, 'classes':c}

        split_no_transform = trw.train.SequenceArray(split_np)
        transform = trw.transforms.TransformRandomCrop(padding=[0, 8, 8])
        split_with_transform = trw.train.SequenceArray(split_np, transforms=transform, sampler=trw.train.SamplerRandom(batch_size=500))

        time_raw_augmentation_start = time.time()
        _ = transform(split_np)
        time_raw_augmentation_end = time.time()
        time_raw_augmentation = time_raw_augmentation_end - time_raw_augmentation_start
        print('TIME transform alone=', time_raw_augmentation)

        time_no_augmentation_start = time.time()
        for _ in split_no_transform:
            pass
        time_no_augmentation_end = time.time()
        time_no_augmentation = time_no_augmentation_end - time_no_augmentation_start
        print('TIME no augmentation=', time_no_augmentation)

        time_with_augmentation_start = time.time()
        for _ in split_with_transform:
            pass
        time_with_augmentation_end = time.time()
        time_with_augmentation = time_with_augmentation_end - time_with_augmentation_start
        print('TIME with augmentation=', time_with_augmentation)

        split_with_transform_last_main_thread = trw.train.SequenceArray(split_np, sampler=trw.train.SamplerRandom(batch_size=500)).map(transform)
        time_split_with_transform_last_main_start = time.time()
        for _ in split_with_transform_last_main_thread:
            pass
        time_split_with_transform_last_main_end = time.time()
        time_split_with_transform_last_main = time_split_with_transform_last_main_end - time_split_with_transform_last_main_start
        print('TIME with augmentation last main thread=', time_split_with_transform_last_main)

        split_with_transform_last_workers_thread = trw.train.SequenceArray(split_np, sampler=trw.train.SamplerRandom(batch_size=25)).map(transform, nb_workers=2, max_jobs_at_once=5).batch(20)
        for _ in split_with_transform_last_workers_thread:
            break  # we MUST exclude the process creation time

        time_split_with_transform_last_workers_start = time.time()
        for _ in split_with_transform_last_workers_thread:
            pass
        time_split_with_transform_last_workers_end = time.time()
        time_split_with_transform_last_workers = time_split_with_transform_last_workers_end - time_split_with_transform_last_workers_start
        print('TIME with augmentation last main thread=', time_split_with_transform_last_workers)

    def test_transform_base_criteria(self):
        # filter by name
        batch = {
            'test_1': 0,
            'test_2': 42,
        }

        criteria_fn = functools.partial(trw.transforms.criteria_feature_name, feature_names=['test_2'])
        transform_fn = lambda _1, _2: 43

        transformer = trw.transforms.TransformBatchWithCriteria(criteria_fn=criteria_fn, transform_fn=transform_fn)
        transformed_batch = transformer(batch)
        assert transformed_batch['test_1'] == 0
        assert transformed_batch['test_2'] == 43

    def test_transform_random_flip_numpy(self):
        batch = {
            'images': np.asarray([
                [1, 2, 3],
                [4, 5, 6]
            ])
        }

        criteria_fn = functools.partial(trw.transforms.criteria_feature_name, feature_names=['images'])
        transformer = trw.transforms.TransformRandomFlip(criteria_fn=criteria_fn, axis=1, flip_probability=1.0)
        transformed_batch = transformer(batch)

        image_fliped = transformed_batch['images']
        assert len(image_fliped) == 2
        assert image_fliped.shape == (2, 3)

        assert image_fliped[0, 0] == 3
        assert image_fliped[0, 1] == 2
        assert image_fliped[0, 2] == 1

        assert image_fliped[1, 0] == 6
        assert image_fliped[1, 1] == 5
        assert image_fliped[1, 2] == 4

        # make sure the original images are NOT flipped!
        image = batch['images']
        assert image[0, 0] == 1
        assert image[0, 1] == 2
        assert image[0, 2] == 3

    def test_transform_random_flip_torch(self):
        batch = {
            'images': torch.from_numpy(np.asarray([
                [1, 2, 3],
                [4, 5, 6]
            ]))
        }

        criteria_fn = functools.partial(trw.transforms.criteria_feature_name, feature_names=['images'])
        transformer = trw.transforms.TransformRandomFlip(criteria_fn=criteria_fn, axis=1, flip_probability=1.0)
        transformed_batch = transformer(batch)

        image_fliped = transformed_batch['images']
        assert len(image_fliped) == 2
        assert image_fliped.shape == (2, 3)

        assert image_fliped[0, 0] == 3
        assert image_fliped[0, 1] == 2
        assert image_fliped[0, 2] == 1

        assert image_fliped[1, 0] == 6
        assert image_fliped[1, 1] == 5
        assert image_fliped[1, 2] == 4

        # make sure the original images are NOT flipped!
        image = batch['images']
        assert image[0, 0] == 1
        assert image[0, 1] == 2
        assert image[0, 2] == 3

    def test_cutout_numpy(self):
        batch = {
            'images': np.ones([50, 3, 64, 128], dtype=np.uint8) * 255
        }
        transformer = trw.transforms.TransformRandomCutout(cutout_size=(3, 16, 32))
        transformed_batch = transformer(batch)
        assert np.min(batch['images']) == 255, 'original image was modified!'

        assert np.min(transformed_batch['images']) == 0, 'transformed image was NOT modified!'
        for i in transformed_batch['images']:
            nb_0 = np.where(i == 0)
            assert len(nb_0[0]) == 3 * 16 * 32

    def test_cutout_torch(self):
        batch = {
            'images': torch.ones([50, 3, 64, 128], dtype=torch.uint8) * 255
        }
        transformer = trw.transforms.TransformRandomCutout(cutout_size=(3, 16, 32))
        transformed_batch = transformer(batch)
        assert torch.min(batch['images']) == 255, 'original image was modified!'

        assert torch.min(transformed_batch['images']) == 0, 'transformed image was NOT modified!'
        for i in transformed_batch['images']:
            nb_0 = np.where(i.numpy() == 0)
            assert len(nb_0[0]) == 3 * 16 * 32
