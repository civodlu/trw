from unittest import TestCase
import trw.train
import trw.transforms
import numpy as np
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

    def test_random_crop_no_padding(self):
        size = [1, 31, 63]
        d = np.zeros([1000] + size, dtype=np.float)
        d[:, size[0] // 2, size[1] // 2, size[2] // 2] = 1.0
        d = torch.from_numpy(d)

        transform = trw.transforms.TransformRandomCrop(padding=None, size=[1, 16, 32])
        batch = transform({'d': d})

        d_transformed = batch['d'].data.numpy()

        assert d_transformed.shape == (1000, 1, 16, 32)

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

    def test_random_crop_pad_joint(self):
        batch = {
            'images': torch.zeros([50, 3, 64, 128], dtype=torch.int64),
            'segmentations': torch.zeros([50, 1, 64, 128], dtype=torch.float32)
        }

        batch['images'][:, :, 32, 64] = 42
        batch['segmentations'][:, :, 32, 64] = 42

        transformer =trw.transforms.TransformRandomCropJoint(feature_names=['images', 'segmentations'], padding=[0, 16, 16], mode='constant')
        transformed_batch = transformer(batch)

        indices_images_42 = np.where(transformed_batch['images'].numpy()[:, 0, :, :] == 42)
        indices_segmentations_42 = np.where(transformed_batch['segmentations'].numpy()[:, 0, :, :] == 42)
        assert (indices_segmentations_42[2] == indices_images_42[2]).all()
        assert (indices_segmentations_42[1] == indices_images_42[1]).all()

    def test_random_flipped_joint(self):
        batch = {
            'images': torch.randint(high=42, size=[50, 3, 64, 128], dtype=torch.int64),
        }
        batch['segmentation'] = batch['images'].float()

        transformer = trw.transforms.TransformRandomFlipJoint(feature_names=['images', 'segmentations'], axis=2)
        transformed_batch = transformer(batch)

        images = batch['images'].float()
        assert (transformed_batch['segmentation'].numpy() == images.numpy()).all()

    def test_random_resize_torch(self):
        batch = {
            'images': torch.randint(high=42, size=[50, 3, 16, 32], dtype=torch.int64),
        }

        transformer = trw.transforms.TransformResize(size=[32, 64])
        transformed_batch = transformer(batch)

        images = transformed_batch['images']
        assert images.shape == (50, 3, 32, 64)
        assert np.average(batch['images'].numpy()) == np.average(transformed_batch['images'].numpy())

    def test_random_resize_numpy(self):
        batch = {
            'images': np.random.randint(low=0, high=42, size=[50, 3, 16, 32], dtype=np.int64),
        }

        transformer = trw.transforms.TransformResize(size=[32, 64], mode='nearest')
        transformed_batch = transformer(batch)

        images = transformed_batch['images']
        assert images.shape == (50, 3, 32, 64)
        assert np.average(batch['images']) == np.average(transformed_batch['images'])

    def test_normalize_numpy(self):
        images = np.random.randint(low=0, high=42, size=[10, 3, 5, 6], dtype=np.int64)
        images[:, 1] *= 2
        images[:, 2] *= 3

        batch = {
            'images': images,
        }

        mean = np.mean(images, axis=(0, 2, 3))
        std = np.std(images, axis=(0, 2, 3))

        transformer = trw.transforms.TransformNormalize(mean=mean, std=std)
        transformed_batch = transformer(batch)
        normalized_images = transformed_batch['images']

        mean_normalized = np.mean(normalized_images, axis=(0, 2, 3))
        std_normalized = np.std(normalized_images, axis=(0, 2, 3))

        assert abs(np.average(mean_normalized)) < 0.1
        assert abs(np.average(std_normalized) - 1) < 0.1

    def test_normalize_torch(self):
        images = torch.randint(high=42, size=[10, 3, 5, 6], dtype=torch.float32)
        images[:, 1] *= 2
        images[:, 2] *= 3

        batch = {
            'images': images,
        }

        mean = np.mean(images.numpy(), axis=(0, 2, 3), dtype=np.float32)
        std = np.std(images.numpy(), axis=(0, 2, 3), dtype=np.float32)

        transformer = trw.transforms.TransformNormalize(mean=mean, std=std)
        transformed_batch = transformer(batch)
        normalized_images = transformed_batch['images']

        mean_normalized = np.mean(normalized_images.numpy(), axis=(0, 2, 3))
        std_normalized = np.std(normalized_images.numpy(), axis=(0, 2, 3))

        assert abs(np.average(mean_normalized)) < 0.1
        assert abs(np.average(std_normalized) - 1) < 0.1

    def test_transform_compose(self):
        batch = {
            'images': torch.randint(high=42, size=[10, 3, 5, 6], dtype=torch.int64)
        }

        transforms = [
            trw.transforms.TransformNormalize(mean=[np.int64(10), np.int64(10), np.int64(10)], std=[np.int64(1), np.int64(1), np.int64(1)]),
            trw.transforms.TransformNormalize(mean=[np.int64(100), np.int64(100), np.int64(100)], std=[np.int64(1), np.int64(1), np.int64(1)]),
        ]

        transformer = trw.transforms.TransformCompose(transforms)
        transformed_batch = transformer(batch)

        max_error = torch.max(torch.abs(batch['images'] - transformed_batch['images'] - 110))
        assert float(max_error) < 1e-5
