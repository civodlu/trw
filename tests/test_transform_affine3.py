import os
import math
import unittest
import trw
import torch
import numpy as np


class TestTransformsAffine(unittest.TestCase):
    def test_2d_identity_nn(self):
        matrix2 = [
            [1, 0, 0],
            [0, 1, 0],
        ]
        matrix2 = torch.FloatTensor(matrix2)
        images = torch.arange(2 * 5 * 10, dtype=torch.float32).view((2, 1, 5, 10))

        images_tfm2 = trw.transforms.affine_transform(images, matrix2, interpolation='nearest')
        assert int((images == images_tfm2).all()) == 1

    def test_3d_identity_nn(self):
        matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
        matrix = torch.FloatTensor(matrix)
        images = torch.arange(2 * 5 * 10 * 3, dtype=torch.float32).view((2, 1, 5, 10, 3))

        images_tfm = trw.transforms.affine_transform(images, matrix)
        assert torch.max((images - images_tfm).abs()) < 1e-5

    def test_2d_translation_nn(self):
        images = torch.arange(2 * 5 * 10, dtype=torch.float).view((2, 1, 5, 10))
        m = [
                [1, 0, -1],
                [0, 1, -2],
                [0, 0, 1]
            ],
        m = torch.FloatTensor(m)[0]
        m = trw.transforms.to_voxel_space_transform(m, images[0].shape)

        images_tfm = trw.transforms.affine_transform(images, torch.cat((m.unsqueeze(0), m.unsqueeze(0))), interpolation='nearest')
        assert torch.max(torch.abs(images[:, :, 2:, 1:] - images_tfm[:, :, :-2, :-1])) < 1e-5

    def test_2d_image(self):
        matrix = trw.transforms.affine_transformation_translation([80, 0])
        matrix = torch.mm(matrix, trw.transforms.affine_transformation_rotation2d(1 * math.pi / 4))

        from PIL import Image
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'tutorials', 'input_images', '2007_008764.jpg')
        images = Image.open(image_path)
        images = np.asarray(images).transpose((2, 0, 1))
        images = images.reshape([1] + list(images.shape))
        images = torch.from_numpy(images).float()

        images_tfm = trw.transforms.affine_transform(
            images,
            trw.transforms.to_voxel_space_transform(matrix, images[0].shape),
            interpolation='nearest')

        i = np.uint8(images_tfm.numpy())[0, 0]
        options = trw.train.create_default_options()
        root = options['workflow_options']['logging_directory']
        Image.fromarray(np.stack((i, i, i), axis=2)).save(os.path.join(root, 'transformed.png'))

    def test_affine_2d_joint(self):
        options = trw.train.create_default_options()
        root = options['workflow_options']['logging_directory']

        from PIL import Image
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'tutorials', 'input_images', '2007_008764.jpg')
        images = Image.open(image_path)
        images.save(os.path.join(root, 'affine_original.png'))

        images = np.asarray(images).transpose((2, 0, 1))
        images = images.reshape([1] + list(images.shape))
        images = torch.from_numpy(images).float()

        batch = {
            'images': images,
            'images_joint': images
        }

        tfm = trw.transforms.TransformAffine([(-200, 200), (0, 0)], 1, 0)
        transformed_batch = tfm(batch)

        i = np.uint8(transformed_batch['images'].numpy())[0, 0]
        Image.fromarray(np.stack((i, i, i), axis=2)).save(os.path.join(root, 'affine_transformed_.png'))

        assert (transformed_batch['images'] == transformed_batch['images_joint']).all()


if __name__ == '__main__':
    unittest.main()
