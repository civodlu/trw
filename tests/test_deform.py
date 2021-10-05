import os


import unittest
from PIL import Image
import numpy as np
import torch
import trw.transforms

from utils import root_output


here = os.path.abspath(os.path.dirname(__file__))


class TestDeform(unittest.TestCase):
    def test_image_2d(self):
        image = Image.open(os.path.join(here, 'images/checkerboard.png'))
        image = np.array(image) * 255

        images_deformed_torch = []

        for i in range(100):
            image_torch = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            image_deformed_torch = trw.transforms.deform_image_random(
                [image_torch, image_torch],
                # we have 6 cells (2 margin, 4 free): 1 + 4 + 1
                control_points=(3, 4),
                max_displacement=None,
                interpolation='linear')

            # all images must be transformed the same way
            assert len(image_deformed_torch) == 2
            assert (image_deformed_torch[0] - image_deformed_torch[1]).abs().max() <= 1e-5

            images_deformed_torch.append(image_deformed_torch[0])

            image_deformed = Image.fromarray(image_deformed_torch[0].squeeze().numpy().astype(np.uint8))
            image_deformed.save(os.path.join(root_output, f'deformed_{i}.png'))

        # we should have half voxels 255 and half voxels 0, even after applying
        # the random transformation
        images_deformed_torch = torch.cat(images_deformed_torch)
        mean_value = images_deformed_torch.mean()
        expected_mean = 255.0 / 2
        print(f'test_image_2d mean={mean_value}, expected={expected_mean}')
        assert abs(mean_value - expected_mean) < 0.5, 'difference is too big! Images need to be inspected visually!'


if __name__ == '__main__':
    unittest.main()
