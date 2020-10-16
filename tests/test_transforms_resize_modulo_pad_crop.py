import unittest
import trw
import torch
import numpy as np


class TestTransformsResizeModuloPadCrop(unittest.TestCase):
    def test_crop_mode_torch(self):
        batch = {
            'images': torch.rand([2, 3, 64, 64], dtype=torch.float32)
        }

        tfm = trw.transforms.TransformResizeModuloCropPad(60)
        transformed = tfm(batch)
        assert transformed['images'].shape == (2, 3, 60, 60)
