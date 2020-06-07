import trw
import torch
import torch.nn as nn
from unittest import TestCase


class TestLayersUnetBase(TestCase):
    def test_unet_no_latent(self):
        unet = trw.layers.UNetBase(2, 1, [8, 16, 32], 5)
        o = unet(torch.zeros([3, 1, 64, 64]))
        assert o.shape == (3, 5, 64, 64)

    def test_unet_with_latent(self):
        unet = trw.layers.UNetBase(2, 1, [8, 16, 32], 5, latent_channels=10)
        o = unet(torch.zeros([3, 1, 64, 64]), latent=torch.ones(3, 10, 1, 1))
        assert o.shape == (3, 5, 64, 64)
