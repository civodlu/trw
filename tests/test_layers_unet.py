from unittest import TestCase
import trw
import torch


class TestUNet(TestCase):
    def test_unet_2d(self):
        """
        Simply test instantiation of a UNet-like model
        """
        for nb_blocs in [3, 4]:
            model = trw.layers.UNet(2, 3, 2, linear_upsampling=False, nb_blocs=nb_blocs, base_filters=32)

            inputs = [
                torch.zeros([10, 3, 64, 32]),
                torch.zeros([10, 3, 32, 64]),
                torch.zeros([10, 3, 128, 128])
            ]

            for i in inputs:
                o = model(i)

                assert len(o.shape) == len(i.shape)
                assert o.shape[0] == i.shape[0]
                assert o.shape[1] == 2
                assert o.shape[2:] == i.shape[2:]

    def test_unet_2d_bilinear(self):
        """
        Simply test instantiation of a UNet-like model
        """
        for nb_blocs in [3, 4]:
            model = trw.layers.UNet(2, 3, 2, linear_upsampling=True, nb_blocs=nb_blocs, base_filters=16)

            inputs = [
                torch.zeros([10, 3, 64, 32]),
                torch.zeros([10, 3, 32, 64]),
                torch.zeros([10, 3, 128, 128])
            ]

            for i in inputs:
                o = model(i)

                assert len(o.shape) == len(i.shape)
                assert o.shape[0] == i.shape[0]
                assert o.shape[1] == 2
                assert o.shape[2:] == i.shape[2:]

    def test_unet_3d(self):
        """
        Simply test instantiation of a UNet-like model
        """
        for nb_blocs in [3, 4]:
            model = trw.layers.UNet(3, 3, 2, linear_upsampling=False, nb_blocs=nb_blocs, base_filters=16)

            inputs = [
                torch.zeros([4, 3, 16, 32, 16]),
                torch.zeros([4, 3, 16, 32, 16]),
                torch.zeros([4, 3, 16, 16, 32])
            ]

            for i in inputs:
                o = model(i)

                assert len(o.shape) == len(i.shape)
                assert o.shape[0] == i.shape[0]
                assert o.shape[1] == 2
                assert o.shape[2:] == i.shape[2:]

    def test_unet_3d_bilinear(self):
        """
        Simply test instantiation of a UNet-like model
        """
        for nb_blocs in [3, 4]:
            model = trw.layers.UNet(3, 3, 2, linear_upsampling=True, nb_blocs=nb_blocs, base_filters=16)

            inputs = [
                torch.zeros([4, 3, 16, 32, 16]),
                torch.zeros([4, 3, 16, 32, 16]),
                torch.zeros([4, 3, 16, 16, 32])
            ]

            for i in inputs:
                o = model(i)

                assert len(o.shape) == len(i.shape)
                assert o.shape[0] == i.shape[0]
                assert o.shape[1] == 2
                assert o.shape[2:] == i.shape[2:]