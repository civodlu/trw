import trw
import torch
from unittest import TestCase


class TestLayers(TestCase):
    def test_convs_base(self):
        convs = trw.layers.ConvsBase(cnn_dim=2, channels=[1, 2, 4, 8])

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        o = convs.forward_with_intermediate(i)
        assert len(o) == 3

    def test_convs_fcnn(self):
        convs = trw.layers.ConvsBase(cnn_dim=2, channels=[1, 16, 32, 64])
        fcnn = trw.layers.FullyConvolutional(
            cnn_dim=2,
            base_model=convs,
            deconv_filters=[64, 32, 16, 8],
            convolution_kernels=3,
            strides=2,
            nb_classes=2
        )

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        o = fcnn(i)
        assert o.shape == (5, 2, 32, 32)

    def test_convs_fcnn_different_kernels(self):
        convs = trw.layers.ConvsBase(cnn_dim=2, channels=[1, 16, 32, 64])
        fcnn = trw.layers.FullyConvolutional(
            cnn_dim=2,
            base_model=convs,
            deconv_filters=[64, 32, 16, 8],
            convolution_kernels=[3, 5, 3],
            strides=[2] * 3,
            nb_classes=2
        )

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        o = fcnn(i)
        assert o.shape == (5, 2, 32, 32)

    def test_convs_fcnn_kernel7(self):
        convs = trw.layers.ConvsBase(cnn_dim=2, channels=[1, 16, 32, 64])
        fcnn = trw.layers.FullyConvolutional(
            cnn_dim=2,
            base_model=convs,
            deconv_filters=[64, 32, 16, 8],
            convolution_kernels=7,
            strides=[2] * 3,
            nb_classes=2
        )

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        o = fcnn(i)
        assert o.shape == (5, 2, 32, 32)
