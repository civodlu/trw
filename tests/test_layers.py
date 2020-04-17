import trw
import torch
from unittest import TestCase

from trw.layers import AutoencoderConvolutionalVariational, AutoencoderConvolutionalVariationalConditional
from trw.train.losses import one_hot


class TestLayers(TestCase):
    def test_convs_base(self):
        convs = trw.layers.ConvsBase(input_channels=1, cnn_dim=2, channels=[2, 4, 8])

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        o = convs.forward_with_intermediate(i)
        assert len(o) == 3

    def test_convs_fcnn(self):
        convs = trw.layers.ConvsBase(input_channels=1, cnn_dim=2, channels=[16, 32, 64])
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

    def test_convs_fcnn_concatenate_mode(self):
        convs = trw.layers.ConvsBase(input_channels=1, cnn_dim=2, channels=[16, 32, 64])
        fcnn = trw.layers.FullyConvolutional(
            cnn_dim=2,
            base_model=convs,
            deconv_filters=[64, 32, 16, 8],
            conv_filters=[16, 32, 64],
            convolution_kernels=3,
            strides=2,
            nb_classes=2,
            concat_mode='concatenate'
        )

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        o = fcnn(i)
        assert o.shape == (5, 2, 32, 32)

    def test_convs_fcnn_different_kernels(self):
        convs = trw.layers.ConvsBase(cnn_dim=2, input_channels=1, channels=[16, 32, 64])
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
        convs = trw.layers.ConvsBase(cnn_dim=2, input_channels=1, channels=[16, 32, 64])
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

    def test_autoencoder_conv_padding(self):
        model = trw.layers.AutoencoderConvolutional(2, 1, [4, 8, 16], [8, 4, 1], last_layer_is_output=True)

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        intermediates = model.forward_with_intermediate(i)
        assert len(intermediates) == 2
        encoding, reconstruction = intermediates

        assert encoding.shape == (5, 16, 4, 4)
        assert reconstruction.shape == i.shape

        last_stage = model.decoder.layers[-1]
        assert len(last_stage) == 1

    def test_autoencoder_conv_cropping(self):
        model = trw.layers.AutoencoderConvolutional(2, 1, [4, 8, 16], [16, 8, 4, 1], last_layer_is_output=True)

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        intermediates = model.forward_with_intermediate(i)
        assert len(intermediates) == 2
        encoding, reconstruction = intermediates

        assert encoding.shape == (5, 16, 4, 4)
        assert reconstruction.shape == i.shape

        last_stage = model.decoder.layers[-1]
        assert len(last_stage) == 1

    def test_sub_tensor(self):
        i = torch.randn([5, 1, 32, 32], dtype=torch.float32)

        layer = trw.layers.SubTensor([0, 10, 15], [1, 14, 22])
        o = layer(i)
        assert o.shape == (5, 1, 4, 7)

    def test_crop_or_pad__pad_only(self):
        i = torch.randn([5, 1, 32, 32], dtype=torch.float32)
        i_shaped = trw.layers.crop_or_pad_fun(i, (38, 40))
        assert i_shaped.shape == (5, 1, 38, 40)
        assert (i_shaped[:, :, 3:35, 4:36] == i).all()

    def test_crop_or_pad__crop_only(self):
        i = torch.randn([5, 1, 32, 32], dtype=torch.float32)
        i_shaped = trw.layers.crop_or_pad_fun(i, (16, 20))
        assert i_shaped.shape == (5, 1, 16, 20)
        assert (i[:, :, 8:24, 6:26] == i_shaped).all()

    def test_autoencoder_conv_var(self):
        z_size = 20

        encoder = trw.layers.ConvsBase(
            cnn_dim=2,
            input_channels=1,
            channels=[8, 16, 32],
            convolution_kernels=3,
            batch_norm_kwargs=None,
        )

        decoder = trw.layers.ConvsTransposeBase(
            cnn_dim=2,
            input_channels=z_size,
            channels=[32, 16, 8, 1],
            strides=[2, 2, 2, 2],
            convolution_kernels=3,
            last_layer_is_output=True,
            squash_function=torch.sigmoid,
            paddings=0
        )

        x = torch.randn([10, 1, 28, 28], dtype=torch.float32)
        model = AutoencoderConvolutionalVariational([1, 1, 28, 28], encoder, decoder, z_size)
        recon, mu, logvar = model(x)

        assert recon.shape == (10, 1, 28, 28)
        assert mu.shape == (10, 20)
        assert mu.shape == logvar.shape

        loss_bce = AutoencoderConvolutionalVariational.loss_function(recon, x, mu, logvar, recon_loss_name='BCE')
        assert loss_bce.shape == (10,)
        loss_mse = AutoencoderConvolutionalVariational.loss_function(recon, x, mu, logvar, recon_loss_name='MSE')
        assert loss_mse.shape == (10,)

    def test_autoencoder_conv_var_conditional(self):
        z_size = 20
        y_size = 5

        encoder = trw.layers.ConvsBase(
            cnn_dim=2,
            input_channels=1,
            channels=[8, 16, 32],
            convolution_kernels=3,
            batch_norm_kwargs=None,
        )

        decoder = trw.layers.ConvsTransposeBase(
            cnn_dim=2,
            input_channels=z_size + y_size,
            channels=[32, 16, 8, 1],
            strides=[2, 2, 2, 2],
            convolution_kernels=3,
            last_layer_is_output=True,
            squash_function=torch.sigmoid,
            paddings=0
        )

        y = one_hot(torch.tensor([0] * 10, dtype=torch.long), y_size)
        x = torch.randn([10, 1, 28, 28], dtype=torch.float32)
        model = AutoencoderConvolutionalVariationalConditional([1, 1, 28, 28], encoder, decoder, z_size, y_size=y_size)
        recon, mu, logvar = model(x, y)

        assert recon.shape == (10, 1, 28, 28)
        assert mu.shape == (10, z_size)
        assert mu.shape == logvar.shape
