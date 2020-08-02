import functools

import trw
import torch
import torch.nn as nn
from unittest import TestCase

from trw.layers import AutoencoderConvolutionalVariational, AutoencoderConvolutionalVariationalConditional
from trw.train.losses import one_hot


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_size, nb_digits=10):
        super(ConditionalGenerator, self).__init__()

        self.nb_digits = nb_digits
        self.convs_t = trw.layers.ConvsTransposeBase(
            2,
            input_channels=latent_size + nb_digits,
            channels=[1024, 512, 256, 1],
            convolution_kernels=4,
            strides=[1, 2, 2, 2],
            batch_norm_kwargs={},
            paddings=[0, 1, 1, 1],
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            squash_function=torch.tanh,
            target_shape=[28, 28]
        )

    def forward(self, latent, digits):
        assert len(digits.shape) == 1

        digits_one_hot = one_hot(digits, self.nb_digits).unsqueeze(2).unsqueeze(3)
        full_latent = torch.cat((digits_one_hot, latent), dim=1)
        x = self.convs_t(full_latent)
        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(self, nb_digits=10):
        super(ConditionalDiscriminator, self).__init__()

        self.nb_digits = nb_digits
        self.convs = trw.layers.convs_2d(
            1 + nb_digits,
            [64, 128, 256, 2],
            convolution_kernels=[4, 4, 4, 3],
            strides=[2, 4, 4, 2],
            batch_norm_kwargs={},
            pooling_size=None,
            with_flatten=True,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True
        )

    def forward(self, input, digits):
        input_class = torch.ones(
            [digits.shape[0], self.nb_digits, input.shape[2], input.shape[3]],
            device=input.device) * one_hot(digits, 10).unsqueeze(2).unsqueeze(3)
        x = self.convs(torch.cat((input, input_class), dim=1))
        return x


class TestLayers(TestCase):
    def test_convs_layers(self):
        """
        Make sure the sequence is done as expected
        """
        convs = trw.layers.ConvsBase(
            input_channels=1,
            cnn_dim=2,
            channels=[2, 4, 8],
            convolution_repeats=[2, 1, 1],
            batch_norm_kwargs={},
            lrn_kwargs={},
            dropout_probability=0.1,
            activation=nn.LeakyReLU,
            last_layer_is_output=True
        )

        # we have 3 groups (defined by number of ``channels``)
        assert len(convs.layers) == 3

        group_0 = list(convs.layers[0])
        assert len(group_0) == 10
        assert isinstance(group_0[0], nn.Conv2d)
        assert isinstance(group_0[1], nn.LeakyReLU)
        assert isinstance(group_0[2], nn.BatchNorm2d)
        assert isinstance(group_0[3], nn.LocalResponseNorm)

        assert isinstance(group_0[4], nn.Conv2d)
        assert isinstance(group_0[5], nn.LeakyReLU)

        # Dropout after maxpooling, see ``Towards Principled Design of Deep
        # Convolutional Networks: Introducing SimpNet``
        assert isinstance(group_0[6], nn.MaxPool2d)
        assert isinstance(group_0[7], nn.BatchNorm2d)
        assert isinstance(group_0[8], nn.LocalResponseNorm)
        assert isinstance(group_0[9], nn.Dropout2d)

        group_1 = list(convs.layers[1])
        assert len(group_1) == 6
        assert isinstance(group_1[0], nn.Conv2d)
        assert isinstance(group_1[1], nn.LeakyReLU)
        assert isinstance(group_1[2], nn.MaxPool2d)
        assert isinstance(group_1[3], nn.BatchNorm2d)
        assert isinstance(group_1[4], nn.LocalResponseNorm)
        assert isinstance(group_1[5], nn.Dropout2d)

        # the last group will be used as classification ``last_layer_is_output`` == True
        # So do not add activation, regularization or maxpool
        group_2 = list(convs.layers[2])
        assert isinstance(group_2[0], nn.Conv2d)

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

    def test_gan_dc(self):
        latent_size = 32
        activation = functools.partial(torch.nn.LeakyReLU, negative_slope=0.2)

        discriminator = trw.layers.ConvsBase(
            2,
            input_channels=1,
            channels=[32, 64, 128, 2],
            batch_norm_kwargs={},
            with_flatten=True,
            activation=activation,
            last_layer_is_output=True
        )

        generator = trw.layers.ConvsTransposeBase(
            2,
            input_channels=latent_size,
            channels=[64, 32, 16, 8, 1],
            strides=[2, 2, 2, 1, 1],
            batch_norm_kwargs={},
            paddings=0,
            target_shape=[28, 28],
            activation=activation,
            squash_function=torch.tanh,
            last_layer_is_output=True
        )

        optimizer_fn = functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999))

        model = trw.layers.GanConditional(
            discriminator=discriminator,
            generator=generator,
            latent_size=latent_size,
            optimizer_discriminator_fn=optimizer_fn,
            optimizer_generator_fn=optimizer_fn,
            real_image_from_batch_fn=lambda batch: 2 * batch['images'] - 1
        )

        batch = {
            'images': torch.zeros([10, 1, 28, 28]),
            'split_name': 'train'
        }
        o = model(batch)
        assert 'fake' in o
        assert 'real' in o
        assert 'classifier_real' in o
        assert 'classifier_fake' in o

    def test_gan_conditional(self):
        optimizer_fn = functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999))

        latent_size = 32
        discriminator = ConditionalDiscriminator()
        generator = ConditionalGenerator(latent_size)

        model = trw.layers.GanConditional(
            discriminator=discriminator,
            generator=generator,
            latent_size=latent_size,
            optimizer_discriminator_fn=optimizer_fn,
            optimizer_generator_fn=optimizer_fn,
            real_image_from_batch_fn=lambda batch: batch['images'],
            observed_discriminator_fn=lambda batch: {'digits': batch['targets']},
            observed_generator_fn=lambda batch: {'digits': batch['targets']},
        )

        batch = {
            'images': torch.zeros([10, 1, 28, 28]),
            'targets': torch.arange(10, dtype=torch.long),
            'split_name': 'train'
        }

        o = model(batch)
        assert 'fake' in o
        assert 'real' in o
        assert 'classifier_real' in o
        assert 'classifier_fake' in o
