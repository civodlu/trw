import collections

import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools

from trw.layers.gan import Gan
from trw.train import OutputEmbedding, OutputLoss
from trw.train.losses import one_hot, LossMsePacked
from trw.train.outputs_trw import OutputClassification2


class Generator(nn.Module):
    def __init__(self, latent_size, nb_digits=10):
        super().__init__()

        self.nb_digits = nb_digits
        self.convs_t = trw.layers.ConvsTransposeBase(
            2,
            input_channels=latent_size + nb_digits,
            channels=[1024, 512, 256, 1],
            convolution_kernels=4,
            strides=[1, 2, 2, 2],
            paddings=[0, 1, 1, 1],
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            squash_function=torch.tanh,
            target_shape=[28, 28]
        )

    def forward(self, batch, latent):
        digits = batch['targets']
        real = batch['images']
        assert len(digits.shape) == 1

        # introduce the target as one hot encoding input to the generator
        digits_one_hot = one_hot(digits, self.nb_digits).unsqueeze(2).unsqueeze(3)
        latent = latent.unsqueeze(2).unsqueeze(3)

        full_latent = torch.cat((digits_one_hot, latent), dim=1)
        o = self.convs_t(full_latent)
        return o, collections.OrderedDict([
            ('image', OutputEmbedding(o)),
            #('l1', OutputLoss(0.5 * torch.nn.L1Loss()(o, real)))  # on average, the generated & real should match
        ])


class Discriminator(nn.Module):
    def __init__(self, nb_digits=10):
        super().__init__()

        self.nb_digits = nb_digits
        self.convs = trw.layers.convs_2d(
            1 + nb_digits,
            [64, 128, 256, 2],
            convolution_kernels=[4, 4, 4, 3],
            strides=[2, 4, 4, 2],
            pooling_size=None,
            with_flatten=True,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True
        )

    def forward(self, batch, image, is_real):
        digits = batch['targets']

        # introduce the target as one hot encoding input to the discriminator
        input_class = torch.ones(
            [image.shape[0], self.nb_digits, image.shape[2], image.shape[3]],
            device=image.device) * one_hot(digits, 10).unsqueeze(2).unsqueeze(3)
        o = self.convs(torch.cat((image, input_class), dim=1))
        o_expected = int(is_real) * torch.ones(len(image), device=image.device, dtype=torch.long)

        return {
            'classification': OutputClassification2(
                o, o_expected,
                criterion_fn=LossMsePacked,  # LSGan loss function
            )
        }


def get_image(batch):
    return 2 * batch['images'] - 1


def get_target(batch):
    return {'digits': batch['targets']}


def create_model(options):
    latent_size = 64

    discriminator = Discriminator()
    generator = Generator(latent_size)

    optimizer_fn = functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999))

    model = Gan(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=optimizer_fn,
        optimizer_generator_fn=optimizer_fn,
        real_image_from_batch_fn=get_image,
    )

    return model


def per_epoch_callbacks():
    return [
        trw.train.CallbackReportingExportSamples(
            max_samples=200,
            reporting_scatter_x='targets',
            reporting_scatter_y='split_name',
            reporting_display_with='term_gen_image_output'),
        trw.train.CallbackEpochSummary(),
        trw.train.CallbackReportingRecordHistory(),
    ]


def pre_training_callbacks():
    return [
        trw.train.CallbackReportingStartServer(),
        trw.train.CallbackReportingModelSummary(),
    ]


options = trw.train.create_default_options(num_epochs=15)
trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_callbacks,
    callbacks_pre_training_fn=pre_training_callbacks
)

model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=1,
    inputs_fn=lambda: trw.datasets.create_mnist_dataset(batch_size=32, normalize_0_1=True),
    eval_every_X_epoch=1,
    model_fn=create_model,
    run_prefix='mnist_gan_conditional',
    optimizers_fn=None  # the module has its own optimizers
)
