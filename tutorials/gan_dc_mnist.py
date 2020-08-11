import collections

import trw.train
import trw.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from trw.layers.gan_conditional import Gan
from trw.train import LossMsePacked, OutputEmbedding
from trw.train.outputs_trw import OutputClassification2


def per_epoch_callbacks():
    return [
        trw.train.CallbackReportingExportSamples(
            split_exclusions=['test']
        ),
        trw.train.CallbackEpochSummary(),
        trw.train.CallbackReportingRecordHistory(),
    ]


def get_image(batch):
    return 2 * batch['images'] - 1


def global_max_pooling_2d(tensor):
    assert len(tensor.shape) == 4, 'must be a NCHW tensor!'
    return F.max_pool2d(tensor, tensor.shape[2:]).squeeze(2).squeeze(2)


def global_average_pooling_2d(tensor):
    assert len(tensor.shape) == 4, 'must be a NCHW tensor!'
    return F.avg_pool2d(tensor, tensor.shape[2:]).squeeze(2).squeeze(2)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = trw.layers.convs_2d(
            input_channels=1,
            channels=[64, 128, 256, 2],
            convolution_kernels=[4, 4, 4, 3],
            strides=[2, 2, 2, 1],
            batch_norm_kwargs={},
            pooling_size=None,
            with_flatten=False,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True
        )

    def forward(self, batch, image, is_real):
        o = self.model(image)
        o = global_average_pooling_2d(o)
        o_expected = int(is_real) * torch.ones(len(image), device=image.device, dtype=torch.long)
        batch['o_expected'] = o_expected

        return {
            'classification': OutputClassification2(
                o, o_expected,
                criterion_fn=LossMsePacked,  # LSGan loss function
            )
        }


class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.model = trw.layers.ConvsTransposeBase(
            2,
            input_channels=latent_size,
            channels=[256, 128, 64, 1],
            convolution_kernels=4,
            strides=[1, 2, 2, 2],
            batch_norm_kwargs={},
            paddings=[0, 1, 1, 1],
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            squash_function=torch.tanh,
            target_shape=[28, 28]
        )

    def forward(self, batch, latent):
        latent = latent.unsqueeze(2).unsqueeze(2)
        o = self.model(latent)
        return o, collections.OrderedDict([
            ('image', OutputEmbedding(o)),
        ])


def create_model(options):
    latent_size = 64

    generator = Generator(latent_size=latent_size)
    discriminator = Discriminator()

    optimizer_fn = functools.partial(torch.optim.Adam, lr=0.001, betas=(0.5, 0.999))

    model = Gan(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=optimizer_fn,
        optimizer_generator_fn=optimizer_fn,
        real_image_from_batch_fn=get_image
    )

    return model


options = trw.train.create_default_options(num_epochs=50)
trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_callbacks,
)
model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=1,
    inputs_fn=lambda: trw.datasets.create_mnist_dataset(batch_size=32, normalize_0_1=True),
    eval_every_X_epoch=1,
    model_fn=create_model,
    run_prefix='mnist_dcgan2',
    optimizers_fn=None  # the module has its own optimizers
)
