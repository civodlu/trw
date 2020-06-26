import trw.train
import trw.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from trw.reporting import len_batch
from trw.train import OutputEmbedding, OutputClassification
from trw.train.losses import one_hot


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.unet = trw.layers.UNet(dim=2, input_channels=3, n_outputs=3, activation_fn=nn.LeakyReLU)

    def forward(self, latent, segmentation):
        x = self.unet(segmentation)
        return torch.tanh(x)  # force -1..1 range


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.convs = trw.layers.convs_2d(
            3,
            [64, 128, 256, 2],
            convolution_kernels=[4, 4, 4, 3],
            strides=[2, 4, 4, 2],
            batch_norm_kwargs={},
            pooling_size=None,
            with_flatten=True,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True
        )

    def forward(self, input):
        x = self.convs(input)
        return x


def get_image(batch):
    return 2 * batch['images'] - 1


def get_facade_outline(batch):
    return 2 * batch['segmentations'] - 1


def create_model(options):
    latent_size = 1

    discriminator = Discriminator()
    generator = Generator(latent_size)

    optimizer_fn = functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999))

    model = trw.layers.GanConditional(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=optimizer_fn,
        optimizer_generator_fn=optimizer_fn,
        image_from_batch_fn=get_image,
        observed_discriminator_fn=lambda _: [],
        observed_generator_fn=get_facade_outline,
    )

    model.apply(functools.partial(normal_init, mean=0.0, std=0.02))
    return model


def per_epoch_callbacks():
    return [
        trw.train.CallbackSkipEpoch(10, [
            trw.train.CallbackReportingExportSamples(split_exclusions=['valid', 'test']),
        ]),
        trw.train.CallbackExportSamples(),
        trw.train.CallbackEpochSummary(),
    ]


def pre_training_callbacks():
    return [
        trw.train.CallbackReportingStartServer(),
    ]


options = trw.train.create_default_options(num_epochs=200)

trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_callbacks,
    callbacks_pre_training_fn=pre_training_callbacks)

model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=1,
    inputs_fn=lambda: trw.datasets.create_facades_dataset(batch_size=1),
    eval_every_X_epoch=20,
    model_fn=create_model,
    run_prefix='facade_pix2pix',
    optimizers_fn=None  # the module has its own optimizers
)

print('DONE')