import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools
from trw.train.losses import one_hot


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


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


class Discriminator(nn.Module):
    def __init__(self, nb_digits=10):
        super().__init__()

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


def get_image(batch):
    return 2 * batch['images'] - 1


def get_target(batch):
    return batch['targets']


def create_model(options):
    latent_size = 64

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
        observed_discriminator_fn=get_target,
        observed_generator_fn=get_target,
        l1_lambda=0.5
    )

    model.apply(functools.partial(normal_init, mean=0.0, std=0.01))
    return model


def per_epoch_callbacks():
    return [
        trw.train.CallbackReportingExportSamples(split_exclusions=['valid', 'test']),
        trw.train.CallbackEpochSummary(),
        trw.train.CallbackReportingRecordHistory(),
    ]


def pre_training_callbacks():
    return [
        trw.train.CallbackReportingStartServer(),
    ]


options = trw.train.create_default_options(num_epochs=10)
trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_callbacks,
    callbacks_pre_training_fn=pre_training_callbacks)
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
