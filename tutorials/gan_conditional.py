import trw.train
import trw.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from trw.layers import Flatten
from trw.reporting import len_batch
from trw.train import OutputEmbedding, OutputClassification


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def per_epoch_callbacks():
    return [
        trw.train.CallbackExportSamples(),
        trw.train.CallbackEpochSummary(),
    ]


class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()

        self.convs_t = trw.layers.ConvsTransposeBase(
            2,
            input_channels=latent_size,
            channels=[1024, 512, 256, 1],
            convolution_kernels=4,
            strides=[1, 2, 2, 2],
            batch_norm_kwargs={},
            paddings=[0, 1, 1, 1],
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            squash_function=torch.tanh,
            target_shape=[28, 28]
        )

    def forward(self, input):
        x = self.convs_t(input)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.convs = trw.layers.convs_2d(
            1,
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


class GanConditionalDc(nn.Module):
    """
    Conditional GAN implementation

    Notes:
        Here the module will have its own optimizer. The :class:`trw.train.Trainer` should have ``optimizers_fn``
        set to ``None``.
    """
    def __init__(
            self,
            discriminator,
            generator,
            latent_size,
            optimizer_discriminator_fn,
            optimizer_generator_fn,
            image_from_batch_fn,
            observed_discriminator_fn,
            observed_generator_fn,
            criterion_fn=functools.partial(F.cross_entropy, reduction='none'),
            train_split_name='train',
    ):
        """

        Args:
            discriminator: a discriminator taking input ``image_from_batch_fn(batch)`` and
                returning Nx2 output (without the activation function applied)
            generator: a generator taking as input [N, latent_size, [1] * dim], with dim=2 for 2D images
                and returning as output the same shape as ``image_from_batch_fn(batch)``. Last layer should not
                be apply an activation function.
            latent_size: the latent size
            optimizer_discriminator_fn: the optimizer function to be used for the discriminator. Takes
                as input a model and return the trainable parameters
            optimizer_generator_fn: the optimizer function to be used for the generator. Takes
                as input a model and return the trainable parameters
            image_from_batch_fn: a function to extract the relevant image from the batch. Takes as input
                a batch and return an image
            criterion_fn: the classification criterion to be optimized
            train_split_name: only this split will be used for the training
        """
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.optmizer_discriminator = optimizer_discriminator_fn(params=self.discriminator.parameters())
        self.optmizer_generator = optimizer_generator_fn(params=self.generator.parameters())
        self.image_from_batch_fn = image_from_batch_fn
        self.latent_size = latent_size
        self.criterion_fn = criterion_fn
        self.train_split_name = train_split_name

    def forward(self, batch):
        if batch['split_name'] != self.train_split_name:
            return {}

        nb_samples = len_batch(batch)
        images_real = self.image_from_batch_fn(batch)

        real = torch.ones(nb_samples, dtype=torch.long, device=images_real.device)
        fake = torch.zeros(nb_samples, dtype=torch.long, device=images_real.device)

        self.optmizer_generator.zero_grad()
        self.discriminator.zero_grad()

        # generator
        with torch.no_grad():
            z = torch.randn(nb_samples, self.latent_size, device=images_real.device)
            z = z.view([z.shape[0], z.shape[1]] + [1] * (len(images_real.shape) - 2))

        images_fake = self.generator(z)

        output_generator = self.discriminator(images_fake)
        generator_loss = self.criterion_fn(output_generator, real)
        generator_loss_mean = generator_loss.mean()

        # discriminator: train with all real
        output_real = self.discriminator(images_real)
        loss_real = self.criterion_fn(output_real, real)

        # discriminator: train with all fakes
        output_fake = self.discriminator(images_fake.detach())
        loss_fake = self.criterion_fn(output_fake, fake)

        discriminator_loss_mean = (loss_fake + loss_real).mean() / 2

        # model updates
        if generator_loss_mean.requires_grad:
            generator_loss_mean.backward()
            self.optmizer_generator.step()
            discriminator_loss_mean.backward()
            self.optmizer_discriminator.step()

        # create some stats
        batch['real'] = real
        batch['fake'] = fake

        return {
            'images_fake': OutputEmbedding(images_fake),
            # do NOT record the gradient here in case the trainer optimizer was not set to ``None``
            'classifier_true': OutputClassification(output_real.data, classes_name='real'),
            'classifier_fake': OutputClassification(output_fake.data, classes_name='fake'),
        }


def get_image(batch):
    return 2 * batch['images'] - 1


def create_model(options):
    latent_size = 64

    discriminator = Discriminator()
    generator = Generator(latent_size)

    optimizer_fn = functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999))

    model = GanConditionalDc(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=optimizer_fn,
        optimizer_generator_fn=optimizer_fn,
        image_from_batch_fn=get_image
    )

    model.apply(functools.partial(normal_init, mean=0.0, std=0.01))
    return model


options = trw.train.create_default_options(num_epochs=100)
trainer = trw.train.Trainer(callbacks_per_epoch_fn=per_epoch_callbacks)
model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=1,
    inputs_fn=lambda: trw.datasets.create_mnist_datasset(batch_size=32, normalize_0_1=True),
    eval_every_X_epoch=1,
    model_fn=create_model,
    run_prefix='mnist_gan_conditional',
    optimizers_fn=None  # the module has its own optimizers
)
