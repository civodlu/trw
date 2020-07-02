import collections
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from trw.reporting import len_batch
from trw.train import OutputEmbedding, OutputClassification


class GanConditional(nn.Module):
    """
    Conditional GAN implementation. Generator and classifier can be conditioned on any variables in the batch.

    additionally, L1 loss is added to the generator loss.

    Examples:
        - generator conditioned by concatenating a one-hot attribute to the latent or conditioned
            by another image (e.g., using UNet)
        - discriminator conditioned by concatenating a one-hot image sized to the image
            or one-hot concatenated to intermediate layer

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
            l1_lambda=1,
    ):
        """

        Args:
            discriminator: a discriminator taking input ``image_from_batch_fn(batch)`` and
                returning Nx2 output (without the activation function applied)
            generator: a generator taking as input [N, latent_size, [1] * dim], with dim=2 for 2D images
                and returning as output the same shape as ``image_from_batch_fn(batch)``. Last layer should not
                be apply an activation function.
            latent_size: the latent size (random vector to seed the generator)
            optimizer_discriminator_fn: the optimizer function to be used for the discriminator. Takes
                as input a model and return the trainable parameters
            optimizer_generator_fn: the optimizer function to be used for the generator. Takes
                as input a model and return the trainable parameters
            image_from_batch_fn: a function to extract the relevant image from the batch. Takes as input
                a batch and return an image
            criterion_fn: the classification criterion to be optimized
            train_split_name: only this split will be used for the training
            observed_discriminator_fn: function taking argument a batch and returning a tensor or a list of tensors
                to be used by the discriminator
            observed_generator_fn: function taking argument a batch and returning a tensor or a list of
                tensors to be used by the generator
            l1_lambda: the weight of the L1 loss to be applied on the generator
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
        self.observed_discriminator_fn = observed_discriminator_fn
        self.observed_generator_fn = observed_generator_fn
        self.l1_lambda = l1_lambda

    def forward(self, batch):
        if 'split_name' not in batch or batch['split_name'] != self.train_split_name:
            return {}

        observed_discriminator = self.observed_discriminator_fn(batch)
        if not isinstance(observed_discriminator, list):
            observed_discriminator = [observed_discriminator]
        observed_generator = self.observed_generator_fn(batch)
        if not isinstance(observed_generator, list):
            observed_generator = [observed_generator]

        nb_samples = len_batch(batch)
        images_real = self.image_from_batch_fn(batch)

        real = torch.ones(nb_samples, dtype=torch.long, device=images_real.device, requires_grad=False)
        fake = torch.zeros(nb_samples, dtype=torch.long, device=images_real.device, requires_grad=False)

        self.optmizer_generator.zero_grad()
        self.discriminator.zero_grad()

        # generator
        with torch.no_grad():
            z = torch.randn(nb_samples, self.latent_size, device=images_real.device)
            z = z.view([z.shape[0], z.shape[1]] + [1] * (len(images_real.shape) - 2))

        images_fake = self.generator(z, *observed_generator)
        assert images_fake.shape == images_real.shape, f'generator output must have the same size as discriminator ' \
                                                       f'input! Generator={images_fake.shape}, ' \
                                                       f'Input={images_real.shape}'

        output_generator = self.discriminator(images_fake, *observed_discriminator)
        generator_l1 = F.l1_loss(images_fake, images_real)
        generator_loss = self.criterion_fn(output_generator, real)
        generator_loss_mean = generator_loss.mean() + self.l1_lambda * generator_l1

        if generator_loss_mean.requires_grad:
            generator_loss_mean.backward()
            self.optmizer_generator.step()

        # discriminator: train with all real
        output_real = self.discriminator(images_real, *observed_discriminator)
        loss_real = self.criterion_fn(output_real, real)

        # discriminator: train with all fakes
        output_fake = self.discriminator(images_fake.detach(), *observed_discriminator)
        loss_fake = self.criterion_fn(output_fake, fake)

        discriminator_loss_mean = (loss_fake + loss_real).mean() / 2

        # model updates
        if discriminator_loss_mean.requires_grad:
            discriminator_loss_mean.backward()
            self.optmizer_discriminator.step()

        # create some stats
        batch['real'] = real
        batch['fake'] = fake

        observed_generator = [(f'observed_generator_{n}', OutputEmbedding(o)) for n, o in enumerate(observed_generator)]
        observed_discriminator = [(f'observed_discriminator_{n}', OutputEmbedding(o)) for n, o in enumerate(observed_discriminator)]

        return collections.OrderedDict([
            ('fake', OutputEmbedding(images_fake.detach())),
            ('real', OutputEmbedding(images_real.detach())),
            ('classifier_real', OutputClassification(output_real.detach(), classes_name='real')),
            ('classifier_fake', OutputClassification(output_fake.detach(), classes_name='fake')),
        ] + observed_generator + observed_discriminator)
