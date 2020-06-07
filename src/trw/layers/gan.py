from trw.reporting import len_batch
from trw.train import OutputEmbedding, OutputClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class Gan(nn.Module):
    """
    Implementation of ``Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks``,
        https://arxiv.org/abs/1511.06434

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
