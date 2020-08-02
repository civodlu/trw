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
        - simple GAN (i.e., no observation)

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
            real_image_from_batch_fn,
            observed_discriminator_fn=None,
            observed_generator_fn=None,
            criterion_fn=functools.partial(F.cross_entropy, reduction='none'),
            train_split_name='train',
            l1_lambda=0,
            provide_source_to_discriminator=False,
    ):
        """

        Args:
            discriminator: a discriminator taking input ``image_from_batch_fn(batch)`` and
                returning Nx2 output (without the activation function applied)
            generator: a generator taking as input [N, latent_size, [1] * dim], with dim=2 for 2D images
                and returning as output the same shape as ``image_from_batch_fn(batch)``. Last layer should not
                be apply an activation function.
            latent_size: the latent size (random vector to seed the generator), `None` for no random latent (e.g., pix2pix)
            optimizer_discriminator_fn: the optimizer function to be used for the discriminator. Takes
                as input a model and return the trainable parameters
            optimizer_generator_fn: the optimizer function to be used for the generator. Takes
                as input a model and return the trainable parameters
            real_image_from_batch_fn: a function to extract the relevant image from the batch. Takes as input
                a batch and return an image
            criterion_fn: the classification criterion to be optimized
            train_split_name: only this split will be used for the training
            observed_discriminator_fn: function taking argument a batch and returning a tensor or a list of tensors
                to be used by the discriminator
            observed_generator_fn: function taking argument a batch and returning a tensor or a list of
                tensors to be used by the generator
            l1_lambda: the weight of the L1 loss to be applied on the generator
            provide_source_to_discriminator: if True, an additional arguments ``source`` provided to the discriminator.
                It can take value ``fake`` if it was generated or ``real`` if it was not generated. It is sometimes
                useful to know the source of the data so that it can be handled differently in the discriminator
        """
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.optmizer_discriminator = optimizer_discriminator_fn(params=self.discriminator.parameters())
        self.optmizer_generator = optimizer_generator_fn(params=self.generator.parameters())
        self.image_from_batch_fn = real_image_from_batch_fn
        self.latent_size = latent_size
        self.criterion_fn = criterion_fn
        self.train_split_name = train_split_name
        self.observed_discriminator_fn = observed_discriminator_fn
        self.observed_generator_fn = observed_generator_fn
        self.l1_lambda = l1_lambda
        self.provide_source_to_discriminator = provide_source_to_discriminator

    def _get_observed(self, batch):
        if self.observed_discriminator_fn is not None:
            observed_discriminator = self.observed_discriminator_fn(batch)
            assert isinstance(observed_discriminator, collections.Mapping), \
                f'observed func should return a dict. Got={type(observed_discriminator)}'
        else:
            observed_discriminator = {}

        if self.observed_generator_fn is not None:
            observed_generator = self.observed_generator_fn(batch)
            assert isinstance(observed_generator, collections.Mapping), \
                f'observed func should return a dict. Got={type(observed_generator)}'
        else:
            observed_generator = {}

        return observed_discriminator, observed_generator

    def forward(self, batch):
        if 'split_name' not in batch:
            # MUST have the split name!
            return {}

        observed_discriminator, observed_generator = self._get_observed(batch)
        nb_samples = len_batch(batch)
        images_real = self.image_from_batch_fn(batch)

        # generate a fake image
        if self.latent_size is not None:
            with torch.no_grad():
                z = torch.randn(nb_samples, self.latent_size, device=images_real.device)
                z = z.view([z.shape[0], z.shape[1]] + [1] * (len(images_real.shape) - 2))
        else:
            z = None

        images_fake = self.generator(z, **observed_generator)
        assert images_fake.shape == images_real.shape, f'generator output must have the same size as discriminator ' \
                                                       f'input! Generator={images_fake.shape}, ' \
                                                       f'Input={images_real.shape}'

        if batch['split_name'] != self.train_split_name:
            # we are in valid/test mode, return only the generated image!
            return collections.OrderedDict([
                ('fake', OutputEmbedding(images_fake.detach())),
                ('real', OutputEmbedding(images_real.detach())),
                ])

        #
        # train discriminator
        #

        # discriminator: train with all real
        self.optmizer_discriminator.zero_grad()
        if self.provide_source_to_discriminator:
            assert 'source' not in observed_discriminator, 'a conflicting `source` was already provided! ' \
                                                           'change the observed name or set ' \
                                                           '`provide_source_to_discriminator` to False'
            observed_discriminator['source'] = 'real'
        discriminator_output_real = self.discriminator(images_real, **observed_discriminator)  # , real=True

        # discriminator: train with all fakes
        if self.provide_source_to_discriminator:
            observed_discriminator['source'] = 'fake'
        discriminator_output_fake = self.discriminator(images_fake.detach(), **observed_discriminator)  # , real=False

        # extend the real, fake label to the size of the discriminator output (e.g., PatchGan)
        shape_target = discriminator_output_real[:, 0].shape
        real = torch.ones(shape_target, dtype=torch.long, device=images_real.device, requires_grad=False)
        fake = torch.zeros(shape_target, dtype=torch.long, device=images_real.device, requires_grad=False)

        discriminator_loss_fake = self.criterion_fn(discriminator_output_fake, fake)
        discriminator_loss_real = self.criterion_fn(discriminator_output_real, real)
        discriminator_loss_mean = (discriminator_loss_fake + discriminator_loss_real).mean() / 2

        if discriminator_loss_mean.requires_grad:
            discriminator_loss_mean.backward()
            self.optmizer_discriminator.step()

        #
        # train generator
        #
        self.optmizer_generator.zero_grad()
        if self.provide_source_to_discriminator:
            observed_discriminator['source'] = 'fake'
            #observed_discriminator['source'] = 'real'
        discrimator_output_generator = self.discriminator(images_fake, **observed_discriminator)  # , real=False
        if self.l1_lambda > 0:
            generator_l1 = F.l1_loss(images_fake, images_real, reduction='mean')
        else:
            generator_l1 = 0

        generator_loss = self.criterion_fn(discrimator_output_generator, real)
        generator_loss_mean = generator_loss.mean() + self.l1_lambda * generator_l1

        if generator_loss_mean.requires_grad:
            generator_loss_mean.backward()
            self.optmizer_generator.step()

        # create some stats
        batch['real'] = real
        batch['fake'] = fake

        if self.provide_source_to_discriminator:
            # it doesn't make sense to collect this value as it changes within the batch
            # so delete it
            del observed_discriminator['source']

        observed_generator = [(f'observed_generator_{n}', OutputEmbedding(o)) for n, o in observed_generator.items()]
        observed_discriminator = [(f'observed_discriminator_{n}', OutputEmbedding(o)) for n, o in observed_discriminator.items()]

        return collections.OrderedDict([
            ('fake', OutputEmbedding(images_fake.detach())),
            ('real', OutputEmbedding(images_real.detach())),
            ('classifier_real', OutputClassification(discriminator_output_real.detach(), classes_name='real')),
            ('classifier_fake', OutputClassification(discriminator_output_fake.detach(), classes_name='fake')),
            ('classifier_generator', OutputClassification(discrimator_output_generator.detach(), classes_name='fake')),
        ] + observed_generator + observed_discriminator)
