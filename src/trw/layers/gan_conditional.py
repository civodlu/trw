import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from trw.train import OutputEmbedding, OutputClassification, LossMsePacked


class GanHelper(nn.Module):
    def __init__(self,
                 discriminator,
                 generator,
                 latent_size,
                 optimizer_discriminator_fn,
                 optimizer_generator_fn,
                 real_image_from_batch_fn,
                 observed_discriminator_fn,
                 observed_generator_fn,
                 provide_source_to_discriminator,
                 l1_lambda,
                 criterion_fn,
                 train_split_name):

        super().__init__()
        self.provide_source_to_discriminator = provide_source_to_discriminator
        self.criterion_fn = criterion_fn
        self.observed_generator_fn = observed_generator_fn
        self.observed_discriminator_fn = observed_discriminator_fn
        self.real_image_from_batch_fn = real_image_from_batch_fn
        self.optimizer_generator = optimizer_generator_fn(params=generator.parameters())
        self.optimizer_discriminator = optimizer_discriminator_fn(params=discriminator.parameters())
        self.latent_size = latent_size
        self.generator = generator
        self.discriminator = discriminator
        self.l1_lambda = l1_lambda
        self.train_split_name = train_split_name

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

    def _forward(self, images_real, observed_generator):
        nb_samples = len(images_real)

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
        return images_fake

    def _backward_d(self, images_real, images_fake, observed_discriminator):
        if self.provide_source_to_discriminator:
            assert 'source' not in observed_discriminator, 'a conflicting `source` was already provided! ' \
                                                           'change the observed name or set ' \
                                                           '`provide_source_to_discriminator` to False'
            observed_discriminator['source'] = 'real'
        discriminator_output_real = self.discriminator(images_real, **observed_discriminator)

        # discriminator: train with all fakes
        if self.provide_source_to_discriminator:
            observed_discriminator['source'] = 'fake'
        discriminator_output_fake = self.discriminator(images_fake.detach(), **observed_discriminator)

        # extend the real, fake label to the size of the discriminator output (e.g., PatchGan)
        shape_target = discriminator_output_real[:, 0].shape
        real = torch.ones(shape_target, dtype=torch.long, device=images_real.device, requires_grad=False)
        fake = torch.zeros(shape_target, dtype=torch.long, device=images_real.device, requires_grad=False)

        discriminator_loss_fake = self.criterion_fn(discriminator_output_fake, fake)
        discriminator_loss_real = self.criterion_fn(discriminator_output_real, real)
        discriminator_loss_mean = (discriminator_loss_fake + discriminator_loss_real).mean() / 2
        return real, fake, discriminator_loss_mean, discriminator_output_real, discriminator_loss_real, discriminator_output_fake, discriminator_loss_fake

    def _backward_g(self, images_real, images_fake, observed_discriminator):
        if self.provide_source_to_discriminator:
            observed_discriminator['source'] = 'fake'
        discrimator_output_generator = self.discriminator(images_fake, **observed_discriminator)  # , real=False
        if self.l1_lambda > 0:
            generator_l1 = F.l1_loss(images_fake, images_real, reduction='mean')
        else:
            generator_l1 = 0

        shape_target = discrimator_output_generator[:, 0].shape
        real = torch.ones(shape_target, dtype=torch.long, device=images_real.device, requires_grad=False)
        generator_loss = self.criterion_fn(discrimator_output_generator, real)
        generator_loss_mean = generator_loss.mean() + self.l1_lambda * generator_l1
        return discrimator_output_generator, generator_loss_mean

    def update_parameters(self, batch):
        if 'split_name' not in batch:
            # MUST have the split name!
            return {}

        observed_discriminator, observed_generator = self._get_observed(batch)
        images_real = self.real_image_from_batch_fn(batch)
        images_fake = self._forward(images_real, observed_generator)

        if batch['split_name'] != self.train_split_name:
            # we are in valid/test mode, return only the generated image!
            return collections.OrderedDict([
                ('fake', OutputEmbedding(images_fake.detach())),
                ('real', OutputEmbedding(images_real.detach())),
                ])

        # discriminator: train with all real
        self.optimizer_discriminator.zero_grad()
        real, \
        fake, \
        discriminator_loss_mean, \
        discriminator_output_real, \
        discriminator_loss_real, \
        discriminator_output_fake, \
        discriminator_loss_fake = self._backward_d(images_real, images_fake, observed_discriminator)
        if discriminator_loss_mean.requires_grad:
            # in case we are in ``eval mode`` for the callbacks, avoid parameter updates
            discriminator_loss_mean.backward()
        self.optimizer_discriminator.step()

        # train generator
        self.optimizer_generator.zero_grad()
        discrimator_output_generator, generator_loss_mean = self._backward_g(images_real, images_fake, observed_discriminator)
        if generator_loss_mean.requires_grad:
            generator_loss_mean.backward()
        self.optimizer_generator.step()

        if self.provide_source_to_discriminator:
            # it doesn't make sense to collect this value as it changes within the batch
            # so delete it
            del observed_discriminator['source']

        observed_generator = [(f'observed_generator_{n}', OutputEmbedding(o)) for n, o in observed_generator.items()]
        observed_discriminator = [(f'observed_discriminator_{n}', OutputEmbedding(o)) for n, o in observed_discriminator.items()]

        # create some stats
        batch['real'] = real
        batch['fake'] = fake

        return collections.OrderedDict([
            ('fake', OutputEmbedding(images_fake.detach())),
            ('real', OutputEmbedding(images_real.detach())),
            ('classifier_real', OutputClassification(discriminator_output_real.detach(), classes_name='real')),
            ('classifier_fake', OutputClassification(discriminator_output_fake.detach(), classes_name='fake')),
            ('classifier_generator', OutputClassification(discrimator_output_generator.detach(), classes_name='fake')),
        ] + observed_generator + observed_discriminator)


"""
class GanCycle(nn.Module):
    def __init__(self,
                 discriminator_a,
                 discriminator_b,
                 generator_a,
                 generator_b,
                 latent_size_a,
                 latent_size_b,
                 optimizer_discriminator_fn,
                 optimizer_generator_fn,
                 real_image_a_from_batch_fn,
                 real_image_b_from_batch_fn,
                 observed_discriminator_a_fn=None,
                 observed_discriminator_b_fn=None,
                 observed_generator_a_fn=None,
                 observed_generator_b_fn=None,
                 criterion_fn=LossMsePacked(),
                 train_split_name='train',
                 l1_lambda_a=0,
                 l1_lambda_b=0,
                 provide_source_to_discriminator=False
                 ):

        super().__init__()
        self.gan_a = GanHelper(
            discriminator=discriminator_a,
            generator=generator_a,
            latent_size=latent_size_a,
            optimizer_discriminator_fn=None,
            optimizer_generator_fn=None,
            real_image_from_batch_fn=real_image_a_from_batch_fn,
            observed_discriminator_fn=observed_discriminator_a_fn,
            observed_generator_fn=observed_generator_a_fn,
            provide_source_to_discriminator=provide_source_to_discriminator,
            l1_lambda=l1_lambda_a,
            criterion_fn=criterion_fn,
            train_split_name=train_split_name
        )

        self.gan_b = GanHelper(
            discriminator=discriminator_b,
            generator=generator_b,
            latent_size=latent_size_b,
            optimizer_discriminator_fn=None,
            optimizer_generator_fn=None,
            real_image_from_batch_fn=real_image_b_from_batch_fn,
            observed_discriminator_fn=observed_discriminator_b_fn,
            observed_generator_fn=observed_generator_b_fn,
            provide_source_to_discriminator=provide_source_to_discriminator,
            l1_lambda=l1_lambda_b,
            criterion_fn=criterion_fn,
            train_split_name=train_split_name
        )

        self.optimizer_discriminator = optimizer_discriminator_fn(itertools.chain(
            self.gan_a.discriminator.parameters(),
            self.gan_b.discriminator.parameters()
        ))

        self.optimizer_generator = optimizer_generator_fn(itertools.chain(
            self.gan_a.generator.parameters(),
            self.gan_b.generator.parameters()
        ))

    def forward(self, batch):
        if 'split_name' not in batch:
            # MUST have the split name!
            return {}

        observed_discriminator_a, observed_generator_a = self.gan_a._get_observed(batch)
        images_real_a = self.gan_a.real_image_from_batch_fn(batch)
        images_fake_a = self.gan_a._forward(images_real_a, observed_generator_a)

        observed_discriminator_b, observed_generator_b = self.gan_b._get_observed(batch)
        images_real_b = self.gan_b.real_image_from_batch_fn(batch)
        images_fake_b = self.gan_b._forward(images_real_b, observed_generator_b)

        if batch['split_name'] != self.train_split_name:
            # we are in valid/test mode, return only the generated image!
            return collections.OrderedDict([
                ('fake_a', OutputEmbedding(images_fake_a.detach())),
                ('real_a', OutputEmbedding(images_real_a.detach())),
                ('fake_b', OutputEmbedding(images_fake_b.detach())),
                ('real_b', OutputEmbedding(images_real_b.detach())),
            ])

    def backward_d(self, images_real_a, images_fake_a, images_real_b, images_fake_b, observed_discriminator_a, observed_discriminator_b):
        self.optimizer_discriminator.zero_grad()

        real_a, \
        fake_a, \
        discriminator_loss_mean_a, \
        discriminator_output_real_a, \
        discriminator_loss_real_a, \
        discriminator_output_fake_a, \
        discriminator_loss_fake_a = self.gan_a._backward_d(images_real_a, images_fake_a, observed_discriminator_a)

        real_b, \
        fake_b, \
        discriminator_loss_mean_b, \
        discriminator_output_real_b, \
        discriminator_loss_real_b, \
        discriminator_output_fake_b, \
        discriminator_loss_fake_b = self.gan_b._backward_d(images_real_b, images_fake_b, observed_discriminator_b)

        discriminator_loss_mean = discriminator_loss_fake_a + discriminator_loss_fake_b
        if discriminator_loss_mean.requires_grad:
            # in case we are in ``eval mode`` for the callbacks, avoid parameter updates
            discriminator_loss_mean.backward()
        self.optimizer_discriminator.step()

    def backward_g(self, images_real_a, images_fake_a, observed_discriminator_a, images_real_b, images_fake_b, observed_discriminator_b):
        self.optimizer_generator.zero_grad()

        discrimator_output_generator_a, generator_loss_mean_a = self.gan_a._backward_g(images_real_a, images_fake_a, observed_discriminator_a)
        discrimator_output_generator_b, generator_loss_mean_b = self.gan_b._backward_g(images_real_b, images_fake_b, observed_discriminator_b)
        
        #cycle_a_loss = self.gan_b._forward(images_fake_a)

        loss_g = generator_loss_mean_a + generator_loss_mean_b
        if loss_g.requires_grad:
            # in case we are in ``eval mode`` for the callbacks, avoid parameter updates
            loss_g.backward()
        self.optimizer_generator.step()
"""

"""
class GanConditional(nn.Module):
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
            criterion_fn=LossMsePacked(),
            train_split_name='train',
            l1_lambda=0,
            provide_source_to_discriminator=False,
    ):
        
        super().__init__()
        self.gan_helper = GanHelper(
            discriminator=discriminator,
            generator=generator,
            latent_size=latent_size,
            optimizer_discriminator_fn=optimizer_discriminator_fn,
            optimizer_generator_fn=optimizer_generator_fn,
            real_image_from_batch_fn=real_image_from_batch_fn,
            observed_discriminator_fn=observed_discriminator_fn,
            observed_generator_fn=observed_generator_fn,
            provide_source_to_discriminator=provide_source_to_discriminator,
            l1_lambda=l1_lambda,
            criterion_fn=criterion_fn,
            train_split_name=train_split_name
        )

    def forward(self, batch):
        outputs = self.gan_helper.update_parameters(batch)
        return outputs
"""


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
            #criterion_fn=functools.partial(F.cross_entropy, reduction='none'),
            criterion_fn=LossMsePacked(),
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

    def _generate_latent(self, images_real):
        nb_samples = len(images_real)

        if self.latent_size is not None:
            with torch.no_grad():
                z = torch.randn(nb_samples, self.latent_size, device=images_real.device)
                z = z.view([z.shape[0], z.shape[1]] + [1] * (len(images_real.shape) - 2))
        else:
            z = None

        return z

    def forward(self, batch):
        if 'split_name' not in batch:
            # MUST have the split name!
            return {}

        observed_discriminator, observed_generator = self._get_observed(batch)
        images_real = self.image_from_batch_fn(batch)

        # generate a fake image
        latent = self._generate_latent(images_real)

        images_fake = self.generator(latent, **observed_generator)
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
        discriminator_output_real = self.discriminator(images_real, **observed_discriminator)

        # discriminator: train with all fakes
        if self.provide_source_to_discriminator:
            observed_discriminator['source'] = 'fake'
        discriminator_output_fake = self.discriminator(images_fake.detach(), **observed_discriminator)

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

#def create_loss_terms(outputs, batch, is_training):



