import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.unet = trw.layers.UNetBase(2, input_channels=3 + latent_size, output_channels=3, channels=[64, 128, 256, 512])

    def forward(self, latent, segmentation):
        latent = latent.repeat([1, 1, segmentation.shape[2], segmentation.shape[3]])
        i = torch.cat([latent, segmentation], dim=1)
        x = self.unet(i)
        return torch.tanh(x)  # force -1..1 range


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.convs = trw.layers.convs_2d(
            6,
            [64, 128, 256, 512, 512, 2],
            convolution_kernels=[4, 4, 4, 4, 4, 1],
            strides=[2, 2, 2, 2, 1, 1],
            batch_norm_kwargs={},
            pooling_size=None,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True,
            bias=False
        )

    def forward(self, image, segmentation):
        x = self.convs(torch.cat([image, segmentation], dim=1))
        return x


def get_image(batch, source=None):
    return 2 * batch['images'] - 1


def get_observed(batch, source=None):
    return {
        'segmentation': 2 * batch['segmentations'] - 1
    }


def create_model(options):
    latent_size = 16

    discriminator = Discriminator()
    generator = Generator(latent_size)

    optimizer_fn = functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999))

    model = trw.layers.GanConditional(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=optimizer_fn,
        optimizer_generator_fn=optimizer_fn,
        real_image_from_batch_fn=get_image,
        observed_discriminator_fn=get_observed,
        observed_generator_fn=get_observed,
        l1_lambda=0.5,
    )

    generator.apply(init_weights)
    discriminator.apply(init_weights)
    return model


def per_epoch_callbacks():
    return [
        trw.train.CallbackSkipEpoch(4, [
            trw.train.CallbackReportingExportSamples(split_exclusions=['valid', 'test']),
        ], include_epoch_zero=True),
        trw.train.CallbackEpochSummary(),
    ]


def pre_training_callbacks():
    return [
        trw.train.CallbackReportingStartServer(),
    ]


options = trw.train.create_default_options(num_epochs=600, device='cuda:0')

trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_callbacks,
    callbacks_pre_training_fn=pre_training_callbacks)

model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=1,
    inputs_fn=lambda: trw.datasets.create_facades_dataset(
        batch_size=10,
        transforms_train=trw.transforms.TransformCompose([
            trw.transforms.TransformRandomFlip(axis=3),
            trw.transforms.TransformRandomCropPad([0, 16, 16], mode='symmetric')
        ])),
    eval_every_X_epoch=20,
    model_fn=create_model,
    run_prefix='facade_pix2pix',
    optimizers_fn=None  # the module has its own optimizers
)

print('DONE')