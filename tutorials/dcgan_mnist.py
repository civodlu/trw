import trw.train
import trw.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from trw.layers import Flatten


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
    def __init__(self, latent_size, d=128):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.deconv1 = nn.ConvTranspose2d(latent_size, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)

    def forward(self, input):
        input = input.view((input.shape[0], self.latent_size, 1, 1))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.deconv4(x)

        x = torch.tanh(x)
        x = x[:, 0:1, 2:30, 2:30]
        return x


class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 2, 3, 1, 0)
        self.flatten = Flatten()

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.conv4(x)
        x = self.flatten(x)
        return x


def get_image(batch):
    return 2 * batch['images'] - 1


def create_model(options):
    latent_size = 64

    discriminator = Discriminator()
    generator = Generator(latent_size)

    optimizer_fn = functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999))

    model = trw.layers.GanDc(
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
    run_prefix='mnist_dcgan2',
    optimizers_fn=None  # the module has its own optimizers
)
