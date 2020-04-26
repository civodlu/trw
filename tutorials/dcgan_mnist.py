import trw.train
import trw.datasets
from trw.layers import Flatten
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

batch_size = 32
latent_size = 64
mnist_size = 28 * 28
hidden_size = 256


class Generator(nn.Module):
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_size, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)

    def forward(self, input):
        input = input.view((input.shape[0], latent_size, 1, 1))
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


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class DcGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.optmizer_discriminator = torch.optim.Adam(params=self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optmizer_generator = torch.optim.Adam(params=self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self, batch):
        if batch['split_name'] != 'train':
            return {}

        nb_samples = trw.train.len_batch(batch)
        images_real = 2 * batch['images'] - 1

        real = torch.ones(nb_samples, dtype=torch.long, device=device)
        fake = torch.zeros(nb_samples, dtype=torch.long, device=device)
        criterion = nn.CrossEntropyLoss(reduction='none')

        self.optmizer_generator.zero_grad()
        self.discriminator.zero_grad()

        # generator
        z = torch.randn(nb_samples, latent_size, device=images_real.device)
        images_fake = self.generator(z)

        output = self.discriminator(images_fake)
        generator_loss = criterion(output, real)
        generator_loss_mean = generator_loss.mean()

        # discriminator: train with all real
        output_real = self.discriminator(images_real)
        loss_real = criterion(output_real, real)

        # discriminator: train with all fakes
        output_fake = self.discriminator(images_fake.detach())
        loss_fake = criterion(output_fake, fake)

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
            'images_fake': trw.train.OutputEmbedding(images_fake),
            'classifier_true': trw.train.OutputClassification(output_real, criterion_fn=None, classes_name='real'),
            'classifier_fake': trw.train.OutputClassification(output_fake, criterion_fn=None, classes_name='fake'),
        }


def per_epoch_callbacks():
    callbacks = [
        trw.train.CallbackExportSamples(),
        trw.train.CallbackEpochSummary(),
    ]

    return [
        trw.train.CallbackSkipEpoch(nb_epochs=1, callbacks=callbacks, include_epoch_zero=True),
    ]


def create_model(options):
    model = DcGAN()
    model.apply(functools.partial(normal_init, mean=0.0, std=0.01))
    return model


device = torch.device('cuda:0')
options = trw.train.create_default_options(num_epochs=100, device=device)

trainer = trw.train.Trainer(
    callbacks_pre_training_fn=None,
    callbacks_per_epoch_fn=per_epoch_callbacks
)

model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=1,
    inputs_fn=lambda: trw.datasets.create_mnist_datasset(batch_size=batch_size, normalize_0_1=True),
    eval_every_X_epoch=1,
    model_fn=create_model,
    run_prefix='mnist_dcgan2',
    optimizers_fn=None  # the module has its own optimizers
)
