import trw.train
import trw.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import collections

batch_size = 32
latent_size = 64
mnist_size = 28 * 28
hidden_size = 256


class Flatten(nn.Module):
    def forward(self, i):
        return i.view(i.shape[0], -1)


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


class BranchedModule(nn.ModuleDict):
    def __init__(self):
        super().__init__()
        self['discriminator'] = Discriminator()
        self['generator'] = Generator()
    
    def forward(self, batch):
        dataset_name = batch.get('dataset_name')
        z = batch['z']
        discriminator = self['discriminator']
        generator = self['generator']
        
        if dataset_name == 'discriminator':
            images = batch['images']
            
            d_real = discriminator(images)
            d_fake = discriminator(generator(z))
            
            return {
                # by default the `classification_real` and `classification_fake` will be accumulated
                # the discriminator must recognize the real image and the fake image
                'classification_real': trw.train.OutputClassification(d_real, 'label_real', loss_scaling=0.5),
                'classification_fake': trw.train.OutputClassification(d_fake, 'label_fake', loss_scaling=0.5)
            }
        elif dataset_name == 'generator':
            generated_image = generator(z)
            d_fake = discriminator(generated_image)
            
            return {
                # the generator must maximize the error of the discriminator
                'classification_fake': trw.train.OutputClassification(d_fake, 'label_real'),
                'generated_image': trw.train.OutputEmbedding(generated_image),
            }
        else:
            assert 0, 'dataset not handled!'


def create_model(options):
    model = BranchedModule()
    model.apply(functools.partial(normal_init, mean=0.0, std=0.01))
    return model


def create_input():
    datasets = trw.datasets.create_mnist_datasset(batch_size=batch_size)
    # datasets = trw.datasets.create_mnist_datasset(batch_size=batch_size, root='D:\data')
    
    mnist_train = datasets[0]['mnist']['train']
    images = datasets[0]['mnist']['train'].split['images'] / 128.0 - 1.0
    nb_samples = len(mnist_train.split['images'])
    
    def random_z(batch):
        return torch.randn(batch_size, latent_size, device=device)
    
    generator = {
        'z': random_z,
        'label_real': torch.zeros(nb_samples, dtype=torch.int64)
    }
    
    discriminator = {
        'z': random_z,
        'label_real': torch.zeros(nb_samples, dtype=torch.int64),
        'label_fake': torch.ones(nb_samples, dtype=torch.int64),
        'images': images
    }
    
    datasets_gan = collections.OrderedDict([
        ('generator', {'train': trw.train.BatcherNumpyResampled(generator, batch_size=batch_size, nb_batches=1, shuffle=True)}),
        ('discriminator', {'train': trw.train.BatcherNumpyResampled(discriminator, batch_size=batch_size, nb_batches=1, shuffle=True)}),
    ])
    
    return datasets_gan


def per_epoch_callbacks():
    callbacks = [
        trw.train.CallbackExportSamples(),
        trw.train.CallbackEpochSummary(),
    ]
    
    return [
        trw.train.CallbackSkipEpoch(nb_epochs=200, callbacks=callbacks),
    ]


device = torch.device('cuda:0')
options = trw.train.create_default_options(num_epochs=150000, logging_directory=r'c:\tmp', device=device)
optimizers_fn = functools.partial(trw.train.create_adam_optimizers_scheduler_step_lr_fn, learning_rate=0.0001, step_size=150000, gamma=0.1)
trainer = trw.train.Trainer(callbacks_per_epoch_fn=per_epoch_callbacks)
model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=1,
    inputs_fn=create_input,
    eval_every_X_epoch=1,
    model_fn=create_model,
    run_prefix='mnist_dcgan',
    optimizers_fn=optimizers_fn)

