import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def per_epoch_callbacks():
    return [
        trw.train.CallbackExportSamples(),
        trw.train.CallbackEpochSummary(),
    ]


def get_image(batch):
    return 2 * batch['images'] - 1


def create_model(options):
    latent_size = 64

    discriminator = trw.layers.convs_2d(
            input_channels=1,
            channels=[64, 128, 256, 2],
            convolution_kernels=[4, 4, 4, 3],
            strides=[2, 2, 2, 1],
            batch_norm_kwargs={},
            pooling_size=None,
            with_flatten=True,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True
        )

    generator = trw.layers.ConvsTransposeBase(
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

    optimizer_fn = functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999))

    model = trw.layers.Gan(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=optimizer_fn,
        optimizer_generator_fn=optimizer_fn,
        image_from_batch_fn=get_image
    )

    model.apply(functools.partial(normal_init, mean=0.0, std=0.01))
    return model


options = trw.train.create_default_options(num_epochs=50)
trainer = trw.train.Trainer(callbacks_per_epoch_fn=per_epoch_callbacks)
model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=1,
    inputs_fn=lambda: trw.datasets.create_mnist_dataset(batch_size=32, normalize_0_1=True),
    eval_every_X_epoch=1,
    model_fn=create_model,
    run_prefix='mnist_dcgan2',
    optimizers_fn=None  # the module has its own optimizers
)
