import torch
import trw
from torch import nn
from trw.layers import AutoencoderConvolutionalVariational, crop_or_pad_fun
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        batch_norm_kwargs = {}

        z_filters_in = 64
        z_filters_out = 4

        activation = nn.LeakyReLU

        encoder = trw.layers.ConvsBase(
            cnn_dim=2,
            input_channels=1,
            channels=[16, 32, 64, z_filters_in],
            convolution_kernels=[6, 6, 6, 6],
            strides=2,
            pooling_size=None,
            batch_norm_kwargs=batch_norm_kwargs,
            activation=activation
        )

        decoder = trw.layers.ConvsTransposeBase(
            cnn_dim=2,
            input_channels=z_filters_out,
            channels=[32, 16, 8, 1],
            strides=[3, 2, 2, 2],
            convolution_kernels=[6, 6, 6, 6],
            last_layer_is_output=True,
            # make sure the decoded image is within [0..1], required for
            # the binary cross entropy loss
            squash_function=torch.sigmoid,
            activation=activation
        )

        self.autoencoder = AutoencoderConvolutionalVariational(2, encoder, decoder, z_filters_in, z_filters_out)

    def forward(self, batch):
        images = batch['images']
        recon, mu, logvar = self.autoencoder.forward(images)

        loss = self.autoencoder.loss_function(recon, images, mu, logvar, kullback_leibler_weight=0.1)
        return {
            'loss': trw.train.OutputLoss(loss),
            'recon': trw.train.OutputEmbedding(recon),
        }


def per_epoch_fn():
    callbacks = [
        trw.train.CallbackEpochSummary(),
        trw.train.CallbackSkipEpoch(
            nb_epochs=10,
            callbacks=[trw.train.CallbackExportSamples(dirname='random_samples', max_samples=5, split_exclusions=['train'])]),
    ]

    return callbacks


def pos_training_fn():
    return [
        trw.train.CallbackExportSamples2(max_samples=500),
        trw.train.CallbackSaveLastModel()
    ]


options = trw.train.create_default_options(num_epochs=200)
trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_fn,
    callbacks_pre_training_fn=None,
    callbacks_post_training_fn=pos_training_fn,
)

model, results = trainer.fit(
    options,
    inputs_fn=lambda: trw.datasets.create_mnist_datasset(normalize_0_1=1, batch_size=1024),
    run_prefix='mnist_autoencoder_variational',
    model_fn=lambda options: Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(
        datasets=datasets, model=model, learning_rate=0.001, step_size=120, gamma=0.1))


model.training = False
nb_images = 20
generated = model.autoencoder.decoder(torch.randn([nb_images, 4, 3, 3], dtype=torch.float32, device=trw.train.get_device(model)))

fig, axes = plt.subplots(nrows=1, ncols=nb_images, figsize=(nb_images, 2.5), sharey=True)
decoded_images = crop_or_pad_fun(generated, [28, 28])
image_width = decoded_images.shape[2]

for ax, img in zip(axes, decoded_images):
    curr_img = img.detach().to(torch.device('cpu'))
    ax.imshow(curr_img.view((image_width, image_width)), cmap='binary')

plt.show()
