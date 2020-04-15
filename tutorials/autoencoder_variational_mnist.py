import torch
import trw
from torch import nn
from trw.layers import AutoencoderConvolutionalVariational


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        batch_norm_kwargs = {}

        z_filters_in = 128
        z_filters_out = 16

        encoder = trw.layers.ConvsBase(
            cnn_dim=2,
            input_channels=1,
            channels=[32, 64, z_filters_in],
            convolution_kernels=3,
            batch_norm_kwargs=batch_norm_kwargs,
        )

        decoder = trw.layers.ConvsTransposeBase(
            cnn_dim=2,
            input_channels=z_filters_out,
            channels=[32, 16, 8, 1],
            strides=[2, 2, 2, 2],
            convolution_kernels=3,
            last_layer_is_output=True,
            # make sure the decoded image is within [0..1], required for
            # the binary cross entropy loss
            squash_function=torch.sigmoid
        )

        self.autoencoder = AutoencoderConvolutionalVariational(2, encoder, decoder, z_filters_in, z_filters_out)

    def forward(self, batch):
        images = batch['images']
        recon, mu, logvar = self.autoencoder.forward(images)

        loss = self.autoencoder.loss_function(recon, images, mu, logvar)
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


options = trw.train.create_default_options(num_epochs=200)
trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_fn,
    callbacks_pre_training_fn=None,
    callbacks_post_training_fn=lambda: [trw.train.CallbackExportSamples2(max_samples=2000)],
)

model, results = trainer.fit(
    options,
    inputs_fn=lambda: trw.datasets.create_mnist_datasset(normalize_0_1=1),
    run_prefix='mnist_autoencoder_variational',
    model_fn=lambda options: Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(
        datasets=datasets, model=model, learning_rate=0.01, step_size=70, gamma=0.1))
