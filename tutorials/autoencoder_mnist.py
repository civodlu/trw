import trw
import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_decoder = trw.layers.AutoencoderConvolutional(
            2,
            1,
            [8, 16, 32],
            [32, 16, 8, 1],  # make sure we are cropping the decoded output by adding another layer
            convolution_kernels=3,
            last_layer_is_output=True  # do not apply the activation on the last layer
        )

    def forward(self, batch):
        x = batch['images']
        encoded_x, decoded_x = self.encoder_decoder.forward_with_intermediate(x)
        decoded_x = torch.sigmoid(decoded_x)

        return {
            'regression': trw.train.OutputRegression(decoded_x, 'images'),
        }


def per_epoch_fn():
    callbacks = [
        trw.train.CallbackEpochSummary(),
        trw.train.CallbackSkipEpoch(
            nb_epochs=10,
            callbacks=[trw.train.CallbackExportSamples(dirname='random_samples', max_samples=5, split_exclusions=['train'])]),
    ]

    return callbacks


options = trw.train.create_default_options(num_epochs=100)
trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_fn,
    callbacks_post_training_fn=lambda: [trw.train.CallbackReportingExportSamples(max_samples=2000)],
)

model, results = trainer.fit(
    options,
    inputs_fn=lambda: trw.datasets.create_mnist_datasset(normalize_0_1=1),
    run_prefix='mnist_autoencoder',
    model_fn=lambda options: Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
        datasets=datasets, model=model, learning_rate=0.25))

