import trw
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.unet = trw.layers.UNet_2d(in_channels=3, n_classes=5, padding=True, depth=1)

    def forward(self, batch):
        x = batch['image']
        x = self.unet(x)

        return {
            'segmentation': trw.train.OutputSegmentation(x, target_name='mask')
        }


trainer = trw.train.Trainer()

model, results = trainer.fit(
    trw.train.create_default_options(num_epochs=10),
    inputs_fn=lambda: trw.datasets.create_fake_symbols_2d_datasset(nb_samples=1000, image_shape=[256, 256], nb_classes_at_once=1, batch_size=50),
    run_prefix='synthetic_segmentation_unet',
    model_fn=lambda options: Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(datasets=datasets, model=model, learning_rate=0.1))

print('DONE')