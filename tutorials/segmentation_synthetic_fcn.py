import trw
import torch.nn as nn


def create_net(options):
    n = trw.simple_layers.Input([None, 1, 256, 256], 'images')
    n = trw.simple_layers.Conv2d(n, out_channels=16, kernel_size=5, stride=2)
    n = trw.simple_layers.ReLU(n)
    n = trw.simple_layers.Conv2d(n, out_channels=16, kernel_size=5, stride=2)
    n = trw.simple_layers.ReLU(n)
    n = trw.simple_layers.Conv2d(n, out_channels=16, kernel_size=5, stride=2)
    n = trw.simple_layers.Conv2d(n, out_channels=64, kernel_size=1)
    n = trw.simple_layers.OutputClassification(n, output_name='softmax', classes_name='targets')
    return trw.simple_layers.compile_nn([n])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.convs =

    def forward(self, batch):
        x = batch['image']
        x = self.unet(x)

        return {
            'segmentation': trw.train.OutputSegmentation(x, target_name='mask')
        }


trainer = trw.train.Trainer()

model, results = trainer.fit(
    trw.train.create_default_options(num_epochs=15),
    inputs_fn=lambda: trw.datasets.create_fake_symbols_2d_datasset(nb_samples=1000, image_shape=[256, 256], nb_classes_at_once=1, batch_size=50),
    run_prefix='synthetic_segmentation_unet',
    model_fn=lambda options: Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_scheduler_step_lr_fn(datasets=datasets, model=model, learning_rate=0.05, step_size=50, gamma=0.3))
