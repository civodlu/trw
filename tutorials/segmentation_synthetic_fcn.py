import trw
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        convs = trw.layers.ConvsBase(cnn_dim=2, input_channels=3, channels=[16, 32, 64])
        fcnn = trw.layers.FullyConvolutional(
            cnn_dim=2,
            base_model=convs,
            deconv_filters=[64, 32, 16, 16],
            convolution_kernels=5,
            strides=2,
            nb_classes=5
        )
        self.fcnn = fcnn

    def forward(self, batch):
        x = batch['image']
        x = self.fcnn(x)

        return {
            'segmentation': trw.train.OutputSegmentation(x, target_name='mask')
        }


trainer = trw.train.Trainer()

model, results = trainer.fit(
    trw.train.create_default_options(num_epochs=50),
    inputs_fn=lambda: trw.datasets.create_fake_symbols_2d_dataset(nb_samples=1000, image_shape=[256, 256], nb_classes_at_once=1, batch_size=50, max_classes=5),
    run_prefix='synthetic_segmentation_fcnn',
    model_fn=lambda options: Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_scheduler_step_lr_fn(datasets=datasets, model=model, learning_rate=0.05, step_size=50, gamma=0.3))
