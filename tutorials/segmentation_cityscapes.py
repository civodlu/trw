# we already use multiprocessing, multithreading will not improve anything here so disable it
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torchvision
import numpy as np
import trw


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.unet = trw.layers.UNet(dim=2, input_channels=3, n_classes=35, base_filters=32)

    def forward(self, batch):
        x = batch['image']
        x = x.float()
        x = self.unet(x)

        return {
            'segmentation': trw.train.OutputSegmentation(x, target_name='segmentation', collect_only_non_training_output=False)
        }


trainer = trw.train.Trainer(callbacks_pre_training_fn=None)
model, results = trainer.fit(
    trw.train.create_default_options(num_epochs=200),
    inputs_fn=lambda: trw.datasets.create_cityscapes_dataset(batch_size=2),
    run_prefix='cityscapes_segmentation_unet',
    model_fn=lambda options: Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(datasets=datasets, model=model, learning_rate=0.001, step_size=50, gamma=0.3))
