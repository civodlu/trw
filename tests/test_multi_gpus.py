import torch.nn as nn
import torch.nn.functional as F
import torch
import trw
from unittest import TestCase
import functools
import warnings


class NetClassic(nn.Module):
    """
    Defines our model for MNIST
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 2)
        self.fc1 = nn.Linear(20 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu_fc1 = nn.ReLU()
        self.relu_conv1 = nn.ReLU()

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images'] / 255.0

        x = self.relu_conv1(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)

        # here we create a softmax output that will use
        # the `targets` feature as classification target
        return {
            'softmax': trw.train.OutputClassification(x, 'targets')
        }


def create_model_simplified(options):
    n = trw.simple_layers.Input([None, 1, 28, 28], 'images')
    n = trw.simple_layers.Conv2d(n, out_channels=16, kernel_size=5, stride=2)
    n = trw.simple_layers.ReLU(n)
    n = trw.simple_layers.MaxPool2d(n, 2, 2)
    n = trw.simple_layers.Flatten(n)
    n = trw.simple_layers.Linear(n, 500)
    n = trw.simple_layers.ReLU(n)
    n = trw.simple_layers.Linear(n, 10)
    n = trw.simple_layers.OutputClassification(n, output_name='softmax', classes_name='targets')
    return trw.simple_layers.compile_nn([n])


def create_model_classic(options):
    model = NetClassic()
    model = trw.train.DataParallelExtended(model)
    return model


def create_dataset():
    split = trw.train.SequenceArray(
        {
            'images': torch.zeros((50, 1, 28, 28), dtype=torch.float32),
            'targets': torch.ones(50, dtype=torch.int64),
        }
    )
    return {
        'mnist': {
            'train': split
        }
    }


class TestMultiGpus(TestCase):
    def test_classic_torch(self):
        # here just make sure the use of multiple GPUs do not create errors
        nb_cuda_devices = torch.cuda.device_count()
        if nb_cuda_devices < 2:
            # we do not have enough GPUs, abot the test
            warnings.warn(f'This test can\'t be run. Requires CUDA devices=2, got={nb_cuda_devices}', ResourceWarning)
            return

        options = trw.train.create_default_options(num_epochs=5)
        trainer = trw.train.Trainer(
            callbacks_per_epoch_fn=None,
            callbacks_per_batch_loss_terms_fn=None,
            callbacks_post_training_fn=None,
            callbacks_per_batch_fn=None,
            callbacks_pre_training_fn=None
        )

        optimizer_fn = functools.partial(trw.train.create_sgd_optimizers_fn, learning_rate=0.01)
        final_model, results = trainer.fit(
            options,
            inputs_fn=create_dataset,
            model_fn=create_model_classic,
            optimizers_fn=optimizer_fn)

        assert trw.train.to_value(results['outputs']['mnist']['train']['overall_loss']['loss']) < 1e-5

    def test_simplified_layers(self):
        # here just make sure the use of multiple GPUs do not create errors
        nb_cuda_devices = torch.cuda.device_count()
        if nb_cuda_devices < 2:
            # we do not have enough GPUs, abot the test
            warnings.warn(f'This test can\'t be run. Requires CUDA devices=2, got={nb_cuda_devices}', ResourceWarning)
            return

        options = trw.train.create_default_options(num_epochs=5)
        trainer = trw.train.Trainer(
            callbacks_per_epoch_fn=None,
            callbacks_per_batch_loss_terms_fn=None,
            callbacks_post_training_fn=None,
            callbacks_per_batch_fn=None,
            callbacks_pre_training_fn=None
        )

        optimizer_fn = functools.partial(trw.train.create_sgd_optimizers_fn, learning_rate=0.01)
        final_model, results = trainer.fit(
            options,
            inputs_fn=create_dataset,
            model_fn=create_model_simplified,
            optimizers_fn=optimizer_fn)

        assert trw.train.to_value(results['outputs']['mnist']['train']['overall_loss']['loss']) < 1e-5

