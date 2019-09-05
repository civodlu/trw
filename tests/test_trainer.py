from unittest import TestCase
import collections
import numpy as np
import torch.nn as nn
import torch
import torch.utils.data
import trw.train
from . import utils
import functools


def create_simple_regression():
    """
    Create a simple dataset where output = input_1 * 2.0
    """
    datasets = collections.OrderedDict()
    datasets['simple'] = {
        'train': torch.utils.data.DataLoader(
            utils.NumpyDatasets(
                input_1=np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
                output=np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])
            ),
            batch_size=100
        )
    }
    return datasets

def create_random_input():
    """
    Create a simple dataset where output = input_1 * 2.0
    """
    datasets = collections.OrderedDict()
    datasets['simple'] = {
        'train': torch.utils.data.DataLoader(
            utils.NumpyDatasets(
                input_1=np.random.rand(10, 100),
            ),
            batch_size=10,
            shuffle=False
        ),
        'valid': torch.utils.data.DataLoader(
            utils.NumpyDatasets(
                input_1=np.random.rand(10, 100),
            ),
            batch_size=10,
            shuffle=False
        )
    }
    return datasets


class CallbackCollectOutput(trw.train.Callback):
    """
    Export random samples from our datasets

    Just for sanity check, it is always a good idea to make sure our data is loaded and processed
    as expected.
    """
    def __init__(self):
        self.outputs = None

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        self.outputs = outputs


class ModelSimpleRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, batch):
        x = self.w * batch['input_1']
        o = trw.train.OutputRegression(output=x, target_name='output')
        return {'regression': o}


class ModelEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1), requires_grad=True)
        self.dropout = nn.Dropout()  # CAREFUL: dropout must be part of the model, else model.valid won't work!

    def forward(self, batch):
        x = batch['input_1'] + 0.0 * self.w
        x = self.dropout(x)

        return {
            'regression': trw.train.OutputRegression(x, target_name='input_1')
        }


def create_model(options):
    m = ModelSimpleRegression()
    return m


optimizer_fn = functools.partial(trw.train.create_sgd_optimizers_fn, learning_rate=0.11)

def create_trainer(callback_per_epoch=[], callback_per_batch=[], callback_per_batch_loss_terms=[]):
    return trw.train.Trainer(
        callbacks_per_epoch_fn=lambda : callback_per_epoch,
        callbacks_per_batch_loss_terms_fn=lambda: callback_per_batch_loss_terms,
        callbacks_post_training_fn=None,
        callbacks_per_batch_fn=lambda  : callback_per_batch
    )

class TestTrainer(TestCase):
    def test_simple_regression(self):
        # the simples API test for model fitting
        options = trw.train.create_default_options(num_epochs=200)
        trainer = create_trainer()
        model, results = trainer.fit(
            options,
            inputs_fn=create_simple_regression,
            model_fn=create_model,
            optimizers_fn=optimizer_fn,
            eval_every_X_epoch=2)

        coef_found = trw.train.to_value(list(model.parameters())[0])
        self.assertAlmostEqual(coef_found, 2.0, delta=1e-3)

    def test_model_mode(self):
        # layers may have different behaviour whether they are in train or eval mode (e.g., dropout, batchnorm)
        # make sure the mode is correctly set for the different phases of train and evaluation
        # in this test, capture the dropout layer applied on an input. In `train` mode we expect random drop
        # of the output while in test mode we should get the identity
        options = trw.train.create_default_options(num_epochs=1)
        callback = CallbackCollectOutput()
        trainer = create_trainer(callback_per_epoch=[callback])

        model, results = trainer.fit(
            options,
            inputs_fn=create_random_input,
            model_fn=lambda _: ModelEmbedding(),
            optimizers_fn=optimizer_fn)
        # do NOT use the results: final iteration is performed in eval mode

        # expected the same
        truth = results['outputs']['simple']['valid']['regression']['output_truth']
        found = results['outputs']['simple']['valid']['regression']['output']
        diff = np.abs(truth - found)
        self.assertTrue(np.max(diff) == 0.0)

        # expected same (result is done in in eval mode)
        truth = results['outputs']['simple']['train']['regression']['output_truth']
        found = results['outputs']['simple']['train']['regression']['output']
        diff = np.abs(truth - found)
        self.assertTrue(np.max(diff) == 0.0)

        truth = callback.outputs['simple']['valid']['regression']['output_truth']
        found = callback.outputs['simple']['valid']['regression']['output']
        diff = np.abs(truth - found)
        self.assertTrue(np.max(diff) == 0.0)

        # this should be done in `train` mode
        truth = callback.outputs['simple']['train']['regression']['output_truth']
        found = callback.outputs['simple']['train']['regression']['output']
        diff = np.abs(truth - found)
        self.assertTrue(np.max(diff) > 0.5)

    def test_batch_and_loss_terms_callbacks(self):
        # make sure the per batch callback is called prior to model.forward call and
        # per batch loss terms is called after model.forward
        all_batches = []
        all_loss_terms = []

        def callback_batch(dataset_name, split_name, batch):
            batch['test_name'] = 'test_value'

        def callback_batch_loss_terms(dataset_name, split_name, batch, loss_terms):
            all_batches.append(batch)
            all_loss_terms.append(loss_terms)

        options = trw.train.create_default_options(num_epochs=1)
        trainer = create_trainer(callback_per_batch=[callback_batch], callback_per_batch_loss_terms=[callback_batch_loss_terms])

        _, _ = trainer.fit(
            options,
            inputs_fn=create_random_input,
            model_fn=lambda _: ModelEmbedding(),
            optimizers_fn=optimizer_fn)

        self.assertTrue(len(all_batches) == 4)
        self.assertTrue(len(all_loss_terms) == 4)

        for batch in all_batches:
            self.assertTrue(batch['test_name'] == 'test_value')

        for loss_terms in all_loss_terms:
            self.assertTrue('regression' in loss_terms)
