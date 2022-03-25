import collections
import unittest
from functools import partial

import torch
import trw
import numpy as np
from torch import nn
from trw.callbacks import Callback
from trw.train import Options
from trw.utils import torch_requires


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1, bias=False)

    def forward(self, i):
        c = i['input1'].float()
        o = self.layer(c)
        loss = (o - c).abs()
        print(self.layer.weight)
        return {
            'Loss': trw.train.OutputLoss(loss)
        }


class CallbackLearningRateRecorderPerStep(Callback):
    """
    Record the learning rate of the optimizers.

    This is useful as a debugging tool.
    """
    def __init__(self):
        self.lr_optimizers = collections.defaultdict(list)

    # options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs
    def __call__(self, dataset_name, **kwargs):
        optimizer = kwargs.get('optimizer')
        if optimizer is not None:
            lr = optimizer.param_groups[0].get('lr')
            self.lr_optimizers[dataset_name].append(lr)


class TestOptimizerV2(unittest.TestCase):
    def test_sgd_step_lr(self):
        a = np.arange(0, 10)
        datasets = {
            'dataset1': {
                'train': trw.train.SequenceArray({'input1': a})
            }
        }
        num_epochs = 90

        model = Model()
        options = Options(device=torch.device('cpu'), num_epochs=num_epochs)
        callback_per_step = CallbackLearningRateRecorderPerStep()
        trainer = trw.train.TrainerV2(
            callbacks_per_batch_loss_terms=[callback_per_step],
            callbacks_pre_training=None,
            callbacks_post_training=None,
            callbacks_per_epoch=None
        )

        optimizers = trw.train.OptimizerSGD(learning_rate=0.1).scheduler_step_lr(step_size=30, gamma=0.1)
        trainer.fit(options, datasets, model, optimizers)

        lrs = np.asarray(callback_per_step.lr_optimizers['dataset1'])
        assert len(lrs) == num_epochs * len(a)
        assert (abs(lrs[0:10 * 30] - 0.1) < 1e-5).all()
        assert (abs(lrs[10 * 30:2 * 10 * 30] - 0.01) < 1e-5).all()
        assert (abs(lrs[2 * 10 * 30: ] - 0.001) < 1e-5).all()


if __name__ == '__main__':
    unittest.main()