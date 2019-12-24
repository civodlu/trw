from unittest import TestCase
import trw.train
import torch
import torch.nn as nn
import functools


class ModelDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Linear(10, 1)

    def forward(self, batch):
        x = batch['x']
        y = self.d1(x)
        return {
            'regression': trw.train.OutputRegression(y, target_name='y')
        }


def create_datasets():
    batch = {
        'x': torch.zeros([20, 10], dtype=torch.float32),
        'y': torch.zeros([20], dtype=torch.float32)
    }
    sampler = trw.train.SamplerSequential(batch_size=10)
    split = trw.train.SequenceArray(batch, sampler=sampler)
    return {
        'dataset1': {
            'train': split
        }
    }


class TestCallbackLearningRateRecorder(TestCase):
    def test_model_fitting_with_lr_scheduler(self):
        lr_recorder = trw.train.CallbackLearningRateRecorder()

        trainer = trw.train.Trainer(
            callbacks_per_epoch_fn=lambda: [lr_recorder],
            callbacks_pre_training_fn=None,
            callbacks_post_training_fn=None,
            trainer_callbacks_per_batch=None
        )

        optimizer_fn = functools.partial(
            trw.train.create_adam_optimizers_scheduler_step_lr_fn,
            learning_rate=1000,
            step_size=10,
            gamma=0.1
        )

        options = trw.train.create_default_options(num_epochs=50)
        r = trainer.fit(
            options,
            inputs_fn=create_datasets,
            model_fn=lambda options: ModelDense(),
            optimizers_fn=optimizer_fn
        )

        lrs = [value for e, value in lr_recorder.lr_optimizers['dataset1']]
        assert len(lrs) == 50

        expected_lr = 1000
        for e, lr in enumerate(lrs):
            if (e + 1) % 10 == 0 and e > 0:
                expected_lr /= 10.0
            assert abs(lr - expected_lr) < 1e-5
