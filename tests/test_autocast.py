import unittest
from functools import partial

import torch
import trw
from torch import nn

from test_trainer import create_simple_regression, create_trainer
from trw.utils import torch_requires


try:
    from torch.cuda.amp import autocast
except ModuleNotFoundError:
    # PyTorch version did not support autocast
    def do_nothing_fn():
        pass

    autocast = lambda: lambda x: do_nothing_fn()


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()

    @autocast()
    def forward(self, v):
        return v + 1


class TestModuleNoAutocast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v):
        return v + 1


class ModelSimpleRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)

    @autocast()  # must be decorated with `autocast` to enable mixed precision
    def forward(self, batch):
        x = self.w * batch['input_1']
        o = trw.train.OutputRegression(output=x, output_truth=batch['output'])
        return {'regression': o}


class TestModuleAutocastDetection(unittest.TestCase):
    @torch_requires(min_version='1.6', silent_fail=True)
    def test_module_without_autocast(self):
        m = TestModuleNoAutocast()
        assert not trw.utils.is_autocast_module_decorated(m)

    @torch_requires(min_version='1.6', silent_fail=True)
    def test_module_with_autocast(self):
        m = TestModule()
        assert trw.utils.is_autocast_module_decorated(m)

    @torch_requires(min_version='1.6', silent_fail=True)
    def test_mixed_precision_single_gpu(self):
        if torch.cuda.device_count() == 0:
            return

        datasets = create_simple_regression()
        model = ModelSimpleRegression()
        trainer = create_trainer()
        optimizer_fn = partial(trw.train.create_sgd_optimizers_fn, learning_rate=0.11)
        options = trw.train.Options(num_epochs=200, device=torch.device('cuda:0'), mixed_precision_enabled=True)

        trainer.fit(
            options=options,
            datasets=datasets,
            model=model,
            optimizers_fn=optimizer_fn,
        )

        coef_found = trw.utils.to_value(list(model.parameters())[0])

        # if failure, it means the gradient was not scaled properly
        self.assertAlmostEqual(coef_found, 2.0, delta=1e-4)


if __name__ == '__main__':
    unittest.main()
