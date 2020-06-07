from unittest import TestCase
import trw
import torch
import torch.nn as nn
from trw.train.callback_reporting_model_statistics import calculate_stats_gradient


class ModelLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(1, 1)

        with torch.no_grad():
            self.w.weight.fill_(1.5)
            self.w.bias.fill_(0)

    def forward(self, x):
        return self.w(x)


class ModelLinearRegressionShell(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = ModelLinearRegression()

    def forward(self, batch):
        x = batch['x']
        o = self.w(x)
        return {
            'regression': trw.train.OutputRegression(o, target_name='y')
        }


class TestReportingModelStatistics(TestCase):
    def test_stats(self):
        # test expected activation and gradient statistics of nested modules
        sampler = trw.train.SamplerSequential(batch_size=1)
        sequence = trw.train.SequenceArray({
            'x': torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32).view((-1, 1)),
            'y': torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32).view((-1, 1)),
        }, sampler=sampler)
        model = ModelLinearRegressionShell()

        gradient_stats, output_stats = calculate_stats_gradient(model, sequence, 4, modules_type_to_trace=None)
        assert len(gradient_stats) == 2  # we have 2 parameters

        # we can directly calculate the gradients to be found:
        # dE/dw = d/dw (y-x * w + b) ** 2
        #       = d/dw ((y+b)**2 - 2 * w * b * (y + b) + (x * w) ** 2)
        #       = 0 - 2 * b * (y + b) + 2 * w * x ** 2
        # and substitute the data in the equation, we can calculate exactly the gradient relative to w
        assert gradient_stats['ModelLinearRegressionShell/ModelLinearRegression/Linear/weight']['min'] == 0
        assert gradient_stats['ModelLinearRegressionShell/ModelLinearRegression/Linear/weight']['max'] == 9
        assert gradient_stats['ModelLinearRegressionShell/ModelLinearRegression/Linear/weight']['mean'] == (9 + 4 + 1 + 0) / 4
        assert gradient_stats['ModelLinearRegressionShell/ModelLinearRegression/Linear/weight']['norm2'] == 3.5
        assert gradient_stats['ModelLinearRegressionShell/ModelLinearRegression/Linear/weight']['nb_items'] == 4

        # we can do the same for the bias
        assert gradient_stats['ModelLinearRegressionShell/ModelLinearRegression/Linear/bias']['min'] == 0
        assert gradient_stats['ModelLinearRegressionShell/ModelLinearRegression/Linear/bias']['max'] == 3
        assert gradient_stats['ModelLinearRegressionShell/ModelLinearRegression/Linear/bias']['mean'] == 1.5
        assert gradient_stats['ModelLinearRegressionShell/ModelLinearRegression/Linear/bias']['norm2'] == 1.5
        assert gradient_stats['ModelLinearRegressionShell/ModelLinearRegression/Linear/bias']['nb_items'] == 4

        # for the activation:
        # we expect mean activations = (0 + 1.5 + 3 + 4.5) / 4 == 2.25
        assert len(output_stats) == 2
        output_stat = next(iter(output_stats.values()))
        assert output_stat['min'] == 0
        assert output_stat['max'] == 4.5
        assert output_stat['mean'] == 2.25
        assert output_stat['norm2'] == 2.25
        assert output_stat['nb_items'] == 4
