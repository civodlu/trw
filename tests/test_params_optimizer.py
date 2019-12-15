from unittest import TestCase
import trw.hparams
import trw.train
import numpy as np
import functools
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


def log_nothing(str):
    pass


optimizer_fn = functools.partial(trw.train.create_sgd_optimizers_fn, learning_rate=0.11)


class Model_XOR(nn.Module):
    """
    XOR problem with hyper-parameter search.

    The only hyper-parameter is the use or not of a non-linear. We know that linear classifier can't classify
    XOR, only the non-linear can. Make sure we can find the expected hyper-parameter
    """
    def __init__(self, hparams):
        super(Model_XOR, self).__init__()
        self.hparams = hparams
        with_dense = hparams.create('with_dense', trw.hparams.DiscreteBoolean(False))

        self.dense1 = nn.Linear(2, 4)
        self.dense2 = None
        if with_dense:
            self.dense2 = nn.Linear(4, 4)
        self.output = nn.Linear(4, 1)


    def forward(self, batch):
        x = self.dense1(batch['var_x1'])
        if self.dense2 is not None:
            x = F.relu(self.dense2(x))
        x = self.output(x)
        o = trw.train.OutputRegression(output=x, target_name='var_y')
        return {'regression': o}


class TestParamsOptimizer(TestCase):
    def test_discrete_values(self):
        params = trw.hparams.HyperParameters()

        choices = ['choice_1', 'choice_2', 'choice_3']
        params.create('choice', trw.hparams.DiscreteValue(choices, 'choice_1'))

        ps = []
        nb_tests = 10000
        for n in range(nb_tests):
            params.generate_random_hparams()
            p = params.create('choice', trw.hparams.DiscreteValue(choices, 'choice_1'))
            assert p in choices
            ps.append(p)

        ps_counts = collections.Counter(ps)
        for choice in choices:
            self.assertTrue(choice in ps_counts)
            self.assertTrue(ps_counts[choice] >= 0.9 * nb_tests / len(choices))

    def test_random_search_single(self):
        #
        # very simple test: minimize x, y given the loss l(x, y) = x^2 + y^2. Best params: x, y = (0, 0)
        #

        def discrete_mapping(value):
            return trw.hparams.DiscreteMapping(
                [
                    ('value_0', 1000),
                    ('value_1', 800),
                    ('value_2', 100),
                    ('value_3', 0)
                ], value)
        
        def score(x, y, d):
            return x * x + y * y + d

        def evaluate_hparams(hparams):
            x = hparams.create('param_x', trw.hparams.ContinuousUniform(5, -10, 10))
            y = hparams.create('param_y', trw.hparams.ContinuousUniform(5, -10, 10))
            d = hparams.create('param_d', discrete_mapping('value_0'))

            return score(x, y, d), {'param_x': x, 'param_y': y, 'param_d': d}

        optimizer = trw.hparams.HyperParametersOptimizerRandomSearchLocal(
            evaluate_hparams_fn=evaluate_hparams,
            log_string=log_nothing,
            repeat=10000)

        tries = optimizer.optimize(result_path=None)

        values = [t[0] for t in tries]
        best_try = tries[np.argmin(values)]
        print('BEST=', str(best_try))
        self.assertTrue(best_try[0] < 1.0)
        
        # make sure the returned score is the same as if we
        # calculate it directly from the best parameters
        for value, params, hparams in tries[:10]:
            assert params['param_x'] == hparams.hparams['param_x'].current_value
            assert params['param_y'] == hparams.hparams['param_y'].current_value
            assert params['param_d'] == hparams.hparams['param_d'].kvp[hparams.hparams['param_d'].current_value]
            assert score(params['param_x'], params['param_y'], params['param_d']) == value
            

    def test_xor(self):
        #
        # test simple hparam optimization
        #
        def inputs_fn():
            datasets = collections.OrderedDict()
            datasets['dataset_1'] = {
                'train': torch.utils.data.DataLoader(
                    utils.NumpyDatasets(
                        var_x1=np.asarray([[-1, -1], [1, -1], [-1, 1], [1, 1]]),
                        var_y=np.asarray([0, 1, 1, 0])
                    ),
                    batch_size=100
                )
            }
            return datasets

        def model_fn(options):
            hparams = options['model_parameters']['hyperparams']
            return Model_XOR(hparams)

        def evaluate_hparams(options, hparams):
            options['model_parameters']['hyperparams'] = hparams

            trainer = trw.train.Trainer(
                callbacks_per_epoch_fn=None,
                callbacks_pre_training_fn=None,
                callbacks_post_training_fn=None
            )
            model, output = trainer.fit(
                options,
                inputs_fn=inputs_fn,
                model_fn=model_fn,
                optimizers_fn=optimizer_fn,
                run_prefix=prefix)
            loss = output['outputs']['dataset_1']['train']['regression']['loss']
            return trw.train.to_value(loss), 'no report'


        prefix = 'hparams'
        options = trw.train.create_default_options(num_epochs=1000)
        optimizer = trw.hparams.params_optimizer_random_search.HyperParametersOptimizerRandomSearchLocal(
            evaluate_hparams_fn=functools.partial(evaluate_hparams, options),
            log_string=log_nothing,
            result_prefix=prefix,
            repeat=10)

        tries = optimizer.optimize(result_path=options['workflow_options']['logging_directory'])

        values = [t[0] for t in tries]
        best_try = tries[np.argmin(values)]
        print(str(best_try))
        self.assertTrue(len(tries) == 10)
        self.assertTrue(best_try[0] < 1e-5)
