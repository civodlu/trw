from unittest import TestCase
import trw.hparams
import trw.train
import numpy as np
import os


class TestParamsOptimizer(TestCase):
    def test_search(self):
        #
        # very simple test just to test the API: minimize x, y given the loss l(x, y) = x^2 + y^2. Best params: x, y = (0, 0)
        # TODO: this is not assessing the perforamce of hyperband vs random search, but we should do that too!
        #

        def evaluate_hparams(hparams, iterations):
            x = hparams.create('param_x', trw.hparams.ContinuousUniform(5, -10, 10))
            y = hparams.create('param_y', trw.hparams.ContinuousUniform(5, -10, 10))
            d = hparams.create('param_d', trw.hparams.ContinuousUniform(5, -10, 10))

            return x * x + y * y + d, {'param_x': x, 'param_y': y, 'param_d': d}
            #return  5 * x * x +  2 * y * y + d * d, {'param_x': x, 'param_y': y, 'param_d': d}
            #return x * y  +  0.1 * d, {'param_x': x, 'param_y': y, 'param_d': d}
            #return x + y + d, {'param_x': x, 'param_y': y, 'param_d': d}

        optimizer = trw.hparams.params_optimizer_hyperband.HyperParametersOptimizerHyperband(
            evaluate_hparams_fn=evaluate_hparams,
            max_iter=81,
            repeat=20)

        options = trw.train.create_default_options()
        tmp = os.path.join(options['workflow_options']['logging_directory'], 'hyperband_test')
        trw.train.create_or_recreate_folder(tmp)
        tries = optimizer.optimize(result_path=tmp)

        values = [t[0] for t in tries]
        best_try = tries[np.argmin(np.abs(values))]
        print(str(best_try))
        self.assertTrue(best_try[0] < 1.0)
