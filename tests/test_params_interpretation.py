import matplotlib
matplotlib.use('Agg')  # change the backend. Issues with CI on windows

from unittest import TestCase
import trw.hparams
import trw.train
import os
import math


def create_data(location_base, loss_generator, nb_samples):
    for n in range(nb_samples):
        file = os.path.join(location_base, 'hparam-%d.pkl' %n)
        loss, params = loss_generator()
        trw.hparams.store_loss_params(file, loss, None, params)


class TestParamsInterpretation(TestCase):
    """
    Here, generate "fake" variable dependencies on the loss function and make sure we can recover them accurately and
    if not, at least understand the possible caveats.

    For now:
    - the ordering of hyper parameter is accurate,

    But:
    - the importance value can be over-estimated (for the best hparam and under-estimated for others)
    - the variation plot can be useful to determine the influence on the loss (width & slope)
    - covariance may be useful only between most important hparam
    """
    def test_independent_params(self):
        """
        Test independent hparams contribution to loss function.

        This is the simplest case: independent, same range, positive only
        """
        options = trw.train.options.create_default_options()
        tmp_path = os.path.join(options['workflow_options']['logging_directory'], 'test_independent_params')
        trw.train.create_or_recreate_folder(tmp_path)

        nb_samples = 500

        def generator():
            hparams = trw.hparams.HyperParameters()
            hparams.create('x', trw.hparams.ContinuousUniform(0, 0, 5))
            hparams.create('y', trw.hparams.ContinuousUniform(0, 0, 5))
            hparams.create('z', trw.hparams.ContinuousUniform(0, 0, 5))
            hparams.create('w', trw.hparams.ContinuousUniform(0, 0, 5))
            hparams.generate_random_hparams()
            return hparams.hparams['x'].current_value * 3 + hparams.hparams['y'].current_value * 2 + hparams.hparams['z'].current_value + 0.01 * hparams.hparams['w'].current_value, hparams

        create_data(tmp_path, generator, nb_samples)

        r = trw.hparams.analyse_hyperparameters(os.path.join(tmp_path, 'hparam-*'),
                                                tmp_path,
                                                params_forest_max_features_ratio=0.6,
                                                params_forest_n_estimators=1000,
                                                create_graphs=True)

        r = dict(zip(r['sorted_param_names'], r['sorted_importances']))
        self.assertTrue(r['x'] > r['y'])
        self.assertTrue(r['y'] > r['z'])
        self.assertTrue(r['z'] > r['w'])

    def test_correlated_params(self):
        """
        Test correlated hparams contribution to loss function.

        In standard random forest, this is an issue: since the 2 variables are correlated, knowing one will not give more information
        so when building the tree, it will not decrease the MI and so one will arbitrarily be seen as having more info.

        The correlated parameters should have a similar importance
        """
        options = trw.train.options.create_default_options()
        tmp_path = os.path.join(options['workflow_options']['logging_directory'], 'test_correlated_params')
        trw.train.create_or_recreate_folder(tmp_path)

        nb_samples = 500

        def generator():
            hparams = trw.hparams.HyperParameters()
            p = trw.hparams.ContinuousUniform(0, 0, 5)
            hparams.create('x', p)
            hparams.create('y', p)
            hparams.create('z', trw.hparams.ContinuousUniform(0, 0, 5))
            hparams.generate_random_hparams()
            return hparams.hparams['x'].current_value * 2 + hparams.hparams['y'].current_value * 2 + hparams.hparams['z'].current_value, hparams

        create_data(tmp_path, generator, nb_samples)

        r = trw.hparams.interpret_params.analyse_hyperparameters(os.path.join(tmp_path, 'hparam-*'),
                                                                 tmp_path,
                                                                 params_forest_max_features_ratio=0.6,
                                                                 params_forest_n_estimators=1000,
                                                                 create_graphs=True)

        r = dict(zip(r['sorted_param_names'], r['sorted_importances']))
        self.assertTrue(math.fabs(r['x'] - r['y']) < 0.05)
        self.assertTrue(r['x'] > r['z'])

    def test_independent_different_ranges(self):
        """
        Test independent hparams contribution to loss function.

        This is the simplest case: independent, same range, positive only
        """
        options = trw.train.options.create_default_options()
        tmp_path = os.path.join(options['workflow_options']['logging_directory'], 'test_independent_different_ranges')
        trw.train.create_or_recreate_folder(tmp_path)

        nb_samples = 1000

        def generator():
            hparams = trw.hparams.HyperParameters()
            hparams.create('x', trw.hparams.ContinuousUniform(0, 0, 15))
            hparams.create('y', trw.hparams.ContinuousUniform(0, 0, 5))
            hparams.create('z', trw.hparams.ContinuousUniform(0, 0, 1))
            hparams.create('w', trw.hparams.ContinuousUniform(0, 0, 0.1))
            hparams.generate_random_hparams()
            return hparams.hparams['x'].current_value + hparams.hparams['y'].current_value + hparams.hparams['z'].current_value + hparams.hparams['w'].current_value, hparams

        create_data(tmp_path, generator, nb_samples)

        r = trw.hparams.interpret_params.analyse_hyperparameters(os.path.join(tmp_path, 'hparam-*'),
                                                                 tmp_path,
                                                                 params_forest_max_features_ratio=0.6,
                                                                 params_forest_n_estimators=1000,
                                                                 create_graphs=True)

        r = dict(zip(r['sorted_param_names'], r['sorted_importances']))
        self.assertTrue(r['x'] > r['y'])
        self.assertTrue(r['y'] > r['z'])
        self.assertTrue(r['z'] > r['w'])


