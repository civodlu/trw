from unittest import TestCase
import trw
import os
from . import utils
import numpy as np


class TestPlots(TestCase):
    def test_history(self):
        # create 2 groups, with 1 group has 2 histories. Check this is displayed properly
        data = {
            'train': [[(0, 1.0), (1, 1.5), (2, 2.5),(3, 3.5),],
                      [(0, 1.1), (1, 1.6), (2, 2.6), (3, 3.6), ]],
            'valid': [[(0, 0.5), (1, 1.0), (3, 2.0), ]]
        }

        trw.train.plot_group_histories(utils.root_output, data, xlabel='epochs', ylabel='values', title='Values by epochs')
        assert os.path.exists(os.path.join(utils.root_output, 'Values by epochs.png'))

    def test_boxplots(self):
        data = {
            'feature_1': [0.8, 0.9, 0.85, 0.95],
            'feature_2': [0.7, 0.71, 0.72, 0.73]
        }

        trw.train.boxplots(utils.root_output, features_trials=data, xlabel='Features', ylabel='loss', title='Loss by features', meanline=True, y_range=[0.65, 1.0], rotate_x=45, showfliers=True)
        assert os.path.exists(os.path.join(utils.root_output, 'Loss by features.png'))

    def test_plot_roc(self):
        d = 100
        trues_1 = np.asarray([0] * d + [1] * d)
        trues_2 = np.asarray([0] * d + [1] * d)
        trues = [trues_1, trues_2]


        values_1 = np.random.rand(d * 2)
        values_2 = np.arange(0, d * 2, dtype=np.float) / (d * 2)
        values = [values_1, values_2]
        trw.train.plot_roc(utils.root_output, trues, values, 'ROC classifier 1 vs classifier 2', label_name=['Random classifier', 'Perfect Classifier'])
