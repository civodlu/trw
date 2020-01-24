from unittest import TestCase
import trw.train
import numpy as np
import torch


class TestMetrics(TestCase):
    def test_classification_accuracy_100(self):
        input_values = torch.from_numpy(np.asarray([[0.0, 100.0], [100.0, 0.0], [100.0, 0.0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 0, 0], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)
        history = r['metrics_results']

        assert history['classification error'] == 0.0
        assert history['loss'] == 0.0

    def test_classification_accuracy_33(self):
        input_values = torch.from_numpy(np.asarray([[0.0, 100.0], [100.0, 0.0], [100.0, 0.0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 0, 1], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)
        history = r['metrics_results']

        assert abs(history['classification error'] - 0.33333) < 1e-4
        assert abs(history['loss'] - 0.33333 * 100) < 1e-2

    def test_classification_sensitivity_1_specificity_0(self):
        #                         Confusion Matrix
        #  -----------------------------------------------------------------
        #  |                | found 0 = no effect  |     found 1 = effect  |
        #  -----------------------------------------------------------------
        #  | no effect = 0  |      TN 2            |        FP 1           |
        #  -----------------------------------------------------------------
        #  | effect = 1     |      FN 0            |        TP 1           |
        #  -----------------------------------------------------------------
        #  Sensitivity = TP / (TP + FN)
        #  Specificity = TN / (TN + FP)
        #
        input_values = torch.from_numpy(np.asarray([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1,      0,      0,      0], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)
        history = r['metrics_results']

        assert abs(history['classification error'] - 1.0 / (4)) < 1e-4
        assert abs(history['1-sensitivity'] - (1 - 1.0)) < 1e-4
        assert abs(history['1-specificity'] - (1 - 2.0 / 3)) < 1e-4

    def test_classification_sensitivity_0_specificity_1(self):
        #                         Confusion Matrix
        #  -----------------------------------------------------------------
        #  |                | found 0 = no effect  |     found 1 = effect  |
        #  -----------------------------------------------------------------
        #  | no effect = 0  |      TN 2            |        FP 0           |
        #  -----------------------------------------------------------------
        #  | effect = 1     |      FN 1            |        TP 1           |
        #  -----------------------------------------------------------------
        #  Sensitivity = TP / (TP + FN)
        #  Specificity = TN / (TN + FP)
        #
        input_values = torch.from_numpy(np.asarray([[0, 1], [1, 0], [1, 0], [1, 0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1,      0,      0,      1], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)
        history = r['metrics_results']

        assert abs(history['classification error'] - 1.0 / 4) < 1e-4
        assert abs(history['1-specificity'] - (1 - 1.0)) < 1e-4
        assert abs(history['1-sensitivity'] - (1 - 1.0 / 2)) < 1e-4
