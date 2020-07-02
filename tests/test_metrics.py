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

        history = trw.train.trainer.aggregate_list_of_metrics([history])
        assert history['classification error'] == 0.0
        assert history['loss'] == 0.0

    def test_classification_accuracy_33(self):
        input_values = torch.from_numpy(np.asarray([[0.0, 100.0], [100.0, 0.0], [100.0, 0.0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 0, 1], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)
        history = r['metrics_results']

        history = trw.train.trainer.aggregate_list_of_metrics([history])
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

        history = trw.train.trainer.aggregate_list_of_metrics([history])
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

        history = trw.train.trainer.aggregate_list_of_metrics([history])
        assert abs(history['classification error'] - 1.0 / 4) < 1e-4
        assert abs(history['1-specificity'] - (1 - 1.0)) < 1e-4
        assert abs(history['1-sensitivity'] - (1 - 1.0 / 2)) < 1e-4

    def test_metrics_sensitivity_specificity_perfect(self):
        input_values = torch.from_numpy(np.asarray([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([0, 0, 0, 0], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)
        history = r['metrics_results']

        history = trw.train.trainer.aggregate_list_of_metrics([history])
        assert abs(history['classification error']) < 1e-4
        assert abs(history['1-specificity']) < 1e-4
        assert history['1-sensitivity'] is None

    def test_metrics_sensitivity_specificity_all_wrong(self):
        input_values = torch.from_numpy(np.asarray([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 1, 1, 1], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)
        history = r['metrics_results']

        history = trw.train.trainer.aggregate_list_of_metrics([history])
        assert abs(history['classification error'] - 1.0) < 1e-4
        assert history['1-specificity'] is None
        assert abs(history['1-sensitivity'] - 1.0) < 1e-4

    def test_metrics_sensitivity_specificity_all_wrong_specificity_none(self):
        input_values = torch.from_numpy(np.asarray([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 1, 1, 1], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)
        history = r['metrics_results']

        history = trw.train.trainer.aggregate_list_of_metrics([history])
        assert abs(history['classification error'] - 1.0) < 1e-4
        assert history['1-specificity'] is None
        assert abs(history['1-sensitivity'] - 1.0) < 1e-4

    def test_metrics_sensitivity_specificity_all_wrong_sensitivity_none(self):
        input_values = torch.from_numpy(np.asarray([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=float))
        target_values = torch.from_numpy(np.asarray([0, 0, 0, 0], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)
        history = r['metrics_results']

        history = trw.train.trainer.aggregate_list_of_metrics([history])
        assert abs(history['classification error'] - 1.0) < 1e-4
        assert history['1-sensitivity'] is None
        assert abs(history['1-specificity'] - 1.0) < 1e-4

    def test_metrics_with_none_aggregated(self):
        input_values = torch.from_numpy(np.asarray([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=float))
        target_values = torch.from_numpy(np.asarray([0, 0, 0, 0], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r1 = o.evaluate_batch(batch, False)

        input_values = torch.from_numpy(np.asarray([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 1, 1, 1], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r2 = o.evaluate_batch(batch, False)

        r = trw.train.trainer.aggregate_list_of_metrics([r1['metrics_results'], r2['metrics_results']])

        # make sure we can aggregate appropriately the metrics, even if there is a `None` value
        assert abs(r['classification error'] - 1.0) < 1e-4
        assert abs(r['1-sensitivity'] - 1.0) < 1e-4
        assert abs(r['1-specificity'] - 1.0) < 1e-4

    def test_auc_random(self):
        # completely random: the AUC should be 0.5
        nb_samples = 100000
        input_values = torch.from_numpy(np.random.choice([0, 1], size=[nb_samples]))

        r = np.random.uniform(0, 1, size=[nb_samples])
        target = np.zeros([nb_samples, 2], dtype=np.float)
        target[:, 0] = r
        target[:, 1] = 1 - r
        target_values = torch.from_numpy(target)

        metric = trw.train.MetricClassificationAUC()
        auc = metric({'output_truth': input_values, 'output_raw': target_values})
        auc = metric.aggregate_metrics([auc])
        assert abs(auc['1-auc'] - 0.5) < 0.1
        print('DONE')

    def test_auc_perfect(self):
        # perfect classification: AUC should be 1.0 (so metric should be 0.0)
        nb_samples = 100

        r = np.random.uniform(0, 1, size=[nb_samples])
        target = np.zeros([nb_samples, 2], dtype=np.float)
        target[:, 0] = r
        target[:, 1] = 1 - r
        target_values = torch.from_numpy(target)

        input_values = (1 - r > 0.5).astype(int)

        metric = trw.train.MetricClassificationAUC()
        auc = metric({'output_truth': input_values, 'output_raw': target_values})
        auc = metric.aggregate_metrics([auc])
        assert abs(auc['1-auc'] - 0.0) < 0.001
