from unittest import TestCase
import trw.train
import numpy as np
import torch


class TestOutput(TestCase):
    def test_regression_0_loss(self):
        input_values = torch.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=float))
        target_values = torch.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=float))

        o = trw.train.OutputRegression(input_values, target_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss == 0.0)


    def test_regression_error(self):
        input_values = torch.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=float))
        target_values = torch.from_numpy(np.asarray([[0, 0], [0, 0]], dtype=float))

        o = trw.train.OutputRegression(input_values, target_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss == (1+2*2+3*3+4*4) / 4)

    def test_regression_scaling(self):
        input_values = torch.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=float))
        target_values = torch.from_numpy(np.asarray([[0, 0], [0, 0]], dtype=float))

        o = trw.train.OutputRegression(input_values, target_name='target', loss_scaling=2.0)
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss == (1+2*2+3*3+4*4) / 4 * 2)

    def test_regression_weight(self):
        input_values = torch.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=float))
        target_values = torch.from_numpy(np.asarray([[0, 0], [0, 0]], dtype=float))
        weights = torch.from_numpy(np.asarray([1, 0], dtype=float))

        o = trw.train.OutputRegression(input_values, target_name='target', weight_name='weights')
        batch = {'target': target_values, 'weights': weights}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss == (1+2*2+3*3*0+4*4*0) / 4 * 2)

    def test_classification_0_loss(self):
        input_values = torch.from_numpy(np.asarray([[0.0, 100.0], [100.0, 0.0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 0], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss < 1e-6)

        assert r.get('output') is not None
        self.assertTrue((r['output'] == [1, 0]).all())

    def test_classification_weight(self):
        input_values = torch.from_numpy(np.asarray([[0.0, 100.0], [100.0, 0.0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 1], dtype=np.int64))
        weights = torch.from_numpy(np.asarray([1, 0], dtype=float))

        o = trw.train.OutputClassification(input_values, classes_name='target', weight_name='weights')
        batch = {'target': target_values, 'weights': weights}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss < 1e-6)
