import os
import tempfile
import unittest
import trw
from trw.utils import ExceptionAbortRun
from trw.callbacks import CallbackEarlyStopping
from trw.hparams import HyperParameters, RunResult
from trw.train import create_default_options


class TestCallbackEarlyStopping(unittest.TestCase):
    def test_no_early_stop(self):
        nb_epochs = 100
        nb_runs = 20
        tmp = tempfile.mkdtemp()

        store_path = os.path.join(tmp, 'store.pkl')
        store = trw.hparams.RunStoreFile(store_path)
        hparams = HyperParameters()

        for r in range(nb_runs):
            history = [{'1-accuracy': 1.0} for n in range(nb_epochs)]
            store.save_run(RunResult(metrics={}, hyper_parameters=hparams, history=history, info=None))

        history = [{'1-accuracy': 0.9999} for n in range(nb_epochs)]
        early_stopping = CallbackEarlyStopping(store=store, loss_fn=lambda hstep: hstep['1-accuracy'])

        # we are better, do NOT raise `ExceptionAbortRun`
        options = create_default_options(num_epochs=nb_epochs)
        early_stopping(options, history[:10], None)
        early_stopping(options, history[:25], None)
        early_stopping(options, history[:50], None)
        early_stopping(options, history[:75], None)

        expected_checkpoints = {
            10: 1.0,
            25: 1.0,
            50: 1.0,
            75: 1.0,
        }
        assert early_stopping.max_loss_by_epoch == expected_checkpoints

    def test_early_stop(self):
        nb_epochs = 100
        nb_runs = 20
        tmp = tempfile.mkdtemp()

        store_path = os.path.join(tmp, 'store.pkl')
        store = trw.hparams.RunStoreFile(store_path)
        hparams = HyperParameters()

        for r in range(nb_runs):
            history = [{'1-accuracy': 0.8} for n in range(nb_epochs)]
            store.save_run(RunResult(metrics={}, hyper_parameters=hparams, history=history, info=None))

        history = [{'1-accuracy': 0.9999} for n in range(nb_epochs)]
        early_stopping = CallbackEarlyStopping(store=store, loss_fn=lambda hstep: hstep['1-accuracy'])

        # we are better, do NOT raise `ExceptionAbortRun`
        aborted_run = False
        try:
            options = create_default_options(num_epochs=nb_epochs)
            early_stopping(options, history[:10], None)
        except ExceptionAbortRun:
            aborted_run = True

        assert aborted_run, 'run should have been aborted!'

        expected_checkpoints = {
            10: 0.8,
            25: 0.8,
            50: 0.8,
            75: 0.8,
        }
        assert early_stopping.max_loss_by_epoch == expected_checkpoints

    def test_notenough_runs_stop(self):
        nb_epochs = 100
        nb_runs = 2
        tmp = tempfile.mkdtemp()

        store_path = os.path.join(tmp, 'store.pkl')
        store = trw.hparams.RunStoreFile(store_path)
        hparams = HyperParameters()

        for r in range(nb_runs):
            history = [{'1-accuracy': 0.8} for n in range(nb_epochs)]
            store.save_run(RunResult(metrics={}, hyper_parameters=hparams, history=history, info=None))

        history = [{'1-accuracy': 0.9999} for n in range(nb_epochs)]
        early_stopping = CallbackEarlyStopping(store=store, loss_fn=lambda hstep: hstep['1-accuracy'])

        # we are better, do NOT raise `ExceptionAbortRun`
        aborted_run = False
        try:
            options = create_default_options(num_epochs=nb_epochs)
            early_stopping(options, history[:10], None)
        except ExceptionAbortRun:
            aborted_run = True

        assert not aborted_run, 'run should NOT have been aborted!'

        expected_checkpoints = {
            10: None,
            25: None,
            50: None,
            75: None,
        }
        assert early_stopping.max_loss_by_epoch == expected_checkpoints

    def test_early_termination(self):
        nb_epochs = 2
        nb_runs = 1
        tmp = tempfile.mkdtemp()

        store_path = os.path.join(tmp, 'store.pkl')
        store = trw.hparams.RunStoreFile(store_path)
        hparams = HyperParameters()

        for r in range(nb_runs):
            history = [{'1-accuracy': 0.99} for n in range(nb_epochs)]
            store.save_run(RunResult(metrics={}, hyper_parameters=hparams, history=history, info=None))

        history = [{'1-accuracy': 0.9999} for n in range(nb_epochs)]

        def check_min_loss(loss, history):
            return loss >= 0.9

        early_stopping = CallbackEarlyStopping(
            store=store,
            loss_fn=lambda hstep: hstep['1-accuracy'],
            raise_stop_fn=check_min_loss
        )

        # we are better, do NOT raise `ExceptionAbortRun`
        aborted_run = False
        try:
            options = create_default_options(num_epochs=nb_epochs)
            early_stopping(options, history[:10], None)
        except ExceptionAbortRun as e:
            aborted_run = True
            assert 'Early termination' in e.reason

        assert aborted_run, 'run should NOT have been aborted!'


if __name__ == '__main__':
    unittest.main()
