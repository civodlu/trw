import trw
from unittest import TestCase
import numpy as np


class TestCallbackActivationStatistics(TestCase):
    def test_simple(self):
        """
        Create a simple network to capture mean, std, min, max activations and compare them
        against expected values.
        """
        def create_dataset():
            dataset = {
                'train': trw.train.SequenceArray({
                    'input': np.ones([100, 10], dtype=np.float32)
                })
            }
            return {'dataset': dataset}

        def create_model(options):
            layers = trw.simple_layers.Input(shape=[None, 10], feature_name='input')
            layers = trw.simple_layers.ShiftScale(layers, mean=0.0, standard_deviation=1.0)
            layers = trw.simple_layers.denses(layers, [2])
            output = trw.simple_layers.OutputEmbedding(layers, output_name='output')
            net = trw.simple_layers.compile_nn([output])
            return net

        log_lines = []

        def logger_fn(line):
            log_lines.append(line)

        trainer = trw.train.Trainer(
            callbacks_per_epoch_fn=lambda: [trw.train.CallbackActivationStatistics(split_name=None, logger_fn=logger_fn)],
            callbacks_pre_training_fn=None,
            callbacks_post_training_fn=None
        )

        options = trw.train.create_default_options(num_epochs=5)
        trainer.fit(
            options,
            inputs_fn=create_dataset,
            model_fn=create_model,
            optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(datasets=datasets, model=model, learning_rate=0.1)
        )

        # locate the `trw.simple_layers.ShiftScale` layer (identity). Here we know
        # the expected statistics to compare against
        # if the log format changes, expect this test to fail!
        assert len(log_lines) >= 5
        line_of_interest = log_lines[5]
        assert 'ShiftScale' in line_of_interest
        line_of_interest = ' '.join(line_of_interest.split()).split()

        assert float(line_of_interest[5]) == 1.0  # mean
        assert float(line_of_interest[6]) == 0.0  # std
        assert float(line_of_interest[7]) == 1.0  # min
        assert float(line_of_interest[8]) == 1.0  # max




