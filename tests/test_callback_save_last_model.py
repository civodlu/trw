from unittest import TestCase
import trw.train
import torch.nn as nn
import tempfile
import glob
import os
import numpy as np
from trw.train.callback_save_last_model import exclude_large_embeddings


class ModelDense(nn.Module):
    def __init__(self, n_output=5):
        super().__init__()

        self.d1 = nn.Linear(10, 100)
        self.d2 = nn.Linear(100, n_output)

    def forward(self, batch):
        return self.d2(self.d1(batch['input']))


class TestCallbackSaveLastModel(TestCase):
    def test_rolling_models(self):
        callback = trw.train.CallbackSaveLastModel(
            model_name='checkpoint',
            with_outputs=True,
            is_versioned=True,
            rolling_size=5)

        model = ModelDense()
        logging_directory = tempfile.mkdtemp()
        options = trw.train.create_default_options(logging_directory=logging_directory)
        history = []
        for epoch in range(20):
            history.append({})
            callback(options, history, model, None, {'results': {'outputs': {}}}, None, {'info': 'dummy'}, None)

        models = glob.glob(os.path.join(logging_directory, '*.model'))
        results = glob.glob(os.path.join(logging_directory, '*.model.result'))
        assert len(models) == 5
        assert len(results) == 5

        oldest_model = os.path.split(sorted(models)[0])[1]
        assert oldest_model == 'checkpoint_e_16.model'

    def test_keep_best_model(self):
        callback = trw.train.CallbackSaveLastModel(
            keep_model_with_lowest_metric=trw.train.ModelWithLowestMetric(
                dataset_name='dataset1',
                split_name='split1',
                output_name='output1',
                metric_name='metric1',
                lowest_metric=0.2
            ))

        model = ModelDense()
        logging_directory = tempfile.mkdtemp()
        options = trw.train.create_default_options(logging_directory=logging_directory)

        outputs = {
            'dataset1': {
                'split1': {
                    'output1': {
                        'metric1': 0.9
                    }
                }
            }
        }
        callback(options, [outputs], model, None, outputs, None, {'info': 'dummy'}, None)

        # metric was not good enough. Do not save!
        models = glob.glob(os.path.join(logging_directory, '*.model'))
        assert len(models) == 1
        assert os.path.basename(models[0]) == 'last.model'

        # update with a better metric: expect `best model` to be exported!
        outputs['dataset1']['split1']['output1']['metric1'] = 0.01
        callback(options, [outputs], model, None, outputs, None, {'info': 'dummy'}, None)

        models = glob.glob(os.path.join(logging_directory, '*.model'))
        models = sorted(models, reverse=True)
        assert len(models) == 2
        assert os.path.basename(models[-1]) == 'best.model'

        # make sure we don't export worst model
        outputs['dataset1']['split1']['output1']['metric1'] = 0.1
        os.remove(models[-1])
        callback(options, [outputs], model, None, outputs, None, {'info': 'dummy'}, None)

        models = glob.glob(os.path.join(logging_directory, '*.model'))
        assert len(models) == 1

    def test_post_process_outputs(self):
        callback = trw.train.CallbackSaveLastModel(post_process_outputs=exclude_large_embeddings)

        model = ModelDense()
        logging_directory = tempfile.mkdtemp()
        options = trw.train.create_default_options(logging_directory=logging_directory)

        outputs = {
            'dataset1': {
                'split1': {
                    'output1': {
                        'output': np.zeros([1, 20000]),
                        'output_ref': trw.train.outputs_trw.OutputEmbedding(np.zeros([10, 1]))
                    }
                }
            }
        }
        callback(options, [outputs], model, None, outputs, None, {'info': 'dummy'}, None)

        # metric was not good enough. Do not save!
        models = glob.glob(os.path.join(logging_directory, '*.model'))
        assert len(models) == 1
        assert os.path.basename(models[0]) == 'last.model'

        _, results_reloaded = trw.train.Trainer.load_model(models[0], with_result=True)
        # the embedding value should be stripped
        assert len(results_reloaded['outputs']['dataset1']['split1']['output1']) == 0
