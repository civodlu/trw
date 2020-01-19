from unittest import TestCase
import trw.train
import torch.nn as nn
import tempfile
import glob
import os


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