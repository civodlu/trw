from trw.train import callback
from trw.train import trainer
import os
import logging


logger = logging.getLogger(__name__)


class CallbackSaveLastModel(callback.Callback):
    """
    When the training is finished, save the full model and result
    """
    def __init__(self, model_name='last'):
        self.model_name = model_name

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):

        result = {
            'history': history,
            'options': options,
            'outputs': outputs,
            'datasets_infos': datasets_infos
        }
        export_path = os.path.join(options['workflow_options']['current_logging_directory'], 'last.model')

        logger.info('started CallbackSaveLastModel.__call__ path={}'.format(export_path))
        trainer.Trainer.save_model(model, result, export_path)
        logger.info('successfully completed CallbackSaveLastModel.__call__')
