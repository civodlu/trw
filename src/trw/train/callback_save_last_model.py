from trw.train import callback
from trw.train import trainer
import os
import logging


logger = logging.getLogger(__name__)


class CallbackSaveLastModel(callback.Callback):
    """
    Save the current model to disk as well as metadata (history, outputs, infos).

    This callback can be used during training (e.g., checkpoint) or at the end of the training.
    """
    def __init__(self, model_name='last', with_outputs=True, is_versioned=False, rolling_size=None):
        """

        Args:
            model_name: the root name of the model
            with_outputs: if True, the outputs will be exported along the model
            is_versioned: if versioned, model name will include the current epoch so that we can have multiple
                versions of the same model
            rolling_size: the number of model files that are kept on the drive. If more models are exported,
                the oldest model files will be erased
        """
        self.model_name = model_name
        self.with_outputs = with_outputs
        self.is_versioned = is_versioned
        self.rolling_size = rolling_size
        self.last_models = []

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        if not self.with_outputs:
            outputs = None

        result = {
            'history': history,
            'options': options,
            'outputs': outputs,
            'datasets_infos': datasets_infos
        }

        if self.is_versioned:
            name = f'{self.model_name}_e_{len(history)}.model'
        else:
            name = f'{self.model_name}.model'
        export_path = os.path.join(options['workflow_options']['current_logging_directory'], name)

        logger.info('started CallbackSaveLastModel.__call__ path={}'.format(export_path))
        trainer.Trainer.save_model(model, result, export_path)
        if self.rolling_size is not None and self.rolling_size > 0:
            self.last_models.append(export_path)

            if len(self.last_models) > self.rolling_size:
                model_location_to_delete = self.last_models.pop(0)
                model_result_location_to_delete = model_location_to_delete + '.result'
                logger.info(f'deleted model={model_location_to_delete}')
                os.remove(model_location_to_delete)
                os.remove(model_result_location_to_delete)

        logger.info('successfully completed CallbackSaveLastModel.__call__')
