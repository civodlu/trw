from typing import List, Callable, Optional

from trw.basic_typing import Datasets
from trw.utils import safe_lookup
from trw.train import callback
from trw.train.outputs_trw import OutputEmbedding
from trw.train import trainer
import os
import logging

logger = logging.getLogger(__name__)


class ModelWithLowestMetric:
    def __init__(self, dataset_name, split_name, output_name, metric_name, lowest_metric=0.2):
        """

        Args:
            dataset_name: the dataset name to be considered for the best model
            split_name: the split name to be considered for the best model
            metric_name: the metric name to be considered for the best model
            lowest_metric: consider only the metric lower than this threshold
            output_name: the output to be considered for the best model selection
        """
        self.output_name = output_name
        self.metric_name = metric_name
        self.split_name = split_name
        self.dataset_name = dataset_name
        self.lowest_metric = lowest_metric


def exclude_large_embeddings(outputs: Datasets, counts_greater_than=10000) -> Datasets:
    """
    Remove from the outputs embeddings larger than a specified threshold.

    Args:
        outputs: the outputs to check
        counts_greater_than: the number of elements above which the embedding will be stripped

    Returns:
        outputs
    """
    for dataset_name, dataset in outputs.items():
        for split_name, split in dataset.items():
            # first collect the outputs to be discarded
            outputs_to_remove = []
            for output_name, output in split.items():
                output_ref = output.get('output_ref')
                if output_ref is not None and isinstance(output_ref, OutputEmbedding):
                    count = output['output'].reshape(-1).shape[0]
                    if count >= counts_greater_than:
                        outputs_to_remove.append(output_name)

            # first collect the outputs to be discarded
            for output_to_remove in outputs_to_remove:
                split[output_to_remove] = {}
    return outputs


class CallbackSaveLastModel(callback.Callback):
    """
    Save the current model to disk as well as metadata (history, outputs, infos).

    This callback can be used during training (e.g., checkpoint) or at the end of the training.

    Optionally, record the best model for a given dataset, split, output and metric.
    """

    def __init__(
            self,
            model_name='last',
            with_outputs=True,
            is_versioned=False,
            rolling_size=None,
            keep_model_with_lowest_metric: ModelWithLowestMetric = None,
            best_model_name='best',
            post_process_outputs: Optional[Callable[[Datasets], Datasets]] = exclude_large_embeddings,
    ):
        """
        Args:
            model_name: the root name of the model
            with_outputs: if True, the outputs will be exported along the model
            is_versioned: if versioned, model name will include the current epoch so that we can have multiple
                versions of the same model
            rolling_size: the number of model files that are kept on the drive. If more models are exported,
                the oldest model files will be erased
            keep_model_with_lowest_metric: if not None, the best model for a given metric will be recorded
            best_model_name: the name to be used by the best model
            post_process_outputs: a function to post-process the outputs just before export. For example,
                if can be used to remove large embeddings to save smaller output files.
        """
        self.best_model_name = best_model_name
        if keep_model_with_lowest_metric is not None:
            assert isinstance(keep_model_with_lowest_metric, ModelWithLowestMetric), \
                'must be ``None`` or ``ModelWithLowestMetric`` instance'
        self.keep_model_with_lowest_metric = keep_model_with_lowest_metric
        self.model_name = model_name
        self.with_outputs = with_outputs
        self.is_versioned = is_versioned
        self.rolling_size = rolling_size
        self.last_models: List[str] = []
        self.post_process_outputs = post_process_outputs

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):
        result = {
            'history': history,
            'options': options,
            'outputs': outputs,
            'datasets_infos': datasets_infos
        }

        if not self.with_outputs:
            # discard the outputs (e.g., for large outputs)
            result['outputs'] = None
        elif self.post_process_outputs is not None:
            result['outputs'] = self.post_process_outputs(result['outputs'])

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

        if self.keep_model_with_lowest_metric is not None:
            # look up the correct metric and record the model and results
            # if we obtain a better (lower) metric.
            metric_value = safe_lookup(
                history[-1],
                self.keep_model_with_lowest_metric.dataset_name,
                self.keep_model_with_lowest_metric.split_name,
                self.keep_model_with_lowest_metric.output_name,
                self.keep_model_with_lowest_metric.metric_name,
            )

            if metric_value is not None and metric_value < self.keep_model_with_lowest_metric.lowest_metric:
                self.keep_model_with_lowest_metric.lowest_metric = metric_value
                export_path = os.path.join(
                    options['workflow_options']['current_logging_directory'],
                    f'{self.best_model_name}.model')
                trainer.Trainer.save_model(model, result, export_path)

        logger.info('successfully completed CallbackSaveLastModel.__call__')
