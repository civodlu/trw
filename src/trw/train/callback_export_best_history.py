from trw.train import callback
import logging
import os
import collections


logger = logging.getLogger(__name__)


class CallbackExportBestHistory(callback.Callback):
    """
    Export the best value of the history and epoch for each metric in a single file

    This can be useful to accurately get the best value of a metric and in particular
    at which step it occurred.
    """
    def __init__(self, filename='best_history.txt', metric_to_discard=[]):
        self.filename = filename
        self.metric_to_discard = metric_to_discard

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('CallbackExportHistory.__call__ started')
        export_root = options['workflow_options']['current_logging_directory']
        export_file = os.path.join(export_root, self.filename)

        # collect the min value of each metric by output/split/dataset
        best_values_step = collections.OrderedDict()
        for index, history_step in enumerate(history):
            for dataset_name, dataset in history_step.items():
                for split_name, split in dataset.items():
                    for output_name, output in split.items():
                        for metric_name, metric_value in output.items():
                            if metric_name in self.metric_to_discard:
                                continue
                            name = '{}_{}_{}_{}'.format(dataset_name, split_name, output_name, metric_name)
                            best_value_step = best_values_step.get(name)
                            if best_value_step is not None:
                                best_value, best_step = best_value_step
                                if metric_value < best_value:
                                    best_values_step[name] = (metric_value, index)
                            else:
                                best_values_step[name] = (metric_value, index)

        # export as in a text file
        with open(export_file, 'w') as f:
            for name, value in best_values_step.items():
                f.write('{}={}, step={}\n'.format(name, value[0], value[1]))

