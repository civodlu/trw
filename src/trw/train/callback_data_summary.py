from trw.train import callback
from trw.train import utilities
import collections
import numpy as np
import logging


logger = logging.getLogger(__name__)


class CallbackDataSummary(callback.Callback):
    """
    Summarizes the data (min value, max value, number of batches, shapes) for each split of each dataset
    """
    def __init__(self, logger=utilities.log_and_print, collect_stats=True):
        self.logger = logger
        self.collect_stats = collect_stats

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        @utilities.time_it(log=self.logger, time_name='CallbackDataSummary')
        def run():
            stats_to_average = ['mean', 'std']
            for dataset_name, dataset in datasets.items():
                for split_name, split in dataset.items():
                    nb_samples = 0
                    batch_id = 0
                    batch = None
                    features_stats = collections.defaultdict(collections.OrderedDict)
                    for batch_id, batch in enumerate(split):
                        nb_samples += utilities.len_batch(batch)
                        if self.collect_stats:
                            for feature_name, feature_value in batch.items():
                                feature_value = utilities.to_value(feature_value)
                                stats = features_stats[feature_name]
                                if isinstance(feature_value, np.ndarray):
                                    if batch_id == 0:
                                        stats['shape'] = feature_value.shape
                                        stats['max'] = np.max(feature_value)
                                        stats['min'] = np.min(feature_value)
                                        stats['mean'] = np.mean(feature_value)
                                        stats['std'] = np.std(feature_value)
                                    else:
                                        stats['max'] = max(np.max(feature_value), stats['max'])
                                        stats['min'] = min(np.min(feature_value), stats['min'])
                                        stats['mean'] += np.mean(feature_value)
                                        stats['std'] += np.std(feature_value)

                    nb_batches = batch_id + 1
                    self.logger('dataset={}, split={}, nb_samples={}, features={}, nb_batches={}'.
                                format(dataset_name,
                                       split_name,
                                       nb_samples,
                                       list(batch.keys()),
                                       nb_batches))
                    for feature_name, feature_stats in features_stats.items():
                        self.logger('  feature_name={}'.format(feature_name))
                        for stat_name, stat in feature_stats.items():
                            if stat_name in stats_to_average:
                                stat /= nb_batches
                            self.logger('    {}={}'.format(stat_name, stat))

        logger.info('CallbackDataSummary started')
        run()
        logger.info('CallbackDataSummary successfully completed!')
