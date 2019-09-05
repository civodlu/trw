from trw.train import callback
from trw.train import utils
from trw.train import trainer
import collections
import numpy as np
import logging


logger = logging.getLogger(__name__)


def default_statistics():
    # TODO for the statistics to be exported, we must update `callback_export_history.default_metrics()` which is clunky. find a better mechanism
    # keep the ordering so that it is easier to read the logs
    embedding_statistics_fn = collections.OrderedDict()
    embedding_statistics_fn['max'] = np.max
    embedding_statistics_fn['min'] = np.min
    embedding_statistics_fn['mean'] = np.mean
    embedding_statistics_fn['std'] = np.std
    
    aggregate_statistics_fn = {
        'max': np.max,
        'min': np.min,
        'mean': np.mean,
        'std': np.mean,
    }
    
    return embedding_statistics_fn, aggregate_statistics_fn
    
    
class CollectBatchAndProcessStats:
    """
    Collect statistics on batches and aggregate them
    """
    def __init__(self, model, embedding_names, statistics_fn, number_of_samples_to_evaluate, embedding_output_name):
        self.embedding_names = embedding_names
        self.statistics_fn, self.statistics_aggregate_fn = statistics_fn
        self.nb_samples_evaluated = 0
        self.number_of_samples_to_evaluate = number_of_samples_to_evaluate
        self.embedding_output_name = embedding_output_name
        
        assert len(self.statistics_fn) == len(self.statistics_aggregate_fn), 'each stat must have an aggregation function'
        self.statistics = collections.defaultdict(lambda: collections.defaultdict(list))
        
    def __call__(self, dataset_name, split_name, batch, loss_terms):
        if self.nb_samples_evaluated >= self.number_of_samples_to_evaluate:
            raise StopIteration()
        
        for embedding_name in self.embedding_names:
            embedding = loss_terms.get(embedding_name)
            if embedding is None:
                continue
            output = embedding.get(self.embedding_output_name)
            if output is None:
                continue
            output = utils.to_value(output)
            for stat_name, stat_fn in self.statistics_fn.items():
                value = stat_fn(output)
                self.statistics[embedding_name][stat_name].append(value)
                
        self.nb_samples_evaluated += utils.len_batch(batch)
        
    def get_stats(self):
        stats_all = collections.OrderedDict()
        for embedding_name, metrics in self.statistics.items():
            stats = collections.OrderedDict()
            for metric_name, metric_values in metrics.items():
                value = self.statistics_aggregate_fn[metric_name](metric_values)
                stats[metric_name] = value
            stats_all[embedding_name] = stats
        return stats_all


class CallbackTensorboardEmbedding(callback.Callback):
    """
    This callback records the statistics of specified embeddings

    Note: we must recalculate the embedding as we need to associate a specific input (i.e., we can't store
        everything in memory so we need to collect what we need batch by batch)
    """
    def __init__(self, embedding_names, dataset_name=None, split_name='test', number_of_samples=2000, statistics=default_statistics(), embedding_output_name='output'):
        self.embedding_names = embedding_names
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.number_of_samples = number_of_samples
        self.initialized = False
        self.statistics = statistics
        self.embedding_output_name = embedding_output_name
    
    def first_time(self, datasets):
        self.initialized = True
        if self.dataset_name is None:
            self.dataset_name = next(iter(datasets))

        if datasets[self.dataset_name].get(self.split_name) is None:
            logger.error('can\'t find split={} for dataset={}'.format(self.dataset_name, self.split_name))
            self.dataset_name = None
            return
        
    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('nb_samples={}'.format(self.number_of_samples))
        
        if not self.initialized:
            self.first_time(datasets)
        
        if self.dataset_name is None:
            # we failed to initialize
            return

        device = options['workflow_options']['device']

        batch_processor = CollectBatchAndProcessStats(
            model=model,
            embedding_names=self.embedding_names,
            statistics_fn=self.statistics,
            number_of_samples_to_evaluate=self.number_of_samples,
            embedding_output_name=self.embedding_output_name)
        
        # TODO handle gradient statistics
        #trainer.train_loop(
        trainer.eval_loop(
            device,
            self.dataset_name,
            self.split_name,
            datasets[self.dataset_name][self.split_name],
            #None,
            model,
            losses[self.dataset_name],
            history,
            callbacks_per_batch=callbacks_per_batch,
            callbacks_per_batch_loss_terms=[batch_processor],
            #apply_backward=False  # collect the gradients BUT do not apply them!
        )
        
        # populate the history
        stats_by_embeding = batch_processor.get_stats()
        history_step = history[-1][self.dataset_name][self.split_name]
        for name, stats in stats_by_embeding.items():
            history_step[name] = stats
