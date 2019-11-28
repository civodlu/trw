import os
import functools
import logging
from trw.train import callback
from trw.train import trainer
from trw.train import utilities
from trw.train import sample_export


logger = logging.getLogger(__name__)


def callbacks_per_loss_term(
        dataset_name,
        split_name,
        batch,
        loss_terms,
        root,
        datasets_infos,
        loss_terms_inclusion,
        feature_exclusions,
        dataset_exclusions,
        split_exclusions,
        exported_cases,
        max_samples,
        epoch):
    
    # process the exclusion
    if dataset_name in dataset_exclusions:
        raise StopIteration()
    if split_name in split_exclusions:
        raise StopIteration()
    
    # copy to the current batch the specified loss terms
    for loss_term_name, loss_term in loss_terms.items():
        for loss_term_inclusion in loss_terms_inclusion:
            if loss_term_inclusion in loss_term:
                batch['term_{}_{}'.format(loss_term_name, loss_term_inclusion)] = loss_term[loss_term_inclusion]
    
    for feature_exclusion in feature_exclusions:
        if feature_exclusion in batch:
            del batch[feature_exclusion]
            
    # calculate how many samples to export
    nb_batch_samples = utilities.len_batch(batch)
    nb_samples_exported = len(exported_cases)
    nb_samples_to_export = min(max_samples - nb_samples_exported, nb_batch_samples)
    if nb_samples_to_export <= 0:
        raise StopIteration()

    # export the features
    for n in range(nb_samples_to_export):
        id = n + nb_samples_exported
        exported_cases.append(id)
        sample_output = os.path.join(root, dataset_name + '_' + split_name + '_s' + str(id) + '_e' + str(epoch))
        txt_file = sample_output + '.txt'
        classification_mappings = utilities.get_classification_mappings(datasets_infos, dataset_name, split_name)
        with open(txt_file, 'w') as f:
            sample_export.export_sample(
                batch,
                n,
                sample_output + '_',
                f,
                features_to_discard=['output_ref', 'loss'],
                classification_mappings=classification_mappings)


class CallbackExportSamples(callback.Callback):
    def __init__(self, max_samples=20, dirname='samples', loss_terms_inclusion=None, feature_exclusions=None, dataset_exclusions=None, split_exclusions=None):
        """
        Export random samples from our datasets
        
        Just for sanity check, it is always a good idea to make sure our data is loaded and processed
        as expected.
        
        :param max_samples: the maximum number of samples to be exported
        :param dirname: the root of the export directory
        :param loss_terms_inclusion: specifies what output name from each loss term will be exported. if None, defaults to ['output']
        :param feature_exclusions: specifies what feature should be excluded from the export
        :param split_exclusions: specifies what split should be excluded from the export
        :param dataset_exclusions: specifies what dataset should be excluded from the export
        """
        
        self.max_samples = max_samples
        self.dirname = dirname
        if loss_terms_inclusion is None:
            self.loss_terms_inclusion = ['output']
        else:
            self.loss_terms_inclusion = loss_terms_inclusion
        
        if feature_exclusions is not None:
            self.feature_exclusions = feature_exclusions
        else:
            self.feature_exclusions = []
            
        if dataset_exclusions is not None:
            self.dataset_exclusions = dataset_exclusions
        else:
            self.dataset_exclusions = []
            
        if split_exclusions is not None:
            self.split_exclusions = split_exclusions
        else:
            self.split_exclusions = []
    
    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('started CallbackExportSamples.__call__')
        device = options['workflow_options']['device']
        for dataset_name, dataset in datasets.items():
            root = os.path.join(options['workflow_options']['current_logging_directory'], self.dirname, dataset_name)
            if not os.path.exists(root):
                utilities.create_or_recreate_folder(root)
    
            for split_name, split in dataset.items():
                exported_cases = []
                trainer.eval_loop(device, dataset_name, split_name, split, model, losses[dataset_name],
                                  history=None,
                                  callbacks_per_batch=callbacks_per_batch,
                                  callbacks_per_batch_loss_terms=[
                                      functools.partial(
                                          callbacks_per_loss_term,
                                          root=root,
                                          datasets_infos=datasets_infos,
                                          loss_terms_inclusion=self.loss_terms_inclusion,
                                          feature_exclusions=self.feature_exclusions,
                                          dataset_exclusions=self.dataset_exclusions,
                                          split_exclusions=self.split_exclusions,
                                          exported_cases=exported_cases,
                                          max_samples=self.max_samples,
                                          epoch=len(history)
                                      )])
                
        logger.info('successfully completed CallbackExportSamples.__call__!')
