from . import callback
from . import utils
from . import sequence_array
from . import sample_export
import logging
import os
import collections
import numpy as np


logger = logging.getLogger(__name__)


class CallbackExportAugmentations(callback.Callback):
    """
    Export samples
    """
    def __init__(self, nb_samples=10, nb_augmentation=5, dirname='augmentations', split_name=None, uid_name='sample_uid', keep_samples=False):
        """

        Args:
            nb_samples: the number of samples to export
            nb_augmentation: the number of augmentations per sample to export
            dirname: the folder where to export the augmentations
            split_name: the name of the split, typically the training split. If None, we will use the default training name from the options
            uid_name: this is the UID name that will be used to detect the samples (augmentations will be aggregated by `uid_name`)
            keep_samples: if True, the collected augmentations will be kept in `list_of_samples_by_uid`. This is essentially for testing purposes
        """
        self.nb_samples = nb_samples
        self.nb_augmentation = nb_augmentation
        self.dirname = dirname
        self.split_name = split_name
        self.uid_name = uid_name
        self.keep_samples = keep_samples

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('started CallbackExportAugmentations.__call__')
        if self.split_name is None:
            self.split_name = options['workflow_options']['train_split']

        if self.keep_samples:
            self.list_of_samples_by_uid = []

        for dataset_name, dataset in datasets.items():
            logger.info('collecting samples for dataset={}'.format(dataset_name))
            root = os.path.join(options['workflow_options']['current_logging_directory'], self.dirname, dataset_name)
            if not os.path.exists(root):
                utils.create_or_recreate_folder(root)

            split = dataset.get(self.split_name)
            if split is None:
                continue

            # run the augmentations on the subsampled split
            # then collect the samples by UID, these represent our augmentations
            samples_by_uid = collections.defaultdict(list)
            split_subsampled = split.subsample(self.nb_samples)
            for augmentation_id in range(self.nb_augmentation):
                nb_samples_recorded = 0
                for batch in split_subsampled:
                    batch = {name: utils.to_value(values) for name, values in batch.items()}
                    uids = utils.to_value(batch.get(self.uid_name))
                    if uids is None:
                        logger.error('no UID found in the dataset! Can\'t link the augmentations')
                        return
                    assert uids is not None, 'we must have a unique UID for each sample!'
                    nb_samples = len(uids)
                    for index, uid in enumerate(uids):
                        sample = sequence_array.SequenceArray.get(batch, nb_samples, [index], None, use_advanced_indexing=True)
                        samples_by_uid[uid].append(sample)
                    nb_samples_recorded += len(uids)
                    if nb_samples_recorded >= self.nb_samples:
                        # we might have used a resampled sampler, so double check the number
                        # of samples too and not rely solely on the en of the sequence iterator
                        break

            if self.keep_samples:
                # this is mostly useful for unit tests
                self.list_of_samples_by_uid.append(samples_by_uid)

            # finally, export the samples by augmentation with all metadata
            logger.info('exporting samples...')
            for uid, samples in samples_by_uid.items():
                for augmentation_id, sample in enumerate(samples[:self.nb_augmentation]):  # we may end up with more augmentation recorded, so trim them
                    sample = {name: np.asarray([utils.to_value(value)]) for name, value in sample.items()}  # we must have a numpy data here to be exportable

                    sample_output = os.path.join(root, dataset_name + '_' + self.split_name + '_s' + str(uid) + '_a' + str(augmentation_id))
                    txt_file = sample_output + '.txt'
                    classification_mappings = utils.get_classification_mappings(datasets_infos, dataset_name, self.split_name)
                    with open(txt_file, 'w') as f:
                        sample_export.export_sample(
                            sample,
                            0,  # we can only have 1 sample here!
                            sample_output + '_',
                            f,
                            features_to_discard=['output_ref', 'loss'],
                            classification_mappings=classification_mappings)

        logger.info('successfully completed CallbackExportAugmentations.__call__')






