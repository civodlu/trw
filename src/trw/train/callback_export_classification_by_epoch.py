from trw.train import callback
from trw.train import utils
from trw.train import outputs as trw_outputs
from trw.train import sample_export
import os
import logging
import collections
import numpy as np


logger = logging.getLogger(__name__)


def get_first_classification_output(outputs, dataset_name, split_name):
    """
    Return the first classification output of a given dataset name

    Args:
        outputs: a dictionary (datasets) of dictionary (splits) of dictionary (outputs)
        dataset_name: the dataset to consider. If `None`, the first dataset is considered
        split_name: the split name to consider. If `None`, the first split is selected

    Returns:

    """
    if dataset_name is None:
        dataset_name = next(iter(outputs.keys()))

    dataset = outputs.get(dataset_name)
    if dataset is None:
        return None

    if split_name is None:
        split = next(iter(dataset.values()))
    else:
        split = dataset.get(split_name)

    if split is None:
        return None

    for output_name, output in split.items():
        output_ref = output.get('output_ref')
        if output_ref is not None and isinstance(output_ref, trw_outputs.OutputClassification):
            return output_name

    return None


class CallbackExportClassificationByEpoch(callback.Callback):
    """
    The purpose of this callback is to track how a sample is classified during the training of the model

    It is interesting to understand what are the difficult samples (train and test split), are they always
    wrongly during the training or random? Are they the same samples with different models (i.e., initialization
    or model dependent)?
    """
    def __init__(self, split_names=None, output_name=None, dataset_name=None, dirname='error_by_epoch', sort_samples_by_classification_error=True, worst_k_samples=1000):
        """

        Args:
            output_name: the classification output to analyse. If `None`, the first classification output returned by the model will be used
            dataset_name: the dataset to analyze. If `None` keep track of the first dataset only
            split_names: a list of split name. If split is `None`, record all the splits
            dirname: where to export the files
            sort_samples_by_classification_error: if True, sort the data
            worst_k_samples: select the worst K samples to export. If `None`, export all samples
        """
        self.dirname = dirname
        self.split_names = split_names
        self.output_name = output_name
        self.dataset_name = dataset_name
        self.sort_samples_by_classification_error = sort_samples_by_classification_error
        self.errors_by_split = collections.defaultdict(lambda: collections.defaultdict(list))
        self.worst_k_samples = worst_k_samples
        self.current_epoch = None
        self.root = None

    def first_time(self, datasets, outputs):
        logger.info('CallbackExportClassificationByEpoch.first_time')
        if self.dataset_name is None:
            self.dataset_name = next(iter(datasets))
            logger.info('dataset={}'.format(self.dataset_name))

        if self.output_name is None:
            self.output_name = get_first_classification_output(outputs, dataset_name=self.dataset_name, split_name=None)
            logger.info('classification output selected={}'.format(self.output_name))

        if self.dataset_name is not None and self.split_names is None:
            dataset = datasets.get(self.dataset_name)
            self.split_names = list(dataset.keys())

        logger.info('CallbackExportClassificationByEpoch.first_time done!')

    @staticmethod
    def sort_split_data(errors_by_sample, worst_k_samples):
        """
        Helper function to sort the samples

        Args:
            errors_by_sample: the data
            worst_k_samples: the number of samples to select or `None`

        Returns:
            sorted data
        """
        nb_errors = collections.defaultdict(lambda: 0)
        for uid, classification_by_epoch in errors_by_sample.items():
            for classification_correct, epoch in classification_by_epoch:
                if not classification_correct:
                    nb_errors[uid] += 1

        sorted_uid_nb = sorted(list(nb_errors.items()), key=lambda values: values[1], reverse=True)
        nb_samples = len(nb_errors)
        if worst_k_samples is not None:
            nb_samples = min(worst_k_samples, nb_samples)

        sorted_errors_by_sample = []
        for i in range(nb_samples):
            uid, nb_e = sorted_uid_nb[i]
            sorted_errors_by_sample.append((uid, errors_by_sample[uid]))

        return sorted_errors_by_sample

    def export_stats(self):
        if self.current_epoch is None or len(self.errors_by_split) == 0:
            return
        
        nb_epochs = self.current_epoch + 1
        for split_name, split_data in self.errors_by_split.items():
            # create a 2D map (epoch, samples)
            sorted_errors_by_sample = CallbackExportClassificationByEpoch.sort_split_data(split_data, self.worst_k_samples)
            nb_samples = len(sorted_errors_by_sample)
            image = np.zeros([nb_epochs, nb_samples, 3], dtype=np.uint8)

            for sample_index, (uid, correct_epochs) in enumerate(sorted_errors_by_sample):
                for correct, epoch in correct_epochs:
                    if correct:
                        image[epoch, sample_index] = (0, 255, 0)
                    else:
                        image[epoch, sample_index] = (255, 0, 0)

            image_path = os.path.join(self.root, '{}-{}-{}-e{}'.format(self.dataset_name, split_name, self.output_name, nb_epochs))
            sample_export.export_image(image, image_path + '.png')

            # export info so that we can track back the samples
            with open(image_path + '.txt', 'w') as f:
                for uid, correct_epoch_list in sorted_errors_by_sample:
                    nb_errors = 0
                    for correct, epoch in correct_epoch_list:
                        if not correct:
                            nb_errors += 1
                    f.write('uid={}, errors={}, nb={}\n'.format(uid, nb_errors, len(correct_epoch_list)))

    def __del__(self):
        # once we are about to destroy this object,
        # export the results
        self.export_stats()

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        if self.dataset_name is None and outputs is not None:
            self.first_time(datasets, outputs)
        if self.output_name is None or self.split_names is None or self.dataset_name is None:
            return

        self.root = os.path.join(options['workflow_options']['current_logging_directory'], self.dirname)
        if not os.path.exists(self.root):
            utils.create_or_recreate_folder(self.root)

        dataset_output = outputs.get(self.dataset_name)
        if dataset_output is None:
            return

        self.current_epoch = len(history) - 1
        for split_name in self.split_names:
            split_output = dataset_output.get(split_name)
            if split_output is not None:
                output = split_output.get(self.output_name)
                if output is not None and 'uid' in output:
                    uids = output['uid']
                    output_found = output['output']
                    output_truth = output['output_truth']
                    assert len(uids) == len(output_found)
                    assert len(uids) == len(output_truth)
                    for found, truth, uid in zip(output_found, output_truth, uids):
                        # record the epoch: for example if we have resampled dataset,
                        # we may not have all samples selected every epoch so we can
                        # display properly these epochs
                        self.errors_by_split[split_name][uid].append((found == truth, self.current_epoch))
