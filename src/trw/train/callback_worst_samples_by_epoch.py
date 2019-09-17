from trw.train import callback
from trw.train import utils
from trw.train import outputs as trw_outputs
from trw.train import sample_export
from trw.train import sequence_array
from trw.train import sampler
import os
import logging
import collections
import numpy as np
from matplotlib import cm


logger = logging.getLogger(__name__)


def get_first_output_of_interest(outputs, dataset_name, split_name, output_of_interest):
    """
    Return the first output of interest of a given dataset name

    Args:
        outputs: a dictionary (datasets) of dictionary (splits) of dictionary (outputs)
        dataset_name: the dataset to consider. If `None`, the first dataset is considered
        split_name: the split name to consider. If `None`, the first split is selected
        output_of_interest: the output to consider

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
        if output_ref is not None and isinstance(output_ref, output_of_interest):
            return output_name

    return None


def export_samples(dataset_name, split_name, sequence, max_samples, root, uids_name, datasets_infos):
    """
    Export samples of a sequence
    """
    # export the features
    nb_samples_exported = 0
    for batch in sequence:
        nb_samples = utils.len_batch(batch)
        ids_values = utils.to_value(batch.get(uids_name))
        assert ids_values is not None
        for n in range(nb_samples):
            if nb_samples_exported >= max_samples:
                break

            id = ids_values[n]
            sample_output = os.path.join(root, dataset_name + '_' + split_name + '_rank-' + str(nb_samples_exported) + '_s' + str(id))
            txt_file = sample_output + '.txt'
            classification_mappings = utils.get_classification_mappings(datasets_infos, dataset_name, split_name)
            with open(txt_file, 'w') as f:
                sample_export.export_sample(
                    batch,
                    n,
                    sample_output + '_',
                    f,
                    features_to_discard=['output_ref', 'loss'],
                    classification_mappings=classification_mappings)

            nb_samples_exported += 1


class CallbackWorstSamplesByEpoch(callback.Callback):
    """
    The purpose of this callback is to track the samples with the worst loss during the training of the model

    It is interesting to understand what are the difficult samples (train and test split), are they always
    wrongly during the training or random? Are they the same samples with different models (i.e., initialization
    or model dependent)?
    """

    def __init__(
            self,
            split_names=None,
            output_name=None,
            dataset_name=None,
            dirname='worst_samples_by_epoch',
            sort_samples_by_loss_error=True,
            worst_k_samples=1000,
            export_top_k_samples=50,
            uids_name=sequence_array.sample_uid_name,
            output_of_interest=(trw_outputs.OutputClassification, trw_outputs.OutputSegmentation, trw_outputs.OutputRegression)):
        """

        Args:
            output_name: the output to analyse. If `None`, the first classification output returned by the model will be used
            dataset_name: the dataset to analyze. If `None` keep track of the first dataset only
            split_names: a list of split name. If split is `None`, record all the splits
            dirname: where to export the files
            sort_samples_by_loss_error: if True, sort the data
            worst_k_samples: select the worst K samples to export. If `None`, export all samples
            export_top_k_samples: export the top `k` samples with the overall worst loss
            uids_name: the name of the UIDs of the dataset sequence. If `None`, samples may not be exported
            output_of_interest: the first output node to select in case `output_name` is `None`
        """
        self.dirname = dirname
        self.split_names = split_names
        self.output_name = output_name
        self.dataset_name = dataset_name
        self.sort_samples_by_loss_error = sort_samples_by_loss_error
        self.errors_by_split = collections.defaultdict(lambda: collections.defaultdict(list))
        self.worst_k_samples = worst_k_samples
        self.current_epoch = None
        self.root = None
        self.output_of_interest = output_of_interest
        self.export_top_k_samples = export_top_k_samples
        self.uids_name = uids_name

    def first_time(self, datasets, outputs):
        logger.info('CallbackWorstSamplesByEpoch.first_time')
        if self.dataset_name is None:
            self.dataset_name = next(iter(datasets))
            logger.info('dataset={}'.format(self.dataset_name))

        if self.output_name is None:
            self.output_name = get_first_output_of_interest(outputs, dataset_name=self.dataset_name, split_name=None, output_of_interest=self.output_of_interest)
            logger.info('output selected={}'.format(self.output_name))

        if self.dataset_name is not None and self.split_names is None:
            dataset = datasets.get(self.dataset_name)
            self.split_names = list(dataset.keys())

        logger.info('CallbackWorstSamplesByEpoch.first_time done!')

    @staticmethod
    def sort_split_data(errors_by_sample, worst_k_samples, discard_first_n_epochs=0):
        """
        Helper function to sort the samples

        Args:
            errors_by_sample: the data
            worst_k_samples: the number of samples to select or `None`
            discard_first_n_epochs: the first few epochs are typically very noisy, so don't use these

        Returns:
            sorted data
        """
        last_epoch_losses = []
        nb_errors = collections.defaultdict(lambda: 0)
        for uid, loss_by_epoch in errors_by_sample.items():
            last_epoch_losses.append(loss_by_epoch[-1][0])
            for loss, epoch in loss_by_epoch[discard_first_n_epochs:]:
                nb_errors[uid] += loss

        sorted_uid_nb = sorted(list(nb_errors.items()), key=lambda values: values[1], reverse=True)
        nb_samples = len(nb_errors)
        if worst_k_samples is not None:
            nb_samples = min(worst_k_samples, nb_samples)

        sorted_errors_by_sample = []
        for i in range(nb_samples):
            uid, nb_e = sorted_uid_nb[i]
            sorted_errors_by_sample.append((uid, errors_by_sample[uid]))

        mean_loss = np.mean(last_epoch_losses)
        std_loss = np.std(last_epoch_losses)
        return sorted_errors_by_sample, np.median(last_epoch_losses), mean_loss + 6 * std_loss

    def export_stats(self, datasets, datasets_infos):
        if self.current_epoch is None or len(self.errors_by_split) == 0:
            return

        color_map = cm.get_cmap('autumn')
        assert color_map is not None, 'can\'t find colormap!'

        nb_epochs = self.current_epoch + 1
        for split_name, split_data in self.errors_by_split.items():
            # create a 2D map (epoch, samples)
            sorted_errors_by_sample, min_loss, max_loss = CallbackWorstSamplesByEpoch.sort_split_data(
                split_data,
                self.worst_k_samples
            )

            nb_samples = len(sorted_errors_by_sample)
            image = np.zeros([nb_epochs, nb_samples, 3], dtype=np.float)

            for sample_index, (uid, loss_epochs) in enumerate(sorted_errors_by_sample):
                for loss, epoch in loss_epochs:
                    normalized_0_1_loss = (loss - min_loss) / (max_loss - min_loss)
                    if normalized_0_1_loss < 0:
                        normalized_0_1_loss = 0.0
                    if normalized_0_1_loss > 1:
                        normalized_0_1_loss = 1.0
                    image[epoch, sample_index] = color_map(1.0 - normalized_0_1_loss)[:3]

            image = (image * 255.0).astype(np.uint8)
            image_path = os.path.join(self.root, '{}-{}-{}-e{}'.format(self.dataset_name, split_name, self.output_name, nb_epochs))
            sample_export.export_image(image, image_path + '.png')

            # export basic info so that we can at least track back the samples
            with open(image_path + '.txt', 'w') as f:
                for uid, loss_epoch_list in sorted_errors_by_sample:
                    loss_sum = 0
                    for loss, _ in loss_epoch_list:
                        loss_sum += loss
                    f.write('uid={}, loss_sum={}, nb={}\n'.format(uid, loss_sum, len(loss_epoch_list)))

            # optionally, export the actual samples
            if self.export_top_k_samples > 0:
                dataset = datasets[self.dataset_name]
                if dataset is None:
                    return
                split = dataset[split_name]
                if split is None:
                    return
                uids_to_export = [t[0] for t in sorted_errors_by_sample[:self.export_top_k_samples]]
                subsampled_split = split.subsample_uids(uids=uids_to_export, uids_name=self.uids_name, new_sampler=sampler.SamplerSequential())
                export_samples(
                    self.dataset_name,
                    split_name,
                    subsampled_split,
                    self.export_top_k_samples,
                    self.root,
                    uids_name=self.uids_name,
                    datasets_infos=datasets_infos)

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
                    losses = utils.to_value(output['losses'])
                    assert len(uids) == len(losses)
                    for loss, uid in zip(losses, uids):
                        # record the epoch: for example if we have resampled dataset,
                        # we may not have all samples selected every epoch so we can
                        # display properly these epochs
                        self.errors_by_split[split_name][uid].append((loss, self.current_epoch))

        last_epoch = kwargs.get('last_epoch')
        if last_epoch:
            self.export_stats(datasets, datasets_infos)
