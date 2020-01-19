import os
from trw.train import callback
from trw.train import utilities
from trw.train import sample_export
from trw.train import outputs as outputs_trw
from trw.train import trainer
import logging
import functools


logger = logging.getLogger(__name__)


def callback_per_batch(dataset_name, split_name, batch, loss_terms, root, datasets_infos, max_nb_samples, nb_samples_exported, epoch):
    """
    Collect the model output batch by batch until we have reached the requested number of samples to be exported
    """
    for output_name, output in loss_terms.items():
        output_and_batch_merged = {**batch, **output}  # TODO all outputs merged? name collision?

        ref = output.get('output_ref')
        if ref is not None and isinstance(ref, outputs_trw.OutputSegmentation):
            batch_size = utilities.len_batch(output)

            # copy the making (if any) of the class to the output
            classification_mappings = utilities.get_classification_mappings(datasets_infos, dataset_name, split_name)
            if classification_mappings is not None:
                classification_mapping = classification_mappings.get(ref.classes_name)
                if classification_mapping is not None:
                    classification_mappings['output'] = classification_mapping

            # finally, export the errors
            nb_samples_to_export = batch_size
            if nb_samples_exported[0] + batch_size > max_nb_samples:
                nb_samples_to_export = max_nb_samples - nb_samples_exported[0]

            for id in range(nb_samples_to_export):
                sample_output = os.path.join(root, output_name + '_' + split_name + '_s' + str(id + nb_samples_exported[0]) + '_e' + str(epoch))
                txt_file = sample_output + '.txt'
                with open(txt_file, 'w') as f:
                    sample_export.export_sample(
                        output_and_batch_merged,
                        id,
                        sample_output + '_',
                        f,
                        features_to_discard=['output_ref', 'loss'],
                        classification_mappings=classification_mappings)
            nb_samples_exported[0] += nb_samples_to_export

            if nb_samples_exported[0] >= max_nb_samples:
                raise StopIteration()  # abort the loop, we have already too many samples


class CallbackExportSegmentations(callback.Callback):
    """
    Export the segmentations produced by a model for a given sequence
    """
    def __init__(self,
                 nb_samples=100,
                 dirname='segmentations',
                 dataset_name=None,
                 split_name=None):
        self.nb_samples = nb_samples
        self.dirname = dirname
        self.dataset_name = dataset_name
        self.split_name = split_name

        self.export_path = None
        self.batch = None

    def first_time(self, options, datasets):
        # here we only want to collect the kernels a single time per epoch, so fix the dataset/split names
        if self.dataset_name is None or self.split_name is None:
            self.dataset_name, self.split_name = utilities.find_default_dataset_and_split_names(
                datasets,
                default_dataset_name=self.dataset_name,
                default_split_name=self.split_name,
                train_split_name=options['workflow_options']['train_split'])

        if self.dataset_name is None or self.split_name is None:
            logger.error('can\'t find a dataset name or split name!')
            return

        self.export_path = os.path.join(options['workflow_options']['current_logging_directory'], self.dirname)
        utilities.create_or_recreate_folder(self.export_path)
        self.batch = next(iter(datasets[self.dataset_name][self.split_name]))

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        if self.split_name is None or self.dataset_name is None:
            self.first_time(options, datasets)

        if self.split_name is None or self.dataset_name is None:
            return None

        logger.info('started CallbackExportSegmentation.__call__')

        for output_name, output_values in outputs[self.dataset_name][self.split_name].items():
            ref = output_values.get('output_ref')
            if ref is None:
                continue
            if isinstance(ref, outputs_trw.OutputSegmentation):
                nb_exported = [0]
                device = options['workflow_options']['device']
                trainer.eval_loop(
                    device,
                    self.dataset_name,
                    self.split_name,
                    datasets[self.dataset_name][self.split_name], model, losses[self.dataset_name],
                    history=None,
                    callbacks_per_batch=callbacks_per_batch,
                    callbacks_per_batch_loss_terms=[functools.partial(
                        callback_per_batch,
                        root=self.export_path,
                        datasets_infos=datasets_infos,
                        nb_samples_exported=nb_exported,
                        max_nb_samples=self.nb_samples,
                        epoch=len(history))])

        logger.info('successfully completed CallbackExportSegmentation.__call__')