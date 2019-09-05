from . import callback
from . import utils
from . import outputs as trf_outputs
from . import sample_export
from . import trainer
import numpy as np
import os
import functools
import collections
import logging


logger = logging.getLogger(__name__)


class CallbackExportClassificationErrors(callback.Callback):
    """
    Export the classification errors

    Note: since we can't guaranty the repeatability of the input (i.e., from the outputs, we can't
    associate the corresponding batches), we need to re-run the evaluation and collect batch by batch
    the errors.
    """
    def __init__(self, max_samples=100, discard_train=True, dirname='errors'):
        """
        :param max_samples: the maximum number off samples to be exported per split per output. If `None`, all the errors will be recorded
        :param discard_train: if True, the train split will be discarded
        :param dirname: where to store the errors
        """
        self.max_samples = max_samples
        self.discard_train = discard_train
        self.dirname = dirname

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        def callback_per_batch(dataset_name, split_name, batch, loss_terms, errors_by_output, root, datasets_infos):
            all_outputs_maxedout = True
            for output_name, output in loss_terms.items():
                output_and_batch_merged = {**batch, **output}  # TODO all outputs merged? name collision?

                errors = errors_by_output[output_name]
                if self.max_samples is not None and len(errors) > self.max_samples:
                    continue  # no need to collect more samples for this

                ref = output.get('output_ref')
                if ref is not None and isinstance(ref, trf_outputs.OutputClassification):
                    all_outputs_maxedout = False
                    found = output['output']
                    truth = output['output_truth']
                    error_indices = np.where(found != truth)[0]

                    # make sure we can link the output class mapping in text
                    classification_mappings = utils.get_classification_mappings(datasets_infos, dataset_name, split_name)
                    if classification_mappings is not None:
                        classification_mapping = classification_mappings.get(ref.classes_name)
                        if classification_mapping is not None:
                            classification_mappings['output'] = classification_mapping

                    # finally, export the errors
                    max_samples = len(error_indices)
                    for sample_id in error_indices[:max_samples]:
                        id = len(errors)
                        sample_output = os.path.join(root, output_name + '_' + split_name + '_s' + str(id))
                        txt_file = sample_output + '.txt'
                        errors.append(sample_id)
                        with open(txt_file, 'w') as f:
                            sample_export.export_sample(
                                output_and_batch_merged,
                                sample_id,
                                sample_output + '_',
                                f,
                                features_to_discard=['output_ref', 'loss'],
                                classification_mappings=classification_mappings)
            if all_outputs_maxedout:
                # all outputs have reached the maximum allowed error: so stop!
                raise StopIteration()  # abort the loop, we have already too many samples

        logger.info('started CallbackExportClassificationErrors.__call__')
        device = options['workflow_options']['device']
        for dataset_name, dataset in datasets.items():
            root = os.path.join(options['workflow_options']['current_logging_directory'], self.dirname, dataset_name)
            utils.create_or_recreate_folder(root)
            
            for split_name, split in dataset.items():
                if self.discard_train:
                    train_name = options['workflow_options']['train_split']
                    if train_name == split_name:
                        continue

                errors = collections.defaultdict(list)
                trainer.eval_loop(device, dataset_name, split_name, split, model, losses[dataset_name],
                                  history=None,
                                  callbacks_per_batch=callbacks_per_batch,
                                  callbacks_per_batch_loss_terms=[functools.partial(callback_per_batch, errors_by_output=errors, root=root, datasets_infos=datasets_infos)])
        logger.info('successfully completed CallbackExportClassificationErrors.__call__!')
