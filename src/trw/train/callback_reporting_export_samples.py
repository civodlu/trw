import collections
import json
import os
import functools
import logging
from trw import reporting
from trw.reporting import to_value
from trw.train import callback
from trw.train import trainer
from trw.train import utilities
from trw.train import sample_export
from trw.train import outputs as outputs_trw

logger = logging.getLogger(__name__)


def expand_classification_mapping(batch, loss_term_name, loss_term, classification_mappings, suffix='_str'):
    """
    Expand as string the class name for the classification outputs

    Args:
        batch:
        loss_term:
        classification_mappings: classification_mappings: a nested dict recording the class name/value
            associated with a set of ``output_name``

            {``output_name``:
                {'mapping': {name, value}},
                {'mappinginv': {value, name}}
            }

        suffix: the suffix appended to output or target name
    """
    output_ref = loss_term.get('output_ref')
    if isinstance(output_ref, outputs_trw.OutputClassification):
        target_name = output_ref.classes_name
        if target_name is not None:
            mapping = classification_mappings.get(target_name)
            if mapping is not None:
                output = to_value(loss_term['output'])
                if len(output.shape) == 1:
                    output_str = [utilities.get_class_name(mapping, o) for o in output]
                    batch[loss_term_name + suffix] = output_str

                    # if we record the loss term output, record also the
                    # target name as string.
                    target_name_str = target_name + suffix
                    if target_name_str not in batch:
                        target_values = batch.get(target_name)
                        if target_values is not None and len(target_values.shape) == 1:
                            target_values = [utilities.get_class_name(mapping, o) for o in target_values]
                            batch[target_name_str] = target_values


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
        epoch,
        sql_table,
        format):
    # process the exclusion
    if dataset_name in dataset_exclusions:
        raise StopIteration()
    if split_name in split_exclusions:
        raise StopIteration()

    # copy to the current batch the specified loss terms
    classification_mappings = utilities.get_classification_mappings(datasets_infos, dataset_name, split_name)
    for loss_term_name, loss_term in loss_terms.items():
        for loss_term_inclusion in loss_terms_inclusion:
            if loss_term_inclusion in loss_term:
                batch[f'term_{loss_term_name}_{loss_term_inclusion}'] = loss_term[loss_term_inclusion]
                expand_classification_mapping(batch, loss_term_name, loss_term, classification_mappings)

    for feature_exclusion in feature_exclusions:
        if feature_exclusion in batch:
            del batch[feature_exclusion]

    # force recording of epoch
    batch['epoch'] = epoch

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
        name = format.format(dataset_name=dataset_name, split_name=split_name, id=id, epoch=epoch)
        reporting.export_sample(
            root,
            sql_table,
            base_name=name,
            batch=batch,
            sample_ids=[n],
            name_expansions=[],  # we already expanded in the basename!
        )


def recursive_dict_update(dict, dict_update):
    """
    This adds any missing element from ``dict_update`` to ``dict``, while keeping any key not
        present in ``dict_update``

    Args:
        dict: the dictionary to be updated
        dict_update: the updated values
    """
    for updated_name, updated_values in dict_update.items():
        if updated_name not in dict:
            # simply add the missing name
            dict[updated_name] = updated_values
        else:
            values = dict[updated_name]
            if isinstance(values, collections.Mapping):
                # it is a dictionary. This needs to be recursively
                # updated so that we don't remove values in the existing
                # dictionary ``dict``
                recusive_dict_update(values, updated_values)
            else:
                # the value is not a dictionary, we can update directly its value
                dict[updated_name] = values


def update_json_config(path_to_json, config_update):
    """
    Update a JSON document stored on locally.

    Args:
        path_to_json: the path to the local JSON configuration
        config_update: a possibly nested dictionary

    """
    if os.path.exists(path_to_json):
        with open(path_to_json, 'r') as f:
            config = json.loads(f)
    else:
        config = collections.OrderedDict()

    recursive_dict_update(config, config_update)

    json_str = json.dumps(config, indent=3)
    with open(path_to_json, 'w') as f:
        f.write(json_str)


class CallbackReportingExportSamples(callback.Callback):
    def __init__(
            self,
            max_samples=20,
            table_name='samples',
            loss_terms_inclusion=None,
            feature_exclusions=None,
            dataset_exclusions=None,
            split_exclusions=None,
            format='{dataset_name}_{split_name}_s{id}_e{epoch}',
            reporting_config_keep_last_n_rows=None,
            reporting_config_subsampling_factor=1.0
    ):
        """
        Export random samples from our datasets

        Just for sanity check, it is always a good idea to make sure our data is loaded and processed
        as expected.

        :param max_samples: the maximum number of samples to be exported
        :param table_name: the root of the export directory
        :param loss_terms_inclusion: specifies what output name from each loss term will be exported. if None, defaults to ['output']
        :param feature_exclusions: specifies what feature should be excluded from the export
        :param split_exclusions: specifies what split should be excluded from the export
        :param dataset_exclusions: specifies what dataset should be excluded from the export
        :param format: the format of the files exported. Sometimes need evolution by epoch, other time we may want
            samples by epoch so make this configurable
        :param reporting_config_keep_last_n_rows: Only visualize the last ``reporting_config_keep_last_n_rows``
            samples. Prior samples are discarded. This parameter will be added to the reporting configuration.
        :param reporting_config_subsampling_factor: Specifies how the data is sub-sampled. Must be in range [0..1]
            or ``None``. This parameter will be added to the reporting configuration.
        """

        self.format = format
        self.max_samples = max_samples
        self.table_name = table_name
        if loss_terms_inclusion is None:
            self.loss_terms_inclusion = ['output', 'output_raw', 'loss']
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

        # record the viewing configuration
        self.reporting_config_exported = False
        self.reporting_config_keep_last_n_rows = reporting_config_keep_last_n_rows
        self.reporting_config_subsampling_factor = reporting_config_subsampling_factor

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):

        logger.info('started CallbackExportSamples.__call__')
        device = options['workflow_options']['device']

        if not self.reporting_config_exported:
            # export how the samples should be displayed by the reporting
            config_path = options['workflow_options']['sql_database_view_path']
            update_json_config(config_path, {
                self.table_name: {
                    'data': {
                        'keep_last_n_rows': self. reporting_config_keep_last_n_rows,
                        'subsampling_factor': self.reporting_config_subsampling_factor
                    }
                }
            })
            self.reporting_config_exported = True

        sql_database = options['workflow_options']['sql_database']
        sql_table = reporting.TableStream(
            cursor=sql_database.cursor(),
            table_name=self.table_name,
            table_role='data_samples')

        for dataset_name, dataset in datasets.items():
            root = os.path.join(options['workflow_options']['current_logging_directory'], 'static', self.table_name)
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
                                          root=options['workflow_options']['current_logging_directory'],
                                          datasets_infos=datasets_infos,
                                          loss_terms_inclusion=self.loss_terms_inclusion,
                                          feature_exclusions=self.feature_exclusions,
                                          dataset_exclusions=self.dataset_exclusions,
                                          split_exclusions=self.split_exclusions,
                                          exported_cases=exported_cases,
                                          max_samples=self.max_samples,
                                          epoch=len(history),
                                          sql_table=sql_table,
                                          format=self.format
                                      )])

        sql_database.commit()
        logger.info('successfully completed CallbackExportSamples.__call__!')
