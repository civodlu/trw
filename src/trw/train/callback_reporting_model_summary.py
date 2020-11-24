import collections
import logging
import os

import torch
import numpy as np
import trw
import trw.utils
from trw import reporting
from trw.utils import collect_hierarchical_module_name
from trw.reporting.table_sqlite import table_truncate
from trw.train import create_or_recreate_folder, callback, find_default_dataset_and_split_names, utilities
from trw.train.utilities import update_json_config
from trw.train.data_parallel_extented import DataParallelExtended
from trw.train.outputs_trw import Output

logger = logging.getLogger(__name__)


def input_shape(i, root=True):
    if isinstance(i, collections.Mapping):
        return 'Dict'

    if i is None:
        return 'None'

    elif isinstance(i, torch.Tensor):
        shape = tuple(i.shape)

        if root:
            # we want to have all outputs/inputs in a consistent format (list of shapes)
            return [shape]
        return shape

    if isinstance(i, (list, tuple)):
        shapes = []
        for n in i:
            # make sure the shapes are flattened
            sub_shapes = input_shape(n, root=False)
            if isinstance(sub_shapes, list):
                shapes += sub_shapes
            else:
                shapes.append(sub_shapes)
        return shapes

    if isinstance(i, Output):
        return i.output.shape

    # unknown node (e.g., custom output in a model)
    return 'UKN'


def model_summary_base(model, batch):
    def register_hook(module):
        def hook(module, input, output):
            nonlocal parameters_counted
            if module in summary:
                return
            summary[module] = collections.OrderedDict()
            summary[module]['input_shape'] = input_shape(input)
            summary[module]['output_shape'] = input_shape(output)

            total_trainable_params = 0  # ALL parameters of the layer (including sub-module parameters)
            params = 0  # if we have recursion, these are the unique parameters
            for p in module.parameters():
                if p.requires_grad:
                    total_trainable_params += np.prod(np.asarray(p.shape))

                if p not in parameters_counted:
                    # make sure there is double counted parameters!
                    parameters_counted.add(p)
                    if p.requires_grad:
                        params += np.prod(np.asarray(p.shape))

            summary[module]["nb_params"] = params
            summary[module]["total_trainable_params"] = total_trainable_params

        hooks.append(module.register_forward_hook(hook))

    parameters_counted = set()
    summary = collections.OrderedDict()
    hooks = []

    if isinstance(model, (torch.nn.DataParallel, DataParallelExtended)):
        # get the underlying module only. `DataParallel` will replicate the module on the different devices
        model = model.module

    model.apply(register_hook)
    model(batch)

    for h in hooks:
        h.remove()

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer, values in summary.items():
        total_params += values["nb_params"]
        outputs = values["output_shape"]
        if isinstance(outputs, torch.Size):
            outputs = [outputs]
        if not isinstance(outputs, str):
            for output in outputs:
                if not isinstance(output, str):
                    total_output += np.prod(output[1:])  # remove the batch size

        trainable_params += values["nb_params"]

    # assume 4 bytes/number (float on cuda)
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(trw.utils.to_value(total_params) * 4. / (1024 ** 2.))
    return summary, total_output_size, total_params_size, total_params, trainable_params


def export_table(options, table_name, batch, table_role, clear_existing_data, base_name='', table_preamble='', commit=True):
    sql_database = options['workflow_options']['sql_database']
    root = os.path.join(options['workflow_options']['current_logging_directory'], 'static', table_name)

    if clear_existing_data:
        cursor = sql_database.cursor()
        table_truncate(cursor, table_name)
        sql_database.commit()

        # also remove the binary/image store
        root = os.path.dirname(options['workflow_options']['sql_database_path'])
        create_or_recreate_folder(os.path.join(root, 'static', table_name))

    sql_table = reporting.TableStream(
        cursor=sql_database.cursor(),
        table_name=table_name,
        table_role=table_role,
        table_preamble=table_preamble)

    logger.info(f'export table={table_name} started...')
    reporting.export_sample(
        root,
        sql_table,
        base_name=base_name,
        batch=batch,
        name_expansions=[],  # we already expanded in the basename!
    )

    if commit:
        sql_database.commit()

    logger.info(f'export table done!')


def html_list(items, header=None):
    es = []
    for i in items:
        es.append(f'<li>{i}</li>')
    v = f"<ul>{''.join(es)}</ul>"
    if header is not None:
        v = f'{header}:<br>{v}'
    return v


class CallbackReportingModelSummary(callback.Callback):
    def __init__(self, dataset_name=None, split_name=None):
        self.split_name = split_name
        self.dataset_name = dataset_name
        self.reporting_config_exported = False

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):
        logger.info('CallbackReportingModelSummary exporting model...')
        if self.split_name is None or self.dataset_name is None:
            self.dataset_name, self.split_name = find_default_dataset_and_split_names(
                datasets,
                default_dataset_name=self.dataset_name,
                default_split_name=self.split_name)

            if self.split_name is None or self.dataset_name is None:
                # no suitable dataset name
                return

        table_name = 'model_summary'
        if not self.reporting_config_exported:
            self.reporting_config_exported = True
            config_path = options['workflow_options']['sql_database_view_path']
            update_json_config(config_path, {
                table_name: {
                    'default': {
                        'with_column_title_rotation': '0',
                    }
                }
            })

        batch = next(iter(datasets[self.dataset_name][self.split_name]))
        batch['split_name'] = self.split_name
        device = options['workflow_options']['device']
        batch = utilities.transfer_batch_to_device(batch, device=device)
        summary, total_output_size, total_params_size, total_params, trainable_params = model_summary_base(
            model,
            batch)
        module_to_name = collect_hierarchical_module_name(type(model).__name__, model)

        layer_name = []
        input_shape = []
        output_shape = []
        nb_params = []
        nb_trainable_params = []
        for module, values in summary.items():
            module_name = module_to_name.get(module)
            if module_name is None:
                module_name = str(module)

            layer_name.append(module_name)
            input_shape.append(str(values['input_shape']))
            output_shape.append(str(values['output_shape']))
            nb_params.append(str(values['nb_params']))
            nb_trainable_params.append(str(values['total_trainable_params']))

        batch = collections.OrderedDict([
            ('layer name', layer_name),
            ('input_shape', input_shape),
            ('output_shape', output_shape),
            ('parameters', nb_params),
            ('trainable parameters', nb_trainable_params),
        ])

        preamble = html_list([
            f'Total parameters: {total_params / 1000000:.2f}M',
            f'Trainable parameters: {trainable_params / 1000000:.2f}M',
            f'Non-trainable parameters: {(total_params - trainable_params)}',
            f'Forward/backward pass size: {total_output_size:.2f} MB',
            f'Params size: {total_params_size:.2f} MB'
        ], header='Model infos')

        export_table(options, table_name,  batch, table_role='data_tabular', clear_existing_data=True,
                     table_preamble=preamble)

        logger.info('CallbackReportingModelSummary exporting model done!')
