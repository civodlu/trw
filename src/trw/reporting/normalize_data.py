import collections
import os

import numpy as np
from trw.reporting import safe_lookup, len_batch
from trw.reporting.data_category import DataCategory
from trw.reporting.table_sqlite import get_data_types_and_clean_data


def normalize_data(options, data, table_name):
    """
    Normalize, subsample and categorize the data

    The following operations are performed:
    - convert to numpy arrays
    - removed type column and return a specific `type` dictionary
    - normalize the path according to deployment: if static html, nothing to do
        if deployed, we MUST add the `app name` (the top folder name containing the SQL DB)
        as root
    - categorize the data as continuous, discrete unordered, discrete ordered, other
    - recover the DB type (string) from data values (e.g., float or int)
    """
    d = collections.OrderedDict()
    for name, values in data.items():
        d[name] = np.asarray(values)
    types = get_data_types_and_clean_data(d)
    type_categories = {}

    # handle the column removal: maybe they are not useful or maybe the data
    # can't be parsed
    remove_columns = safe_lookup(options.config, table_name, 'data', 'remove_columns', default=[])
    if remove_columns is not None:
        for n in remove_columns:
            if n in data:
                del d[n]
            if n in types:
                del types[n]

    subsampling_factor = safe_lookup(
        options.config,
        table_name,
        'data',
        'subsampling_factor',
        default='1.0')
    subsampling_factor = float(subsampling_factor)
    assert subsampling_factor <= 1.0, 'sub-sampling factor must be <= 1.0'
    if subsampling_factor != 1.0:
        nb_samples = int(len_batch(d) * subsampling_factor)
        d = {
            name: values[:nb_samples] for name, values in d.items()
        }

    if options.embedded:
        appname = os.path.basename(os.path.dirname(options.db_root))
        for n, t in types.items():
            if 'BLOB_' in t:
                # it is a blob, meaning that it was saved on the local HD
                d[n] = list(np.core.defchararray.add(np.asarray([f'{appname}/']), d[n]))

            if 'BLOB_IMAGE' in t:
                type_categories[n] = DataCategory.Other

    # load the numpy arrays
    for name, t in list(types.items()):
        if 'BLOB_NUMPY' in t:
            loaded_np = []
            root = os.path.join(os.path.dirname(options.db_root), '..')
            for relative_path in d[name]:
                path = os.path.join(root, relative_path)
                loaded_np.append(np.load(path))

            # expand the array if satisfying the criteria
            array = np.asarray(loaded_np)
            if len(array.shape) == 2 and array.shape[1] <= options.data.unpack_numpy_arrays_with_less_than_x_columns:
                for n in range(array.shape[1]):
                    name_expanded = name + f'_{n}'
                    value_expanded = array[:, n]
                    d[name_expanded] = value_expanded
                    type_categories[name_expanded] = DataCategory.from_numpy_array(value_expanded)
                del d[name]
                del types[name]
            else:
                d[name] = loaded_np
                type_categories[name] = DataCategory.Other  # dimension too high

    # the DB converted all values to string and have lost the original type. Try to
    # revert back the type from the column values
    for name in list(d.keys()):
        t = d[name]
        if name not in type_categories:
            try:
                t_np = np.asarray(t, dtype=np.int)  # if e have a float, it will raise exception
                type_categories[name] = DataCategory.DiscreteOrdered
                d[name] = list(t_np)
                continue  # success, go to next item
            except ValueError:
                pass

            try:
                t_np = np.asarray(t, dtype=np.float32)
                type_categories[name] = DataCategory.Continuous
                d[name] = list(t_np)
                continue  # success, go to next item
            except ValueError:
                type_categories[name] = DataCategory.Other

            try:
                _ = np.asarray(t, dtype=np.str)  # test data conversion
                type_categories[name] = DataCategory.DiscreteUnordered
                continue  # success, go to next item
            except ValueError:
                type_categories[name] = DataCategory.Other

    return d, types, type_categories
