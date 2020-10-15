import numbers

import collections
import torch
import trw
import trw.utils

import numpy as np
import os

from trw.reporting.table_sqlite import SQLITE_TYPE_PATTERN, TableStream
from PIL import Image


def as_rgb_image(value):
    """
    Try interpreting the value as an image. (e.g., 2D, RGB) and return a RGB image
    :param value: an array of shape (y, x), (1, y, x), (3, y, x)
    :return: return a (3, y, x) array
    """
    if isinstance(value, np.ndarray):
        if len(value.shape) > 3:
            value = np.squeeze(value)

        if len(value.shape) == 2:
            # gray scale 2d image
            value = np.reshape(value, (1, value.shape[0], value.shape[1]))
            if len(value.shape) == 3:
                value = np.concatenate((value, value, value), 0)
                return value

        if len(value.shape) == 3:
            if value.shape[0] == 1:
                # 3 components but still grayscale
                value = np.concatenate((value, value, value), 0)
                return value

            # RGB image
            if value.shape[0] == 3:
                return value
    return None


def as_image_ui8(image, min_value=None, max_value=None):
    """
    Rescale the image to fit in [0..255] range.

    Image min will be mapped to 0 and max to 255. Values in this range are interpolated
    :param image: a RGB float image
    :return: a RGB unsigned char image
    """
    assert len(image.shape) == 3
    assert image.shape[0] == 3

    if min_value is None:
        min_value = np.min(image)

    if max_value is None:
        max_value = np.max(image)

    if max_value != min_value:
        image = (image - min_value) / (max_value - min_value) * 255
        image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


def export_image(image, path):
    """
    Export an image

    :param image: a RGB image (float or ui8) with format (channels, height, width)
    :param path: where to write the image
    :return:
    """
    assert len(image.shape) == 3

    if image.shape[0] == 3:
        # reorder the components to (width, height, channels)
        image = image.transpose((1, 2, 0))

    assert image.shape[2] == 3

    img = Image.fromarray(image)
    img.save(path)


def export_as_image(batch, feature_name, name, sample_id, export_root, feature_attributes):
    samples = trw.utils.to_value(batch[feature_name])
    # an image MUST have a filter component, else we could confuse
    # if as a 2D array that we want to export in a text file
    if not isinstance(samples, list) or not isinstance(samples[sample_id], np.ndarray) or len(samples[sample_id].shape) < 3:
        return False, None
    rgb = as_rgb_image(samples[sample_id])
    if rgb is None:
        return False, None

    if feature_attributes is None or 'min_value' not in feature_attributes:
        # here we normalize the min and max by the batch min/max. If the batch
        # is large enough, it should be ok to compare images between batches
        # we need to keep track of the min/max across `export_as_image` calls
        # since we replace the array values by the name of the exported array
        np_arrays = [i.ravel() for i in samples if isinstance(i, np.ndarray)]
        min_value = min([min(i) for i in np_arrays])  # handle variably sized arrays
        max_value = max([max(i) for i in np_arrays])
        feature_attributes = {
            'min_value': min_value,
            'max_value': max_value,
        }

    batch_min = feature_attributes['min_value']
    batch_max = feature_attributes['max_value']
    ui8 = as_image_ui8(rgb, min_value=batch_min, max_value=batch_max)
    if ui8 is None:
        return False, None
    path = os.path.join(export_root, 'static', name + '.png')
    assert not os.path.exists(path), f'path={path} collided with an existing file! Make sure the ``name``' \
                                     f'has a unique pattern!'
    export_image(ui8, path)

    batch[feature_name][sample_id] = os.path.join('static', name + '.png')

    # we need to record the feature type
    feature_type_name = feature_name + SQLITE_TYPE_PATTERN
    feature_type_name_values = batch.get(feature_type_name)
    if feature_type_name_values is None:
        feature_type_name_values = [None] * len(samples)
        batch[feature_type_name] = feature_type_name_values

    feature_type_name_values[sample_id] = 'BLOB_IMAGE_PNG'
    return True, feature_attributes


def export_as_npy(batch, feature_name, name, sample_id, export_root, feature_attributes):
    samples = trw.utils.to_value(batch[feature_name])

    if isinstance(samples, list) and isinstance(samples[sample_id], np.ndarray):
        sample_shape = samples[sample_id].shape
        if len(sample_shape) == 0:
            # just a number, export as text
            return False, None

        # if 0D or 1D, we want this exported as text
        if len(sample_shape) <= 1 and sample_shape[0] <= 1:
            return False, None

        path = os.path.join(export_root, 'static', name + '.npy')
        assert not os.path.exists(path), f'path={path} collided with an existing file! Make sure the ``{name}``' \
                                         f'has a unique pattern!'
        np.save(path, samples[sample_id])
        batch[feature_name][sample_id] = os.path.join('static', name + '.npy')

        # we need to record the feature type
        feature_type_name = feature_name + SQLITE_TYPE_PATTERN
        feature_type_name_values = batch.get(feature_type_name)
        if feature_type_name_values is None:
            feature_type_name_values = [None] * len(samples)
            batch[feature_type_name] = feature_type_name_values

        feature_type_name_values[sample_id] = 'BLOB_NUMPY'
        return True, None
    return False, None


def export_as_text(batch, feature_name, name, sample_id, export_root, feature_attributes):
    samples = trw.utils.to_value(batch[feature_name])
    if isinstance(samples, list) and isinstance(
            samples[sample_id],
            (np.ndarray, torch.Tensor, list, collections.Mapping, numbers.Number)):
        samples[sample_id] = str(samples[sample_id])
    return True, None


def export_sample(
        export_root,
        table_stream,
        base_name,
        batch,
        sample_ids=None,
        export_fns=[export_as_image, export_as_npy, export_as_text],
        name_expansions=['epoch', 'batch', 'split', 'dataset'],
        ):
    r"""

    Export samples to a SQL database with large binary objects on he local drive
    with the following schema:
        {export_root}/{table_stream.table_name}/{base_name}_{**name_expansions}

    Args:
        export_root: the root from which the data will be exported relatively
        table_stream: the SQL table where the data will be stored
        base_name: the basename for the features to be exported on the local drive
        batch: a key/value store
        sample_ids: the index of the samples to be exported
        export_fns: functions to be run to special features such as images, large numpy arrays
        name_expansions: if the name is present in the batch, the sample name to be exported on the drive
            will be expanded ``base_name``_{name_expansion}_{batch[name_expansion]}.The purpose is
            to make sure the name is unique
    """
    assert isinstance(table_stream, TableStream)
    batch_size = trw.utils.len_batch(batch)
    base_name = trw.utils.safe_filename(base_name)

    # transform the first dim of numpy arrays as lists
    batch_list = collections.OrderedDict()
    for name, value in batch.items():
        value = trw.utils.to_value(value)
        if isinstance(value, np.ndarray):
            # remove the first numpy dimension and replace it as a list
            # this is done so that we can replace a numpy array (e.g., image, large array) to
            # a named array saved on local drive
            if len(value.shape) == 0:
                value = [float(value)] * batch_size
            else:
                value = list(value)
        elif isinstance(value, (numbers.Number, str)):
            # make sure we have one value per sample, so expand the 0-d values
            value = [value] * batch_size
        elif isinstance(value, list):
            # if a list, make sure we make a copy so that the original batch is not modified
            value = list(value)
        elif value is None:
            # discard any ``None``
            continue
        batch_list[name] = value

    if sample_ids is None:
        sample_ids = range(batch_size)

    features_attributes = collections.defaultdict(lambda: None)
    batch_keys = list(batch_list.keys())  # in case we create ne keys in the export functions
    for id in sample_ids:
        # expand the basename based on the batch attributes
        name_expansion = []
        for e in name_expansions:
            v = batch_list.get(e)
            if v is not None:
                name_expansion.append(e)
                name_expansion.append(v[id])
        if len(name_expansion) > 0:
            name_expansion = '_' + '_'.join(name_expansion)
        else:
            name_expansion = ''

        # process the features. Possibly export large features to local drive
        # to make the database lightweight
        for feature_name in batch_keys:
            feature_attributes = features_attributes[feature_name]
            sample_name = f'{table_stream.table_name}/{base_name}{name_expansion}_{feature_name}_{id}'

            for export_fn in export_fns:
                exported, feature_attributes = export_fn(
                    batch_list,
                    feature_name,
                    sample_name,
                    id,
                    export_root,
                    feature_attributes)
                if exported:
                    features_attributes[feature_name] = feature_attributes
                    break

        # final batch_export
        one_sample_batch = trw.utils.get_batch_n(
            batch_list,
            batch_size,
            [id],
            transforms=None,
            use_advanced_indexing=True)
        table_stream.insert(one_sample_batch)
