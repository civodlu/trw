import collections
import torch
import numpy as np


def len_batch(batch):
    """

    Args:
        batch: a data split or a `collections.Sequence`

    Returns:
        the number of elements within a data split
    """
    if isinstance(batch, (collections.Sequence, torch.Tensor)):
        return len(batch)

    assert isinstance(batch, collections.Mapping), 'Must be a dict-like structure! got={}'.format(type(batch))

    for name, values in batch.items():
        if isinstance(values, (list, tuple)):
            return len(values)
        if isinstance(values, torch.Tensor) and len(values.shape) != 0:
            return values.shape[0]
        if isinstance(values, np.ndarray) and len(values.shape) != 0:
            return values.shape[0]
    return 0


def to_value(v):
    """
    Convert where appropriate from tensors to numpy arrays

    Args:
        v: an object. If ``torch.Tensor``, the tensor will be converted to a numpy
            array. Else returns the original ``v``

    Returns:
        ``torch.Tensor`` as numpy arrays. Any other type will be left unchanged
    """
    if isinstance(v, torch.Tensor):
        return v.cpu().data.numpy()
    return v


def get_batch_n(split, nb_samples, indices, transforms, use_advanced_indexing):
    """
    Collect the split indices given and apply a series of transformations

    Args:
        nb_samples: the total number of samples of split
        split: a mapping of `np.ndarray` or `torch.Tensor`
        indices: a list of indices as numpy array
        transforms: a transformation or list of transformations or None
        use_advanced_indexing: if True, use the advanced indexing mechanism else use a simple list (original data is referenced)
            advanced indexing is typically faster for small objects, however for large objects (e.g., 3D data)
            the advanced indexing makes a copy of the data making it very slow.

    Returns:
        a split with the indices provided
    """
    data = {}
    for split_name, split_data in split.items():
        if isinstance(split_data, (torch.Tensor, np.ndarray)) and len(split_data) == nb_samples:
            # here we prefer [split_data[i] for i in indices] over split_data[indices]
            # this is because split_data[indices] will make a deep copy of the data which may be time consuming
            # for large data

            # TODO for small data batch: prefer the indexing, for large data, prefer the referencing
            if use_advanced_indexing:
                split_data = split_data[indices]
            else:
                split_data = [[split_data[i]] for i in indices]
        if isinstance(split_data, list) and len(split_data) == nb_samples:
            split_data = [split_data[i] for i in indices]

        data[split_name] = split_data

    if transforms is None:
        # do nothing: there is no transform
        pass
    elif isinstance(transforms, collections.Sequence):
        # we have a list of transforms, apply each one of them
        for transform in transforms:
            data = transform(data)
    else:
        # anything else should be a functor
        data = transforms(data)

    return data


def safe_lookup(dictionary, *keys, default=None):
    """
    Recursively access nested dictionaries

    Args:
        dictionary: nested dictionary
        *keys: the keys to access within the nested dictionaries
        default: the default value if dictionary is ``None`` or it doesn't contain
            the keys

    Returns:
        None if we can't access to all the keys, else dictionary[key_0][key_1][...][key_n]
    """
    if dictionary is None:
        return default

    for key in keys:
        dictionary = dictionary.get(key)
        if dictionary is None:
            return default

    return dictionary


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
                recursive_dict_update(values, updated_values)
            else:
                # the value is not a dictionary, we can update directly its value
                dict[updated_name] = values
