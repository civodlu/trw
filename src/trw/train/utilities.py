import json
import time
import logging
import torch
import shutil
import os
import numpy as np
import collections
import numbers
import datetime
import traceback as traceback_module
import io

logger = logging.getLogger(__name__)


def safe_filename(filename):
    """
    Clean the filename so that it can be used as a valid filename
    """
    return filename.\
        replace('=', ''). \
        replace('\n', ''). \
        replace('/', '_'). \
        replace('\\', '_'). \
        replace('$', ''). \
        replace(';', ''). \
        replace('*', '_')


def log_info(msg):
    """
    Log the message to a log file as info
    :param msg:
    :return:
    """
    logger.debug(msg)


def log_and_print(msg):
    """
    Log the message to a log file as info
    :param msg:
    :return:
    """
    logger.debug(msg)
    print(msg)


def log_console(msg):
    """
    Log the message to the console
    :param msg:
    :return:
    """
    print(msg)


from trw.utils import to_value, recursive_dict_update


def create_or_recreate_folder(path, nb_tries=3, wait_time_between_tries=2.0):
    """
    Check if the path exist. If yes, remove the folder then recreate the folder, else create it

    Args:
        path: the path to create or recreate
        nb_tries: the number of tries to be performed before failure
        wait_time_between_tries: the time to wait before the next try

    Returns:
        ``True`` if successful or ``False`` if failed.
    """
    assert len(path) > 6, 'short path? just as a precaution...'
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

    def try_create():
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            print('[ignored] create_or_recreate_folder error:', str(e))
            return False

    # multiple tries (e.g., for windows if we are using explorer in the current path)
    import threading
    for i in range(nb_tries):
        is_done = try_create()
        if is_done:
            return True
        threading.Event().wait(wait_time_between_tries)  # wait some time for the FS to delete the files
    return False


class time_it:
    """
    Simple decorator to measure the time taken to execute a function
    :param time_name: the name of the function to time, else we will use `fn.__str__()`
    :param log: how to log the timing
    """
    def __init__(self, time_name=None, log=None):
        self.time_name = time_name
        self.log = log

    def __call__(self, fn, *args, **kwargs):
        def fn_decorated(*args, **kwargs):
            start_time = time.time()
            r = fn(*args, **kwargs)
            end_time = time.time()

            if self.time_name is None:
                time_name = fn
            else:
                time_name = self.time_name

            string = 'time={} fn={}'.format(end_time - start_time, time_name)
            if self.log is None:
                print(string)
            else:
                self.log(string)
            return r
        return fn_decorated


def make_unique_colors():
    """
    Return a set of unique and easily distinguishable colors
    :return: a list of RBG colors
    """
    return [
        (79, 105, 198),  # indigo
        (237, 10, 63),  # red
        (175, 227, 19),  # inchworm
        (193, 84, 193),  # fuchsia
        (175, 89, 62),  # brown
        (94, 140, 49),  # maximum green
        (255, 203, 164),  # peach
        (200, 200, 205),  # blue-Gray
        (255, 255, 255),  # white
        (115, 46, 108),  # violet(I)
        (157, 224, 147),  # granny-smith
    ]


def make_unique_colors_f():
    """
    Return a set of unique and easily distinguishable colors
    :return: a list of RBG colors
    """
    return [
        (79 / 255.0, 105 / 255.0, 198 / 255.0),  # indigo
        (237 / 255.0, 10 / 255.0, 63 / 255.0),  # red
        (175 / 255.0, 227 / 255.0, 19 / 255.0),  # inchworm
        (193 / 255.0, 84 / 255.0, 193 / 255.0),  # fuchsia
        (175 / 255.0, 89 / 255.0, 62 / 255.0),  # brown
        (94 / 255.0, 140 / 255.0, 49 / 255.0),  # maximum green
        (255 / 255.0, 203 / 255.0, 164 / 255.0),  # peach
        (200 / 255.0, 200 / 255.0, 205 / 255.0),  # blue-Gray
        (255 / 255.0, 255 / 255.0, 255 / 255.0),  # white
        (115 / 255.0, 46 / 255.0, 108 / 255.0),  # violet(I)
        (157 / 255.0, 224 / 255.0, 147 / 255.0),  # granny-smith
    ]


def get_class_name(mapping, classid):
    classid = int(classid)
    if mapping is None:
        return None
    return mapping['mappinginv'].get(classid)


def get_classification_mappings(datasets_infos, dataset_name, split_name):
    """
        Return the output mappings of a classification output from the datasets infos

        :param datasets_infos: the info of the datasets
        :param dataset_name: the name of the dataset
        :param split_name: the split name
        :param output_name: the output name
        :return: a dictionary {outputs: {'mapping': {name->ID}, 'mappinginv': {ID->name}}}
        """
    if datasets_infos is None or dataset_name is None or split_name is None:
        return None
    dataset_infos = datasets_infos.get(dataset_name)
    if dataset_infos is None:
        return None

    split_datasets_infos = dataset_infos.get(split_name)
    if split_datasets_infos is None:
        return None

    return split_datasets_infos.get('output_mappings')


def get_classification_mapping(datasets_infos, dataset_name, split_name, output_name):
    """
    Return the output mappings of a classification output from the datasets infos

    :param datasets_infos: the info of the datasets
    :param dataset_name: the name of the dataset
    :param split_name: the split name
    :param output_name: the output name
    :return: a dictionary {'mapping': {name->ID}, 'mappinginv': {ID->name}}
    """
    if output_name is None:
        return None
    output_mappings = get_classification_mappings(datasets_infos, dataset_name, split_name)
    if output_mappings is None:
        return None
    return output_mappings.get(output_name)


def set_optimizer_learning_rate(optimizer, learning_rate):
        """
        Set the learning rate of the optimizer to a specific value

        Args:
            optimizer: the optimizer to update
            learning_rate: the learning rate to set

        Returns:
            None
        """

        # manually change the learning rate. References:
        # - https://discuss.pytorch.org/t/change-learning-rate-in-pytorch/14653
        # - https://discuss.pytorch.org/t/adaptive-learning-rate/320/36
        for param_group in optimizer.param_groups:
            # make sure we have an exising learning rate parameters. If not, it means pytorch changed
            # OR the current optimizer is not supported
            assert 'lr' in param_group, 'internally, the optimizer is not using a learning rate!'
            param_group['lr'] = learning_rate


def collate_tensors(values, device, pin_memory=False, non_blocking=False):
    """
    express `values` as a torch.Tensor


    Args:
        values: nd.array or torch.Tensor
        device: the device where to create the torch.Tensor
        pin_memory: if True, pin the memory. Required to be a `Cuda` allocated torch.Tensor

    Returns:
        a torch.Tensor if of type numpy.ndarray else, the input type
    """
    tensor = values
    if isinstance(values, list) and isinstance(values[0], torch.Tensor):
        if len(values) == 1:
            # no need to concatenate if we have only a single element!
            # this usecase is quite common with SequenceAsyncReservoir
            # and taking significant time with large tensors
            tensor = values[0]
        else:
            tensor = torch.cat(values)
    elif isinstance(values, list) and isinstance(values[0], np.ndarray):
        tensor = torch.as_tensor(np.concatenate(values))
    elif isinstance(values, np.ndarray):
        tensor = torch.as_tensor(values)

    elif isinstance(values, list) and isinstance(values[0], list) and isinstance(values[0][0], torch.Tensor):
        # this is from a list of dictionary
        merged = [torch.cat(value) for value in values]
        if len(merged) == 1:
            tensor = merged[0].view([1] + list(merged[0].shape))
        else:
            tensor = torch.stack(merged)
    elif isinstance(values, list) and isinstance(values[0], list) and isinstance(values[0][0], np.ndarray):
        # this is from a list of dictionary
        merged = [torch.as_tensor(np.concatenate(value)) for value in values]
        if len(merged) == 1:
            tensor = merged[0].view([1] + list(merged[0].shape))
        else:
            tensor = torch.stack(merged)
    elif isinstance(values, list) and isinstance(values[0], numbers.Number):
        tensor = torch.as_tensor(np.asarray(values))
    elif isinstance(values, list) and isinstance(values[0], list) and isinstance(values[0][0], numbers.Number):
        tensor = torch.as_tensor(np.concatenate(values))
    elif isinstance(values, list) and isinstance(values[0], list) and isinstance(values[0][0], str):
        r = []
        for r_append in values:
            r += r_append
        tensor = r

    # on torch.Tensor only
    if isinstance(tensor, torch.Tensor):
        if pin_memory:
            tensor.pin_memory()
        if device is not None and tensor.device != device:
            tensor = tensor.to(device, non_blocking=non_blocking)
    else:
        pass
    return tensor


def collate_dicts(batch, device, pin_memory=False, non_blocking=False):
    """
    Default function to collate a dictionary of samples to a dictionary of torch.Tensor

    Args:
        batch: a dictionary of features
        device: the device where to create the torch.Tensor
        pin_memory: if True, pin the memory. Required to be a `CUDA` allocated torch.Tensor

    Returns:
        a dictionary of torch.Tensor
    """
    assert isinstance(batch, collections.Mapping), 'must be a dictionary like!'

    collated = collections.OrderedDict()
    for name, values in batch.items():
        collated[name] = collate_tensors(values=values, device=device, pin_memory=pin_memory, non_blocking=non_blocking)
    return collated


def collate_list_of_dicts(batches, device, pin_memory=False, non_blocking=False):
    """
        Default function to collate a list of dictionary to a dictionary of `torch.Tensor`s

        Args:
            batches: a list of dictionary of features
            device: the device where to create the torch.Tensor
            pin_memory: if True, pin the memory. Required to be a `CUDA` allocated torch.Tensor

        Returns:
            a dictionary of torch.Tensor
        """
    assert isinstance(batches, collections.Sequence), f'must be a list of dictionary! Got={type(batches)}'
    assert isinstance(batches[0], collections.Mapping), f'must be a list of dictionary! Got={type(batches[0])}, str={str(batches[0])}'

    d = collections.OrderedDict()
    for key in batches[0]:
        bs = [b[key] for b in batches]
        bs = collate_tensors(bs, device=device, pin_memory=pin_memory, non_blocking=non_blocking)
        d[key] = bs

    return d


def default_collate_fn(batch, device, pin_memory=False, non_blocking=False):
    """

    Args:
        batch: a dictionary of features or a list of dictionary of features
        device: the device where to create the torch.Tensor
        pin_memory: if True, pin the memory. Required to be a `CUDA` allocated torch.Tensor

    Returns:
        a dictionary of torch.Tensor
    """
    if isinstance(batch, collections.Sequence):
        return collate_list_of_dicts(batch, device, pin_memory, non_blocking)

    if isinstance(batch, collections.Mapping):
        return collate_dicts(batch, device, pin_memory, non_blocking)

    raise NotImplemented()


def transfer_batch_to_device(batch, device, non_blocking=False):
    """
    Transfer the Tensors and numpy arrays to the specified device. Other types will not be moved.

    Args:
        batch: the batch of data to be transferred
        device: the device to move the tensors to
        non_blocking: non blocking memory transfer to GPU

    Returns:
        a batch of data on the specified device
    """

    device_batch = collections.OrderedDict()
    for name, value in batch.items():
        if isinstance(value, np.ndarray):
            # `torch.from_numpy` to keep the same dtype as our input
            device_batch[name] = torch.as_tensor(value).to(device, non_blocking=non_blocking)
        elif isinstance(value, torch.Tensor) and value.device != device:
            device_batch[name] = value.to(device, non_blocking=non_blocking)
        else:
            device_batch[name] = value
    return device_batch


class CleanAddedHooks:
    """
    Context manager that automatically track added hooks on the model and remove them when
    the context is released
    """
    def __init__(self, model):
        self.initial_hooks = {}
        self.model = model
        self.nb_hooks_removed = 0  # record the number of hooks deleted after the context is out of scope

    def __enter__(self):
        self.initial_module_hooks_forward, self.initial_module_hooks_backward = CleanAddedHooks.record_hooks(self.model)
        return self

    def __exit__(self, type, value, traceback):
        def remove_hooks(hooks_initial, hooks_final, is_forward):
            for module, hooks in hooks_final.items():
                if module in hooks_initial:
                    added_hooks = hooks - hooks_initial[module]
                else:
                    added_hooks = hooks

                for hook in added_hooks:
                    if is_forward:
                        self.nb_hooks_removed += 1
                        del module._forward_hooks[hook]
                    else:
                        self.nb_hooks_removed += 1
                        del module._backward_hooks[hook]

        all_hooks_forward, all_hooks_backward = CleanAddedHooks.record_hooks(self.model)
        remove_hooks(self.initial_module_hooks_forward, all_hooks_forward, is_forward=True)
        remove_hooks(self.initial_module_hooks_backward, all_hooks_backward, is_forward=False)

        if traceback is not None:
            io_string = io.StringIO()
            traceback_module.print_tb(traceback, file=io_string)

            print('Exception={}'.format(io_string.getvalue()))
            logger.error('CleanAddedHooks: exception={}'.format(io_string.getvalue()))

        return True

    @staticmethod
    def record_hooks(module_source):
        """
        Record hooks
        Args:
            module_source: the module to track the hooks

        Returns:
            at tuple (forward, backward). `forward` and `backward` are a dictionary of hooks ID by module
        """
        modules_kvp_forward = {}
        modules_kvp_backward = {}
        for module in module_source.modules():
            if len(module._forward_hooks) > 0:
                modules_kvp_forward[module] = set(module._forward_hooks.keys())

            if len(module._backward_hooks) > 0:
                modules_kvp_backward[module] = set(module._backward_hooks.keys())
        return modules_kvp_forward, modules_kvp_backward


def get_device(module, batch=None):
    """
    Return the device of a module. This may be incorrect if we have a module split accross different devices
    """
    try:
        p = next(module.parameters())
        return p.device
    except StopIteration:
        # the model doesn't have parameters!
        pass

    if batch is not None:
        # try to guess the device from the batch
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                return value.device

    # we can't make an appropriate guess, just fail!
    return None


class RuntimeFormatter(logging.Formatter):
    """
    Report the time since this formatter is instantiated
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        return str(datetime.timedelta(seconds=record.created - self.start_time))


def find_default_dataset_and_split_names(datasets, default_dataset_name=None, default_split_name=None, train_split_name=None):
    """
    Return a good choice of dataset name and split name, possibly not the train split.

    Args:
        datasets: the datasets
        default_dataset_name: a possible dataset name. If `None`, find a suitable dataset, if not, the dataset
            must be present
        default_split_name: a possible split name. If `None`, find a suitable split, if not, the dataset
            must be present. if `train_split_name` is specified, the selected split name will be different from `train_split_name`
        train_split_name: if not `None`, exclude the train split

    Returns:
        a tuple (dataset_name, split_name)
    """
    if default_dataset_name is None:
        default_dataset_name = next(iter(datasets))
    else:
        if default_dataset_name not in datasets:
            return None, None

    if default_split_name is None:
        available_splits = datasets[default_dataset_name].keys()
        for split_name in available_splits:
            if split_name != train_split_name:
                default_split_name = split_name
                break
    else:
        if default_split_name not in datasets[default_dataset_name]:
            return None, None

    return default_dataset_name, default_split_name


def make_triplet_indices(targets):
    """
    Make random index triplets (anchor, positive, negative) such that ``anchor`` and ``positive``
        belong to the same target while ``negative`` belongs to a different target

    Args:
        targets: a 1D integral tensor in range [0..C]

    Returns:
        a tuple of indices (samples, samples_positive, samples_negative)
    """
    # group samples by class
    samples_by_class = collections.defaultdict(list)
    targets = to_value(targets)
    for index, c in enumerate(targets):
        samples_by_class[c].append(index)

    # create the (sample, sample+, sample-) groups
    samples_all = []
    samples_positive_all = []
    samples_negative_all = []
    for c, c_indexes in samples_by_class.items():
        samples = c_indexes.copy()
        samples_positive = c_indexes
        np.random.shuffle(c_indexes)

        other = [idx for cc, idx in samples_by_class.items() if cc != c]
        other = np.concatenate(other)

        # sample with replacement in case the ``negative`` sample are less
        # than the ``positive`` samples
        samples_negative = np.random.choice(other, len(samples))

        samples_all.append(samples)
        samples_positive_all.append(samples_positive)
        samples_negative_all.append(samples_negative)

    samples_all = np.concatenate(samples_all)
    samples_positive_all = np.concatenate(samples_positive_all)
    samples_negative_all = np.concatenate(samples_negative_all)
    min_samples = min(len(samples_all), len(samples_negative_all))
    return samples_all[:min_samples], samples_positive_all[:min_samples], samples_negative_all[:min_samples]


def make_pair_indices(targets, same_target_ratio=0.5):
    """
    Make random indices of pairs of samples that belongs or not to the same target.

    Args:
        same_target_ratio: specify the ratio of same target to be generated for sample pairs
        targets: a 1D integral tensor in range [0..C] to be used to group the samples
            into same or different target

    Returns:
        a tuple with (samples_0 indices, samples_1 indices, same_target)
    """
    # group samples by class
    samples_by_class = collections.defaultdict(list)
    classes = to_value(targets)
    for index, c in enumerate(classes):
        samples_by_class[c].append(index)
    samples_by_class = {name: np.asarray(value) for name, value in samples_by_class.items()}

    # create the (sample, sample+, sample-) groups
    samples_0 = []
    samples_1 = []
    same_target = []
    for c, c_indexes in samples_by_class.items():
        samples = c_indexes.copy()
        np.random.shuffle(c_indexes)
        nb_same_targets = int(same_target_ratio * len(c_indexes))

        other = [idx for cc, idx in samples_by_class.items() if cc != c]
        other = np.concatenate(other)
        np.random.shuffle(other)

        samples_0.append(samples)
        samples_positive = c_indexes[:nb_same_targets]
        same_target += [1] * len(samples_positive)
        # expect to have more negative than positive, so for the negative
        # pick the remaining
        samples_negative = other[:len(c_indexes) - len(samples_positive)]
        same_target += [0] * len(samples_negative)
        samples_1.append(np.concatenate((samples_positive, samples_negative)))

    # in case the assumption was wrong (we, in fact, have more positive than negative)
    # shorten the batch
    samples_0 = samples_0[:len(samples_1)]

    return np.concatenate(samples_0), np.concatenate(samples_1), np.asarray(same_target)


def update_json_config(path_to_json, config_update):
    """
    Update a JSON document stored on a local drive.

    Args:
        path_to_json: the path to the local JSON configuration
        config_update: a possibly nested dictionary

    """
    if os.path.exists(path_to_json):
        with open(path_to_json, 'r') as f:
            text = f.read()
        config = json.loads(text)
    else:
        config = collections.OrderedDict()

    recursive_dict_update(config, config_update)

    json_str = json.dumps(config, indent=3)
    with open(path_to_json, 'w') as f:
        f.write(json_str)


def prepare_loss_terms(outputs, batch, is_training):
    """
    Return the loss_terms for the given outputs
    """
    from trw.train import Output

    loss_terms = collections.OrderedDict()
    for output_name, output in outputs.items():
        assert isinstance(output, Output), f'output must be a `trw.train.Output`' \
                                                       f' instance. Got={type(output)}'
        loss_term = output.evaluate_batch(batch, is_training)
        if loss_term is not None:
            loss_terms[output_name] = loss_term
    return loss_terms


def default_sum_all_losses(dataset_name, batch, loss_terms):
    """
    Default loss is the sum of all loss terms
    """
    sum_losses = 0.0
    for name, loss_term in loss_terms.items():
        loss = loss_term.get('loss')
        if loss is not None:
            # if the loss term doesn't contain a `loss` attribute, it means
            # this is not used during optimization (e.g., embedding output)
            sum_losses += loss
    return sum_losses
