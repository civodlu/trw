import functools
import torch
import collections


def create_scheduler_step_lr(optimizer, step_size=30, gamma=0.1):
    """
    Create a learning rate scheduler. Every `step_size`, the learning late will be multiplied by `gamma`

    Args:
        optimizer: the optimizer
        step_size: every number of epochs composing one step. Each step the learning rate will be decreased
        gamma: apply this factor to the learning rate every time it is adjusted

    Returns:
        a learning rate scheduler
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def create_optimizers_fn(datasets, model, optimizer_fn, scheduler_fn=None):
    """
    Create an optimizer and scheduler

    Note:
        if model is an instance of`ModuleDict`, then the optimizer will only consider the parameters
        `model[dataset_name].parameters()` else `model.parameters()`

    Args:
        datasets: a dictionary of dataset
        model: the model. Should be a `Module` or a `ModuleDict`
        optimizer_fn: the functor to instantiate the optimizer
        scheduler_fn: the functor to instantiate the scheduler. May be None, in that case
            there will be no scheduler

    Returns:
        a dict of optimizers, one per dataset
    """

    schedulers = None
    if scheduler_fn is not None:
        schedulers = collections.OrderedDict()
    optimizers = collections.OrderedDict()
    for dataset_name in datasets.keys():
        if isinstance(model, torch.nn.ModuleDict):
            # this is a collection of model. Assumed we have a different model
            # per dataset to be optimized
            sub_model = model[dataset_name]
            optimizer = optimizer_fn(sub_model.parameters())
        else:
            optimizer = optimizer_fn(model.parameters())

        optimizers[dataset_name] = optimizer

        if schedulers is not None:
            scheduler = scheduler_fn(optimizer)
            schedulers[dataset_name] = scheduler

    return optimizers, schedulers


def create_adam_optimizers_fn(datasets, model, learning_rate, weight_decay=0, scheduler_fn=None):
    """
    Create an ADAM optimizer for each of the dataset with optional scheduler

    Args:
        datasets: a dictionary of dataset
        model: a model to optimize
        learning_rate: the initial learning rate
        weight_decay: the weight decay
        scheduler_fn: a scheduler, or `None`

    Returns:
        An optimizer
    """
    optimizer_fn = functools.partial(torch.optim.Adam, lr=learning_rate, weight_decay=weight_decay)
    return create_optimizers_fn(datasets, model, optimizer_fn, scheduler_fn)


def create_adam_optimizers_scheduler_step_lr_fn(datasets, model, learning_rate, step_size, gamma, weight_decay=0):
    """
    Create an ADAM optimizer for each of the dataset with optional scheduler

    Args:
        datasets: a dictionary of dataset
        model: a model to optimize
        learning_rate: the initial learning rate
        step_size: the number of epoch composing a step. Each step the learning rate will be multiplied by `gamma`
        gamma: the factor to apply to the learning rate every step
        weight_decay : the weight decay

    Returns:
        An optimizer with a step scheduler
    """
    scheduler_fn = functools.partial(create_scheduler_step_lr, step_size=step_size, gamma=gamma)
    return create_adam_optimizers_fn(datasets, model, learning_rate=learning_rate, weight_decay=weight_decay, scheduler_fn=scheduler_fn)


def create_sgd_optimizers_fn(datasets, model, learning_rate, momentum=0.9, weight_decay=0, scheduler_fn=None):
    """
        Create a Stochastic gradient descent optimizer for each of the dataset with optional scheduler

        Args:
            datasets: a dictionary of dataset
            model: a model to optimize
            learning_rate: the initial learning rate
            scheduler_fn: a scheduler, or `None`
            momentum: the momentum of the SGD
            weight_decay: the weight decay

        Returns:
            An optimizer
        """
    optimizer_fn = functools.partial(torch.optim.SGD, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    return create_optimizers_fn(datasets, model, optimizer_fn, scheduler_fn)


def create_sgd_optimizers_scheduler_step_lr_fn(datasets, model, learning_rate, step_size, gamma, weight_decay=0, momentum=0.9):
    """
        Create a Stochastic gradient descent optimizer for each of the dataset with step learning rate scheduler

        Args:
            datasets: a dictionary of dataset
            model: a model to optimize
            learning_rate: the initial learning rate
            step_size: the number of epoch composing a step. Each step the learning rate will be multiplied by `gamma`
            gamma: the factor to apply to the learning rate every step
            weight_decay: the weight decay

        Returns:
            An optimizer with a step scheduler
        """
    scheduler_fn = functools.partial(create_scheduler_step_lr, step_size=step_size, gamma=gamma)
    return create_sgd_optimizers_fn(datasets, model, learning_rate=learning_rate, weight_decay=weight_decay, scheduler_fn=scheduler_fn, momentum=momentum)
