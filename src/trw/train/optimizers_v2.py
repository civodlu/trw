

from functools import partial
from pyparsing import Opt
import torch
from torch import nn
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
from trw.basic_typing import Datasets


SchedulerType = Any
StepSchedulerType = Any


class Optimizer:
    def __init__(
            self, 
            optimizer_fn: Callable[[Iterator[nn.parameter.Parameter]], torch.optim.Optimizer],
            scheduler_fn: Optional[Callable[[torch.optim.Optimizer], SchedulerType]] = None,
            step_scheduler_fn: Optional[Callable[[torch.optim.Optimizer], StepSchedulerType]] = None):
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.per_step_scheduler_fn = step_scheduler_fn

    def set_scheduler_fn(self, scheduler_fn: Optional[Callable[[torch.optim.Optimizer], SchedulerType]]):
        self.scheduler_fn = scheduler_fn

    def set_step_scheduler_fn(self, step_scheduler_fn: Optional[Callable[[torch.optim.Optimizer], StepSchedulerType]]):
        self.step_scheduler_fn = step_scheduler_fn

    def __call__(self, datasets: Datasets, model: nn.Module) -> Tuple[Dict[str, torch.optim.Optimizer], Optional[Dict[str, SchedulerType]], Optional[Dict[str, StepSchedulerType]]]:
        per_step_schedulers = None
        schedulers = None
        if self.scheduler_fn is not None:
            schedulers = {}
        if self.per_step_scheduler_fn is not None:
            per_step_schedulers = {}

        optimizers = {}
        for dataset_name in datasets.keys():
            if isinstance(model, torch.nn.ModuleDict):
                # this is a collection of model. Assumed we have a different model
                # per dataset to be optimized
                sub_model = model[dataset_name]
                optimizer = self.optimizer_fn(sub_model.parameters())
            else:
                optimizer = self.optimizer_fn(model.parameters())

            optimizers[dataset_name] = optimizer

            if self.scheduler_fn is not None and optimizer is not None:
                scheduler = self.scheduler_fn(optimizer)
                schedulers[dataset_name] = scheduler

            if self.per_step_scheduler_fn is not None and optimizer is not None:
                per_step_scheduler = self.per_step_scheduler_fn(optimizer)
                per_step_schedulers[dataset_name] = per_step_scheduler

        return optimizers, schedulers, per_step_schedulers

    def scheduler_step_lr(self, step_size: int, gamma: float = 0.1) -> 'Optimizer':
        scheduler_fn = partial(torch.optim.lr_scheduler.StepLR, step_size=step_size, gamma=gamma)
        self.set_scheduler_fn(scheduler_fn)
        return self


class OptimizerSGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9, weight_decay: float = 0, nesterov: bool = False):
        super().__init__(optimizer_fn=partial(torch.optim.SGD, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov))

class OptimizerAdam(Optimizer):
    def __init__(self, learning_rate: float, weight_decay: float = 0, betas: Tuple[float, float] = (0.9, 0.999), eps: float=1e-8):
        super().__init__(optimizer_fn=partial(torch.optim.Adam, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps))

class OptimizerAdamW(Optimizer):
    def __init__(self, learning_rate: float, weight_decay: float = 1e-2, betas: Tuple[float, float] = (0.9, 0.999), eps: float=1e-8):
        super().__init__(optimizer_fn=partial(torch.optim.AdamW, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps))