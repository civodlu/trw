from typing import Tuple, Optional, Any, Callable

import copy
import logging

from .params_optimizer import HyperParametersOptimizer
from .store import Metrics, RunStore, RunResult
from .params import HyperParameters

logger = logging.getLogger(__name__)


def log_with_logger(string):
    """
    Log string to console
    """
    logger.info(string)


class HyperParametersOptimizerRandomSearchLocal(HyperParametersOptimizer):
    """
    hyper-parameter search using a random walk on a single machine
    """
    def __init__(self,
                 evaluate_fn: Callable[[HyperParameters], Tuple[Metrics, Any]],
                 repeat: int,
                 log_string: Callable[[str], None] = log_with_logger):
        """
        
        Args:
            evaluate_fn: th evaluation function
            repeat: the number of random iterations
            log_string: how to log the search
        """
        self.evaluate_fn = evaluate_fn
        self.repeat = repeat
        self.log_string = log_string

    def optimize(self, store: Optional[RunStore] = None, hyper_parameters: Optional[HyperParameters] = None):
        """
        Optimize the hyper-parameter search using random search
        
        Args:
            store: defines how the runs will be saved
            hyper_parameters: the hyper parameters to be optimized.
        """
        # start with no hyper-parameters
        if hyper_parameters is None:
            hyper_parameters = HyperParameters()

        # then randomly evaluate random hyper parameters and export the results
        iteration = 0
        results = []
        while self.repeat != iteration:
            # by default, evaluate the given hyper parameter default value
            metrics, info = self.evaluate_fn(hyper_parameters)
            self.log_string(f'iteration={iteration}, metrics={metrics}, params={str(hyper_parameters.hparams)}')

            run_result = RunResult(metrics=metrics, hyper_parameters=copy.deepcopy(hyper_parameters), info=info)
            if store is not None:
                store.save_run(run_result)

            results.append(run_result)
            iteration += 1

            if len(hyper_parameters) == 0:
                self.log_string('No hyper-parameter found. Random search stopped!')
                break

            # at the end of the iteration, randomly modify the hyper parameters
            hyper_parameters.randomize()
        return results
