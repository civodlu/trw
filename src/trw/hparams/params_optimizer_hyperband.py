from . import params
from . import params_optimizer_random_search
import os
import copy
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


def log_hyperband(msg):
    logger.info(msg)


class HyperParametersOptimizerHyperband:
    """
    Implementation of `Hyperband: a novel bandit based approach to hyper-parameter optimization`
    https://arxiv.org/abs/1603.06560

    def evaluate_hparams(hparams, nb_epochs):
        # evaluate an hyper-parameter configuration and return a loss value and some additional information
        # e.g., result report
        return 0.0, {}
    """

    def __init__(self, evaluate_hparams_fn,
                 result_prefix='',
                 max_iter=81,
                 eta=3,
                 repeat=100,
                 log_string=log_hyperband,
                 always_include_default_hparams_in_each_cycle=True):
        """
        Table of runs for max_iter=81, eta=3

        max_iter = 81        s=4             s=3             s=2             s=1             s=0
        eta = 3              n_i   r_i       n_i   r_i       n_i   r_i       n_i   r_i       n_i   r_i
        B = 5*max_iter       ---------       ---------       ---------       ---------       ---------
                              81    1         27    3         9     9         6     27        5     81
                              27    3         9     9         3     27        2     81
                              9     9         3     27        1     81
                              3     27        1     81
                              1     81

        n_i = number of configurations
        r_i = number of iteration per configuration

        :param evaluate_hparams_fn: input `(hparams, nb_epochs)` and return `loss, {report}`
        :param max_iter: the maximum number of epoch for the training of the best configuration
        :param eta: downsampling rate
        :param always_include_default_hparams_in_each_cycle: if True, for each outer loop, the default parameters are evaluated for the FIRST repeat only!
        :param repeat: number of times hyperband will be repeated
        """
        self.evaluate_hparams_fn = evaluate_hparams_fn
        self.log_string = log_string
        self.max_iter = max_iter
        self.eta = eta
        self.repeat = repeat
        self.always_include_default_hparams_in_each_cycle = always_include_default_hparams_in_each_cycle
        self.result_prefix = result_prefix

    def _repeat_one(self, result_path, hyper_parameters, repeat_id, nb_runs):
        """
        Run full Hyperband search
        :param result_path:
        :return: the last round of configurations
        """
        log = math.log
        default_parameters = copy.deepcopy(hyper_parameters)

        self.log_string('repeat_id=%d, run=%d' % (repeat_id, nb_runs))

        logeta = lambda x: log(x) / log(self.eta)
        s_max = int(logeta(self.max_iter))  # number of unique executions of Successive Halving (minus one)
        B = (s_max + 1) * self.max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

        #
        # Code adapted from https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
        #

        #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
        results_last = []
        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(int(B / self.max_iter / (s + 1)) * self.eta ** s))  # initial number of configurations
            r = self.max_iter * self.eta ** (-s)  # initial number of iterations to run configurations for

            #### Begin Finite Horizon Successive Halving with (n,r)
            T = []
            for i in range(n):
                if self.always_include_default_hparams_in_each_cycle and i == 0 and repeat_id == 0:
                    # include the default configuration for the 1 repeat
                    T.append(copy.deepcopy(default_parameters))
                else:
                    hyper_parameters.generate_random_hparams()
                    T.append(copy.deepcopy(hyper_parameters))
            for i in range(s + 1):
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = n * self.eta ** (-i)
                r_i = r * self.eta ** (i)

                val_losses = []
                val_infos = []
                for t_index, t in enumerate(T):
                    loss, infos = self.evaluate_hparams_fn(t, r_i)
                    val_losses.append(loss)
                    val_infos.append(infos)
                    self.log_string('run=%d, s=%d, r_i=%d, loss=%f, params=%s, infos=%s' % (nb_runs,
                                                                                            s,
                                                                                            r_i,
                                                                                            loss,
                                                                                            str(t.hparams),
                                                                                            str(infos)))

                    if result_path is not None:
                        output_location = os.path.join(
                            result_path,
                            self.result_prefix + 'loss-%s-iter-%d-s-%d-config-%d-repeat-%d-run-%d.pkl' % (str(loss),
                                                                                                          r_i,
                                                                                                          s,
                                                                                                          t_index,
                                                                                                          repeat_id,
                                                                                                          nb_runs))
                        directory, _ = os.path.split(output_location)
                        if not os.path.exists(directory):
                            os.mkdir(directory)
                            
                        params_optimizer_random_search.store_loss_params(output_location, loss, infos, t)
                    nb_runs += 1

                best_runs = np.argsort(val_losses)[0:int(n_i / self.eta)]
                run_params = []
                for run in range(len(val_losses)):
                    run_params.append((val_losses[run], val_infos[run], T[run]))

                T = [T[i] for i in best_runs]

            results_last += run_params  # keep track of the last round for each `s`
        return results_last, nb_runs

    def optimize(self, result_path):
        """
        Optimize the hyper parameters using Hyperband
        
        Args:
            result_path: where to save the information of each run. Can be None, in this case nothing is exported.

        Returns:
            the results of all the runs
        """
        hyper_parameters = params.HyperParameters()

        # here `discover` the hyper-parameter. We must assume the hyper-parameter list
        # won't change
        self.evaluate_hparams_fn(hyper_parameters, 1)

        results = []
        nb_runs = 0
        for repeat_id in range(self.repeat):
            r, nb_runs = self._repeat_one(result_path,
                                          hyper_parameters=hyper_parameters,
                                          repeat_id=repeat_id,
                                          nb_runs=nb_runs)
            results += r
        return results
