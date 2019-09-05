from trw.hparams import params
import pickle
import os
import copy
import logging
import numbers
import numpy as np
import math


logger = logging.getLogger(__name__)


def log_random(string):
    """
    Log string to console
    """
    logger.info(string)


def store_loss_params(output_location, loss, infos, hyper_parameters):
    """
    Export the loss as self contained .pkl file to be analyzed later
    """
    with open(output_location, 'wb') as f:
        pickle.dump(loss, f, protocol=0)
        pickle.dump(infos, f, protocol=0)
        pickle.dump(hyper_parameters, f, protocol=0)


def load_loss_params(output_location):
    """
    reload the loss as self contained .pkl file to be analyzed later
    """
    with open(output_location, 'rb') as f:

        loss = pickle.load(f)
        infos = pickle.load(f)
        hyper_parameters = pickle.load(f)
    return loss, infos, hyper_parameters


class HyperParametersOptimizerRandomSearchLocal:
    """
    Random hyper parameter run on a single machine

    We need to define the hyper parameter evaluation function::
    
    def evaluate_hparams(hparams):
        # evaluate an hyper-parameter configuration and return a loss value and some additional information
        # e.g., result report
        return 0.0, {}
        
    """

    def __init__(self, evaluate_hparams_fn, repeat, log_string=log_random, result_prefix='hparams-random'):
        """
        
        Args:
            evaluate_hparams_fn: the function to be optimized. It will be valled with `hparams` and expects as result
                a tuple (loss, report)
            repeat: the number of iteration of the search
            log_string: how to log the search
            result_prefix: the prefix used for the result file
        """
        self.evaluate_hparams_fn = evaluate_hparams_fn
        self.repeat = repeat
        self.log_string = log_string
        self.result_prefix = None
        if result_prefix is not None:
            self.result_prefix = result_prefix + '-'

    def optimize(self, result_path):
        """
        Optimize the hyper-parameter search using random search
        
        Args:
            result_path: where to save the information of each run. Can be None, in this case nothing is exported.

        Returns:
            the results of all the runs
        """
        # start with no hyper-parameters
        hyper_parameters = params.HyperParameters()

        # then randomly evaluate random hyper parameters and export the results
        iteration = 0
        results = []
        while self.repeat != iteration:
            # by default, evaluate the given hyper parameter default value
            loss, infos = self.evaluate_hparams_fn(hyper_parameters)
            assert isinstance(loss, (numbers.Number, np.ndarray)), 'loss should be a number'
            
            self.log_string('iteration=%d, loss=%f, infos=%s, params=%s' % (iteration, loss, str(infos), str(hyper_parameters.hparams)))

            # record the actual configuration so that we can reproduce interesting results
            if result_path is not None:
                output_location = os.path.join(result_path, self.result_prefix + 'loss-%s-iter-%d.pkl' % (str(loss), iteration))
                directory, _ = os.path.split(output_location)
                if not os.path.exists(directory):
                    os.mkdir(directory)
                if loss is not None and not math.isnan(loss):
                    # store only valid trials
                    store_loss_params(output_location, loss, infos, hyper_parameters)

            results.append((loss, infos, copy.deepcopy(hyper_parameters)))
            iteration += 1

            if len(hyper_parameters) == 0:
                self.log_string('No hyper-parameter found. Random search stopped!')
                break

            # at the end of the iteration, randomly modify the hyper parameters
            hyper_parameters.generate_random_hparams()
        return results
