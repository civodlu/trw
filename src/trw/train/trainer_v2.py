import logging
import os
import pickle
import sqlite3
import traceback
from io import StringIO

import torch
from .utilities import default_sum_all_losses, create_or_recreate_folder, RuntimeFormatter
from .trainer import default_per_epoch_callbacks, default_pre_training_callbacks, \
    default_post_training_callbacks, trainer_callbacks_per_batch, epoch_train_eval, create_losses_fn, strip_unpickable
from ..utils import safe_lookup, ExceptionAbortRun
from ..utils.graceful_killer import GracefulKiller

logger = logging.getLogger(__name__)


class TrainerV2:

    def __init__(
            self,
            callbacks_per_batch=None,
            callbacks_per_batch_loss_terms=None,
            callbacks_per_epoch=default_per_epoch_callbacks(),
            callbacks_pre_training=default_pre_training_callbacks(),
            callbacks_post_training=default_post_training_callbacks(),
            trainer_callbacks_per_batch=trainer_callbacks_per_batch,
            run_epoch_fn=epoch_train_eval,
            skip_eval_epoch_0=True):
        """

        Args:
            callbacks_per_batch:
            callbacks_per_batch_loss_terms:
            callbacks_per_epoch:
            callbacks_pre_training:
            callbacks_post_training:
            trainer_callbacks_per_batch:
            run_epoch_fn:
            skip_eval_epoch_0: if ``True``, validation/test will not be run for epoch 0
        """
        self.callbacks_per_batch = callbacks_per_batch
        self.callbacks_per_epoch = callbacks_per_epoch
        self.callbacks_pre_training = callbacks_pre_training
        self.callbacks_post_training = callbacks_post_training
        self.callbacks_per_batch_loss_terms = callbacks_per_batch_loss_terms
        self.trainer_callbacks_per_batch = trainer_callbacks_per_batch
        self.run_epoch_fn = run_epoch_fn
        self.skip_eval_epoch_0 = skip_eval_epoch_0

    @staticmethod
    def save_model(model, result, path, pickle_module=pickle):
        """
        Save a model to file

        Args:
            model: the model to serialize
            result: an optional result file associated with the model
            path: the base path to save the model
            pickle_module: the serialization module that will be used to save the model and results

        """
        result_cp = None
        sql_database = None
        if result is not None:
            import copy
            # we don't want this function to have side effects so copy
            # the result and strip what can't be pickled
            result_cp = copy.copy(result)

            if 'outputs' in result_cp is not None:
                result_cp['outputs'] = strip_unpickable(result_cp['outputs'])

            sql_database = safe_lookup(result_cp, 'options', 'workflow_options', 'sql_database')
            if sql_database is not None:
                del result_cp['options']['workflow_options']['sql_database']

        result_cp_path = path + '.result'
        with open(result_cp_path, 'wb') as f:
            pickle_module.dump(result_cp, f)
        torch.save(model, path, pickle_module=pickle_module)

        if sql_database is not None:
            # TODO find a cleaner and generic way of doing this...
            result_cp['options']['workflow_options']['sql_database'] = sql_database

    @staticmethod
    def load_model(path, with_result=False, device=None, pickle_module=pickle):
        """
        load a saved model

        Args:
            path: where to store the model. result's will be loaded from `path + '.result'`
            with_result: if True, the results of the model will be loaded
            device: where to load the model. For example, models are typically trained on GPU,
                but for deployment, CPU might be good enough. If `None`, use the same device as
                when the model was exported
            pickle_module: the de-serialization module to be used to load model and results

        Returns:
            a tuple `model, result`
        """
        result = None
        if with_result:
            result_path = path + '.result'
            with open(result_path, 'rb') as f:
                result = pickle_module.load(f)
        model = torch.load(path, map_location=device, pickle_module=pickle_module)
        return model, result

    def fit(self, options, datasets, model, optimizers_fn,
            losses_fn=default_sum_all_losses,
            loss_creator=create_losses_fn,
            log_path=None,
            with_final_evaluation=True,
            history=None,
            erase_logging_folder=True,
            eval_every_X_epoch=1):
        """
        Fit the model

        Requirements:

        * enough main memory to store the outputs of all the datasets of a single epoch.
            If this cannot be satisfied, sub-sample the epoch so that it can fit in main memory.

        Notes:

        * if a feature value is Callable, its value will be replaced by the result of the call
            (e.g., this can be useful to generate `z` embedding in GANs)

        :param options:
        :param datasets: a functor returning a dictionary of datasets. Alternatively, datasets infos can be specified.
                        `inputs_fn` must return one of:

                        * datasets: dictionary of dataset
                        * (datasets, datasets_infos): dictionary of dataset and additional infos

                        We define:

                        * datasets: a dictionary of dataset. a dataset is a dictionary of splits. a split is a dictionary of batched features.
                        * Datasets infos are additional infos useful for the debugging of the dataset (e.g., class mappings, sample UIDs).
                        Datasets infos are typically much smaller than datasets should be loaded in loadable in memory

        :param model: a functor with parameter `options` and returning a `Module` or a `ModuleDict`

        Depending of the type of the model, this is how it will be used:

        * `Module`: optimizer will optimize `model.parameters()`
        * `ModuleDict`: for each dataset name, the optimizer will optimize
            `model[dataset_name].parameters()`. Note that a `forward` method will need to be implemented

        :param losses_fn:
        :param optimizers_fn:
        :param loss_creator:
        :param eval_every_X_epoch: evaluate the model every `X` epochs
        :param log_path: where the trainer will log its info. If `None`, default to
            a folder in options['workflow_options']['logging_directory']
        :param with_final_evaluation: if True, once the model is fitted, evaluate all the data again in eval mode
        :return: a tuple `model, result`
        """
        # reset the abort state
        GracefulKiller.abort_event.clear()

        # set up our log path. This is where all the analysis of the model will be exported
        if log_path is None:
            log_path = os.path.join(
                options['workflow_options']['logging_directory'],
                'default_r{}'.format(options['workflow_options']['trainer_run']))
        options['workflow_options']['current_logging_directory'] = log_path

        if history is None:
            # no prior history
            history = []

        # now clear our log path to remove previous files if needed
        if erase_logging_folder:
            create_or_recreate_folder(log_path)
        elif not os.path.exists(log_path):
            os.makedirs(log_path)

        if len(logging.root.handlers) == 0:
            # there is no logger configured, so add a basic one
            logging.basicConfig(
                filename=os.path.join(options['workflow_options']['logging_directory'], 'trainer_logging.log'),
                format='%(asctime)s %(levelname)s %(name)s %(message)s',
                level=logging.DEBUG,
                filemode='w')

        # create the reporting SQL database
        sql_path = os.path.join(log_path, 'reporting_sqlite.db')
        sql = sqlite3.connect(sql_path)
        options['workflow_options']['sql_database'] = sql
        options['workflow_options']['sql_database_path'] = sql_path
        options['workflow_options']['sql_database_view_path'] = sql_path.replace('.db', '.json')

        # here we want to have our logging per training run, so add a handler
        handler = logging.FileHandler(os.path.join(log_path, 'trainer.log'))
        formatter = RuntimeFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logging.root.addHandler(handler)

        # instantiate the datasets, model, optimizers and losses
        logger.info('started Trainer.fit(). Options={}'.format(options))

        def clean_up(datasets):
            if datasets is not None:
                # make sure the datasets are closed properly: threads and processes
                # are stopped in a controlled manner to avoid memory leaks
                for dataset_name, dataset in datasets.items():
                    for split_name, split in dataset.items():
                        logger.info(f'closing dataset={dataset_name} split={split_name}')
                        split.close()
                        logger.info(f'closed dataset={dataset_name} split={split_name}!')

                # resource are released, just continue the shutdown
                logger.info(f'datasets all closed!')

            # increment the number of runs
            options['workflow_options']['trainer_run'] += 1

            logger.info('removing logging handlers...')
            logging.root.removeHandler(handler)

            logger.info('training completed!')

            sql.commit()
            sql.close()

        datasets_infos = None  # TODO REFACTOR THIS
        assert datasets is not None, '`datasets` is None!'
        if isinstance(datasets, tuple):
            if len(datasets) == 2:
                logger.info('inputs_fn specified `datasets, datasets_infos`')
                datasets, datasets_infos = datasets
            else:
                assert 0, 'expected tuple `datasets` or `datasets, datasets_infos`'

        assert isinstance(model, torch.nn.Module), f'the model MUST be a `torch.nn.Module`, got={type(model)}'
        if isinstance(model, torch.nn.ModuleDict):
            # if we have sub-models, we MUST define a `forward` method
            # to orchestrate the calls of sub-models
            assert 'forward' in dir(model)

        outputs_epoch = None
        try:
            # migrate the model to the specified device
            device = options['workflow_options']['device']

            logger.info('model moved to device={}'.format(device))
            model.to(device)

            # instantiate the optimizer and scheduler
            logger.info('creating optimizers...')
            if optimizers_fn is not None:
                optimizers, schedulers, per_step_scheduler_fn = optimizers_fn(datasets, model)
                logger.info('optimizers created successfully!')
            else:
                logger.info('optimizer fn is None! No optimizer created.')
                optimizers, schedulers, per_step_scheduler_fn = None, None, None

            logger.info('creating losses...')
            losses = loss_creator(datasets, losses_fn)
            logger.info('losses created successfully!')

            num_epochs = options['training_parameters']['num_epochs']

            callbacks_per_epoch = []
            if self.callbacks_per_epoch is not None:
                callbacks_per_epoch += self.callbacks_per_epoch

            callbacks_per_batch = []
            if self.trainer_callbacks_per_batch is not None:
                callbacks_per_batch.append(self.trainer_callbacks_per_batch)
            if self.callbacks_per_batch is not None:
                callbacks_per_batch += self.callbacks_per_batch

            callbacks_per_batch_loss_terms = []
            if self.callbacks_per_batch_loss_terms is not None:
                callbacks_per_batch_loss_terms += self.callbacks_per_batch_loss_terms
            logger.info('callbacks created successfully!')

            # run the callbacks  before training
            if self.callbacks_pre_training is not None:
                logger.info('running pre-training callbacks...')
                for callback in self.callbacks_pre_training:
                    try:
                        callback(options, history, model, losses=losses, outputs=None,
                                 datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch,
                                 optimizers_fn=optimizers_fn, optimizers=optimizers)
                    except Exception as e:
                        f = StringIO()
                        traceback.print_exc(file=f)
                        print(f'callback={callback} failed with exception={e}. Stacktrace=\n{f.getvalue()}')
                        logger.error(f'callback={callback} failed with exception={e}. Stacktrace=\n{f.getvalue()}')
                logger.info('pre-training callbacks completed!')

            for epoch in range(num_epochs):
                logger.info('started training epoch {}'.format(epoch))
                run_eval = (epoch == 0 and not self.skip_eval_epoch_0) or (epoch + 1) % eval_every_X_epoch == 0

                outputs_epoch, history_epoch = self.run_epoch_fn(
                    options,
                    datasets,
                    optimizers,
                    model,
                    losses,
                    schedulers,
                    per_step_scheduler_fn,
                    history,
                    callbacks_per_batch,
                    callbacks_per_batch_loss_terms,
                    run_eval=run_eval,
                    force_eval_mode=False)
                history.append(history_epoch)

                logger.info('finished training epoch {}'.format(epoch))

                last_epoch = epoch + 1 == num_epochs

                logger.info('callbacks started')
                for callback in callbacks_per_epoch:
                    try:
                        callback(options, history, model, losses=losses, outputs=outputs_epoch,
                                 datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch,
                                 optimizers_fn=optimizers_fn, optimizers=optimizers, last_epoch=last_epoch)
                    except Exception as e:
                        f = StringIO()
                        traceback.print_exc(file=f)
                        logger.error(f'callback={callback} failed with exception={e}.\n Stack={f.getvalue()}')

                logger.info(f'callbacks epoch {epoch} finished')

            # finally run the post-training callbacks
            if with_final_evaluation:
                logger.info('started final evaluation...')

                outputs_epoch, history_epoch = self.run_epoch_fn(
                    options=options,
                    datasets=datasets,
                    optimizers=None,
                    model=model,
                    losses=losses,
                    schedulers=None,
                    per_step_schedulers=None,
                    history=history,
                    callbacks_per_batch=callbacks_per_batch,
                    callbacks_per_batch_loss_terms=callbacks_per_batch_loss_terms,
                    run_eval=True,
                    force_eval_mode=True)
                logger.info('finished final evaluation...')
                history.append(history_epoch)

            if self.callbacks_post_training is not None:
                logger.info('started post training callbacks...')
                for callback in self.callbacks_post_training:
                    try:
                        callback(options, history, model, losses=losses, outputs=outputs_epoch,
                                 datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch,
                                 optimizers_fn=optimizers_fn)
                    except Exception as e:
                        f = StringIO()
                        traceback.print_exc(file=f)
                        print(f'callback={callback} failed with exception={e}.\n Stack={f.getvalue()}')
                        logger.error(f'callback={callback} failed with exception={e}.\n Stack={f.getvalue()}')

                logger.info('finished post training callbacks...')

        except (KeyboardInterrupt, RuntimeError, ExceptionAbortRun) as e:
            # since we are about to exit the process, explicitly
            # dispose the datasets to make sure resources are properly disposed of
            logger.info('KeyboardInterrupt received. closing datasets explicitly')
            clean_up(datasets)

            # since the resources are released, we can now re-raise the exception
            raise e

        # do not explicitly clean up the datasets since these were
        # created outside the trainer
        clean_up(datasets=None)

        return model, {
            'history': history,
            'options': options,
            'outputs': outputs_epoch,
            'datasets_infos': datasets_infos
        }
