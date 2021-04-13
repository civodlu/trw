import collections
from typing import Callable, Sequence

from ..utils import ExceptionAbortRun
from ..basic_typing import HistoryStep
from .callback import Callback
from ..hparams import RunStore
import logging


logger = logging.getLogger(__name__)


class CallbackEarlyStopping(Callback):
    """
    Use historical runs to evaluate if a run is promising. If not, early stop will raise :class:`ExceptionAbortRun`
    """
    def __init__(
            self,
            store: RunStore,
            loss_fn: Callable[[HistoryStep], float],
            checkpoints: Sequence[float] = (0.1, 0.25, 0.5, 0.75),
            discard_if_among_worst_X_performers: float = 0.8,
            min_number_of_runs: int = 10):
        """

        Args:
            store: how to retrieve previous runs
            loss_fn: extract a loss value from an history step. This will be used to rank the runs
            checkpoints: define the number of checks (expressed as fraction of total epochs) to evaluate
                this run against the historical database of runs.
            discard_if_among_worst_X_performers: for each checkpoint, the current run is ranked among all the
                runs using `loss_fn` and `store`. If the runs is X% worst performer, discard the run
            min_number_of_runs: collect at least this number of runs before applying the early stopping.
                larger number means better estimation of the worst losses.
        """
        self.min_number_of_runs = min_number_of_runs
        self.discard_if_among_worst_X_performers = discard_if_among_worst_X_performers
        self.checkpoints = checkpoints
        self.loss_fn = loss_fn
        self.store = store
        self.max_loss_by_epoch = None

        assert 0 < discard_if_among_worst_X_performers < 1, 'must be a fraction!'
        for c in checkpoints:
            assert 0 < c < 1, 'must be a fraction!'

    def _initialize(self, num_epochs):
        logger.info('initializing run analysis...')
        checkpoints_epoch = [int(f * num_epochs) for f in self.checkpoints]
        all_runs = self.store.load_all_runs()

        # collect loss for all runs at given checkpoints
        losses_by_step = collections.defaultdict(list)
        for e in checkpoints_epoch:
            for run in all_runs:
                if len(run.history) > e:
                    loss = self.loss_fn(run.history[e])
                    losses_by_step[e].append(loss)

        # for each checkpoint, sort the losses, and calculate the worst X% of the runs
        # the current run MUST be better than the threshold or it will be pruned
        max_loss_by_epoch = {}
        for e, values in losses_by_step.items():
            if len(values) < self.min_number_of_runs:
                # not enough runs to get reliable estimate, keep this run!
                max_loss_by_epoch[e] = None
                continue

            values = sorted(values)
            rank = round(len(values) * (1.0 - self.discard_if_among_worst_X_performers))
            threshold = values[rank]
            max_loss_by_epoch[e] = threshold
        self.max_loss_by_epoch = max_loss_by_epoch
        logger.info(f'max_loss_by_step={max_loss_by_epoch}')

    def __call__(self, options, history, model, **kwargs):
        num_epochs = options['training_parameters']['num_epochs']

        if self.max_loss_by_epoch is None:
            self._initialize(num_epochs)

        epoch = len(history)
        max_loss = self.max_loss_by_epoch.get(epoch)
        if max_loss is not None:
            loss = self.loss_fn(history[-1])
            if loss > max_loss:
                logger.info(f'epoch={epoch}, loss={loss} > {max_loss}, the run is discarded!')
                raise ExceptionAbortRun(
                    history=history,
                    reason=f'loss={loss} is too high (threshold={max_loss}, '
                           f'minimum={self.discard_if_among_worst_X_performers}%')
