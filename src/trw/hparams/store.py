import io
from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence
from .params import HyperParameters
import pickle


Metrics = Dict[str, float]


class RunResult:
    """
    Represent the result of a run
    """
    def __init__(self, metrics: Metrics, hyper_parameters: HyperParameters, info: Any = None):
        """
        Args:
            metrics: the metrics to be recorded
            info: additional info related to the run
            hyper_parameters: hyper parameter that led to these metrics
        """
        self.info = info
        self.hyper_parameters = hyper_parameters
        self.metrics = metrics


class RunStore(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the store
        """
        pass

    @abstractmethod
    def save_run(self, run_result: RunResult) -> None:
        """
        Save the results of a run

        Args:
            run_result: the results to record
        """

    @abstractmethod
    def load_all_runs(self) -> Sequence[RunResult]:
        """
        Load all the runs
        """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RunStoreFile(RunStore):
    def __init__(self, store_location: str, serializer=pickle):
        super().__init__()
        self.serializer = serializer
        self.f = open(store_location, mode='ab+')

    def close(self) -> None:
        """
        Close the store
        """
        if self.f is not None:
            self.f.close()

    def save_run(self, run_result: RunResult) -> None:
        """
        Save the results of a run

        Args:
            run_result: the results to record
        """
        assert self.f is not None, 'file is already closed!'
        self.f.seek(0, io.SEEK_END)
        self.serializer.dump(run_result, self.f)

    def load_all_runs(self) -> Sequence[RunResult]:
        """
        Load all the runs
        """
        self.f.seek(0, io.SEEK_SET)

        results = []
        while True:
            try:
                r = self.serializer.load(self.f)
                results.append(r)
            except EOFError:
                break

        return results
