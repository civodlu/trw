import numpy as np
from trw.train import utilities


class Metric:
    """
    A metric base class

    Calculate interesting metric
    """
    def __call__(self, outputs):
        """
        Calculate a metric from the `outputs`
        :param outputs: the data required to calculate the metric from
        :return: a tuple (`metric name`, `metric value`) or `None`
        """
        metric_value = 0
        return 'metric_name', metric_value


class MetricLoss(Metric):
    """
    Extract the loss from the outputs
    """
    def __call__(self, outputs):
        loss = outputs.get('loss')
        if loss is not None:
            return 'loss', float(utilities.to_value(loss))
        return None


class MetricClassificationError(Metric):
    """
    Calculate the accuracy using the `output_truth` and `output`
    """
    def __call__(self, outputs):
        truth = outputs.get('output_truth')
        found = outputs.get('output')
        if truth is not None and found is not None:
            return 'classification error', 1.0 - np.sum(found == truth) / len(truth)
        return None


def default_classification_metrics():
    """"
    Default list of metrics used for classification
    """
    return [
        MetricLoss(),
        MetricClassificationError(),
    ]


def default_regression_metrics():
    """"
    Default list of metrics used for classification
    """
    return [
        MetricLoss(),
    ]


def default_segmentation_metrics():
    """"
    Default list of metrics used for classification
    """
    return [
        MetricLoss(),
    ]
