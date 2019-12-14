import numpy as np
from trw.train import utilities
from sklearn import metrics


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


class MetricClassificationSensitivitySpecificity(Metric):
    """
    Calculate the sensitivity and specificity for a binary classification using the `output_truth` and `output`
    """
    def __call__(self, outputs):
        output_raw = outputs.get('output_raw')
        if output_raw is None:
            return None
        if len(output_raw.shape) != 2 or output_raw.shape[1] != 2:
            return None

        truth = outputs.get('output_truth')
        found = outputs.get('output')
        if truth is not None and found is not None:
            cm = metrics.confusion_matrix(y_pred=found, y_true=truth)
            if len(cm) == 2:
                # special case: binary classification
                tn, fp, fn, tp = cm.ravel()
                return {
                    'sensitivity': tp / (tp + fn),
                    'specificity': tn / (fp + tn),
                }
        return None


def default_classification_metrics():
    """"
    Default list of metrics used for classification
    """
    return [
        MetricLoss(),
        MetricClassificationError(),
        MetricClassificationSensitivitySpecificity(),
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
