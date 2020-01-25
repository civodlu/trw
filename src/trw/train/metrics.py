import numpy as np
from trw.train import utilities
from sklearn import metrics
from trw.train import losses
import torch


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
        return {
            'metric_name': metric_value
        }


class MetricLoss(Metric):
    """
    Extract the loss from the outputs
    """
    def __call__(self, outputs):
        loss = utilities.to_value(outputs.get('loss'))
        if loss is not None:
            return {
                'loss': float(loss)
            }
        return None


class MetricClassificationError(Metric):
    """
    Calculate the accuracy using the `output_truth` and `output`
    """
    def __call__(self, outputs):
        truth = utilities.to_value(outputs.get('output_truth'))
        found = utilities.to_value(outputs.get('output'))
        if truth is not None and found is not None:
            return {
                'classification error': 1.0 - np.sum(found == truth) / len(truth)
            }
        return None


class MetricSegmentationDice(Metric):
    """
    Calculate the average dice score of a segmentation map 'output_truth' and class
    segmentation probabilities 'output_raw'
    """
    def __init__(self, dice_fn=losses.LossDiceMulticlass()):
        self.dice_fn = dice_fn

    def __call__(self, outputs):
        # keep the torch variable. We want to use GPU if available since it can
        # be slow use numpy for this
        truth = outputs.get('output_truth')
        found = outputs.get('output_raw')

        if found is None or truth is None:
            return {}

        assert len(found.shape) == len(truth.shape) + 1, f'expecting dim={len(truth.shape)}, got={len(found.shape)}'
        with torch.no_grad():
            one_minus_dices = self.dice_fn(found, truth)
            mean_dices = utilities.to_value(torch.mean(one_minus_dices))

        return {
            '1-dice': mean_dices
        }


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

        truth = utilities.to_value(outputs.get('output_truth'))
        found = utilities.to_value(outputs.get('output'))
        if truth is not None and found is not None:
            cm = metrics.confusion_matrix(y_pred=found, y_true=truth)
            if len(cm) == 2:
                # special case: binary classification
                tn, fp, fn, tp = cm.ravel()

                if tp + fn > 0:
                    one_minus_sensitivity = 1.0 - tp / (tp + fn)
                    one_minus_specificity = 1.0 - tn / (fp + tn)
                else:
                    # invalid! so return the worst stats possible
                    one_minus_sensitivity = 1.0
                    one_minus_specificity = 1.0

                return {
                    # we return the 1.0 - metric, since in the history we always keep the smallest number
                    '1-sensitivity': one_minus_sensitivity,
                    '1-specificity': one_minus_specificity,
                }
            else:
                return {
                    # we return the 1.0. We can't calculate the stats
                    '1-sensitivity': 1.0,
                    '1-specificity': 1.0,
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
        MetricSegmentationDice(),
    ]
