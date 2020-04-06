import numpy as np
from trw.train import utilities
from sklearn import metrics
from trw.train import losses
import collections
import torch


class Metric:
    """
    A metric base class

    Calculate interesting metric
    """
    def __call__(self, outputs):
        """

        Args:
            outputs:
                the outputs of a batch
        Returns:
            a dictionary of metric names/values or None
        """
        metric_value = 0
        return Metric, {'metric_name': metric_value}

    @staticmethod
    def aggregate_metrics(metric_by_batch):
        """

        Args:
            metric_by_batch: a list of metrics, one for each batch

        Returns:
            a dictionary of result name and value
        """
        raise NotImplemented()


class MetricLoss(Metric):
    """
    Extract the loss from the outputs
    """
    def __call__(self, outputs):
        loss = utilities.to_value(outputs.get('loss'))
        if loss is not None:
            return {'loss': float(loss)}
        return None

    @staticmethod
    def aggregate_metrics(metric_by_batch):
        loss = 0.0
        for m in metric_by_batch:
            loss += m['loss']
        return {'loss': loss / len(metric_by_batch)}


class MetricClassificationError(Metric):
    """
    Calculate the accuracy using the `output_truth` and `output`
    """
    def __call__(self, outputs):
        truth = utilities.to_value(outputs.get('output_truth'))
        found = utilities.to_value(outputs.get('output'))
        if truth is not None and found is not None:
            return {
                'nb_trues': np.sum(found == truth),
                'total': len(truth)
            }
        return None

    @staticmethod
    def aggregate_metrics(metric_by_batch):
        nb_trues = 0
        total = 0
        for m in metric_by_batch:
            nb_trues += m['nb_trues']
            total += m['total']
        return {'classification error': 1.0 - nb_trues / total}


class MetricSegmentationDice(Metric):
    """
    Calculate the average dice score of a segmentation map 'output_truth' and class
    segmentation probabilities 'output_raw'
    """
    def __init__(self, dice_fn=losses.LossDiceMulticlass(return_dice_by_class=True)):
        self.dice_fn = dice_fn

    def __call__(self, outputs):
        # keep the torch variable. We want to use GPU if available since it can
        # be slow use numpy for this
        truth = outputs.get('output_truth')
        found = outputs.get('output_raw')

        if found is None or truth is None:
            return None

        assert len(found.shape) == len(truth.shape) + 1, f'expecting dim={len(truth.shape)}, got={len(found.shape)}'
        with torch.no_grad():
            dice_by_class = utilities.to_value(self.dice_fn(found, truth))

        return {
            'dice_by_class': dice_by_class
        }

    @staticmethod
    def aggregate_metrics(metric_by_batch):
        sum_dices = metric_by_batch[0]['dice_by_class']
        for m in metric_by_batch[1:]:
            sum_dices += m['dice_by_class']

        nb_batches = len(metric_by_batch)
        if nb_batches > 0:
            # calculate the dice score by class
            one_minus_dice = 1 - sum_dices / len(metric_by_batch)
            r = collections.OrderedDict()
            for c in range(len(sum_dices)):
                r[f'1-dice[class={c}]'] = one_minus_dice[c]
            r['1-dice'] = np.average(one_minus_dice)

            return r

        return {'1-dice': 1}


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
                # special case: only binary classification
                tn, fp, fn, tp = cm.ravel()

                return {
                    'tn': tn,
                    'fn': fn,
                    'fp': fp,
                    'tp': tp,
                }
            else:
                if truth[0] == 0:
                    # 0, means perfect classification of the negative
                    return {
                        'tn': cm[0, 0],
                        'fn': 0,
                        'fp': 0,
                        'tp': 0,
                    }
                else:
                    # 1, means perfect classification of the positive
                    return {
                        'tp': cm[0, 0],
                        'fn': 0,
                        'fp': 0,
                        'tn': 0,
                    }

        # something is missing, don't calculate the stats
        return None

    @staticmethod
    def aggregate_metrics(metric_by_batch):
        tn = 0
        fp = 0
        fn = 0
        tp = 0

        for m in metric_by_batch:
            tn += m['tn']
            fn += m['fn']
            tp += m['tp']
            fp += m['fp']

        if tp + fn > 0:
            one_minus_sensitivity = 1.0 - tp / (tp + fn)
        else:
            # invalid! `None` will be discarded
            one_minus_sensitivity = None

        if fp + tn > 0:
            one_minus_specificity = 1.0 - tn / (fp + tn)
        else:
            # invalid! `None` will be discarded
            one_minus_specificity = None

        return {
            # we return the 1.0 - metric, since in the history we always keep the smallest number
            '1-sensitivity': one_minus_sensitivity,
            '1-specificity': one_minus_specificity,
        }


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
    Default list of metrics used for regression
    """
    return [
        MetricLoss(),
    ]


def default_segmentation_metrics():
    """"
    Default list of metrics used for segmentation
    """
    return [
        MetricLoss(),
        MetricSegmentationDice(),
    ]


def default_generic_metrics():
    """"
    Default list of metrics
    """
    return [
        MetricLoss(),
    ]
