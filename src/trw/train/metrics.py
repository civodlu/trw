import numpy as np
from trw.train import utilities
from sklearn import metrics
from trw.train import losses
import collections
import torch
from .analysis_plots import auroc


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

    def aggregate_metrics(self, metric_by_batch):
        """

        Args:
            metric_by_batch: a list of metrics, one for each batch

        Returns:
            a dictionary of result name and value
        """
        raise NotImplementedError()


class MetricLoss(Metric):
    """
    Extract the loss from the outputs
    """
    def __call__(self, outputs):
        loss = utilities.to_value(outputs.get('loss'))
        if loss is not None:
            return {'loss': float(loss)}
        return None

    def aggregate_metrics(self, metric_by_batch):
        loss = 0.0
        for m in metric_by_batch:
            loss += m['loss']
        return {'loss': loss / len(metric_by_batch)}


class MetricClassificationBinaryAUC(Metric):
    """
    Calculate the Area under the Receiver operating characteristic (ROC) curve.

    For this, the output needs to provide an ``output_raw`` of shape [N, 2] (i.e., binary classification).
    """
    def __call__(self, outputs):
        truth = utilities.to_value(outputs.get('output_truth'))
        found = utilities.to_value(outputs.get('output_raw'))
        if truth is None or found is None:
            # data is missing
            return None

        if len(found.shape) != len(truth.shape) + 1 or len(found.shape) < 2 or found.shape[1] != 2:
            # dimensions are of the expected shape
            return None

        if len(found.shape) > 2:
            # TODO: implement for N-dimensions! We probably can't keep everything in memory
            return None

        return {
            'output_raw': found,
            'output_truth': truth,
        }

    def aggregate_metrics(self, metric_by_batch):
        all_output_raw = [m['output_raw'] for m in metric_by_batch]
        all_output_raw = np.concatenate(all_output_raw)
        all_output_truth = [m['output_truth'] for m in metric_by_batch]
        all_output_truth = np.concatenate(all_output_truth)
        auc = auroc(all_output_truth, all_output_raw[:, 1])
        return {'1-auc': 1.0 - auc}


class MetricClassificationError(Metric):
    """
    Calculate the ``1 - accuracy`` using the `output_truth` and `output`
    """
    def __call__(self, outputs):
        truth = utilities.to_value(outputs.get('output_truth'))
        found = utilities.to_value(outputs.get('output'))
        if truth is not None and found is not None:
            return collections.OrderedDict([
                ('nb_trues', np.sum(found == truth)),
                ('total', truth.size),  # for multi-dimension, use the size! (e.g., patch discriminator, segmentation)
            ])
        return None

    def aggregate_metrics(self, metric_by_batch):
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
    def __init__(self, dice_fn=losses.LossDiceMulticlass(normalization_fn=None, return_dice_by_class=True)):
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

    def aggregate_metrics(self, metric_by_batch):
        sum_dices = metric_by_batch[0]['dice_by_class'].copy()
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


class MetricClassificationF1(Metric):
    def __init__(self, average=None):
        """
        Calculate the Multi-class ``1 - F1 score``.

        Args:
            average: one of ``binary``, ``micro``, ``macro`` or ``weighted`` or None. If ``None``, use
                ``binary`` if only 2 classes or ``macro`` if more than two classes
        """
        self.average = average
        self.max_classes = 0

    def __call__(self, outputs):
        output_raw = utilities.to_value(outputs.get('output_raw'))
        if output_raw is None:
            return None
        if len(output_raw.shape) != 2:
            return None

        truth = utilities.to_value(outputs.get('output_truth'))
        if truth is None:
            return None

        self.max_classes = max(self.max_classes, output_raw.shape[1])
        found = np.argmax(output_raw, axis=1)
        return {
            'truth': truth,
            'found': found
        }

    def aggregate_metrics(self, metric_by_batch):
        truth = [m['truth'] for m in metric_by_batch]
        truth = np.concatenate(truth)
        found = [m['found'] for m in metric_by_batch]
        found = np.concatenate(found)

        if self.average is None:
            if self.max_classes <= 1:
                average = 'binary'
            else:
                average = 'macro'
        else:
            average = self.average

        score = 1.0 - metrics.f1_score(y_true=truth, y_pred=found, average=average)
        return {
            f'1-f1[{average}]': score
        }


class MetricClassificationBinarySensitivitySpecificity(Metric):
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

                return collections.OrderedDict([
                    ('tn', tn),
                    ('fn', fn),
                    ('fp', fp),
                    ('tp', tp),
                ])
            else:
                if truth[0] == 0:
                    # 0, means perfect classification of the negative
                    return collections.OrderedDict([
                        ('tn', cm[0, 0]),
                        ('fn', 0),
                        ('fp', 0),
                        ('tp', 0),
                    ])
                else:
                    # 1, means perfect classification of the positive
                    return collections.OrderedDict([
                        ('tp', cm[0, 0]),
                        ('fn', 0),
                        ('fp', 0),
                        ('tn', 0),
                    ])

        # something is missing, don't calculate the stats
        return None

    def aggregate_metrics(self, metric_by_batch):
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

        return collections.OrderedDict([
            # we return the 1.0 - metric, since in the history we always keep the smallest number
            ('1-sensitivity', one_minus_sensitivity),
            ('1-specificity', one_minus_specificity),
        ])


def default_classification_metrics():
    """"
    Default list of metrics used for classification
    """
    return [
        MetricLoss(),
        MetricClassificationError(),
        MetricClassificationBinarySensitivitySpecificity(),
        MetricClassificationBinaryAUC(),
        MetricClassificationF1(),
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
