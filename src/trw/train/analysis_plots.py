"""
Defines the main plots and reports used for the analysis of our models
"""

import matplotlib.pyplot as plt
import numpy as np
from trw.train import utilities
import collections
from textwrap import wrap
import itertools
import sklearn.metrics
import logging
import numbers
import os
import matplotlib.patches


logger = logging.getLogger(__name__)


def fig_tight_layout(fig):
    """
    Make the figures not overlapping
    """
    try:
        fig.tight_layout()
    except Exception as e:
        logger.error('tight_layout failed=%s' % str(e))


def auroc(trues, found_1_scores):
    """
    Calculate the area under the curve of the ROC plot (AUROC)

    :param trues: the expected class
    :param found_1_scores: the score found for the `class 1`. Must be a numpy array of floats
    :return: the AUROC
    """
    assert len(trues) == len(found_1_scores)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=trues, y_score=found_1_scores, pos_label=None)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    return roc_auc


def export_figure(path, name, maximum_length=259, dpi=300):
    """
    Export a figure

    :param path: the folder where to export the figure
    :param name: the name of the figure.
    :param maximum_length: the maximum length of the full path of a figure. If the full path name is greater than `maximum_length`, the `name` will be subs-ampled to the maximal allowed length
    :param dpi: Dots Per Inch: the density of the figure
    """
    figure_name = utilities.safe_filename(name) + '.png'

    if len(path) >= maximum_length:
        logger.error('path is too long=', path)
        return

    full_path = os.path.join(path, figure_name)
    if len(full_path) >= maximum_length:
        # the name is too long! subsample the name so that it fits in the maximum path length
        nb_chars = maximum_length - len(path) - 4 - 1  # we need to keep extension so forbid to sample the last 4 characters + separator
        ids = np.asarray(list(range(len(figure_name) - 4)))
        np.random.shuffle(ids)
        sorted_ids = sorted(ids[:nb_chars])
        new_name = ''
        for i in sorted_ids:
            new_name += figure_name[i]
        figure_name = new_name + figure_name[-4:]
        full_path = os.path.join(path, figure_name)

    plt.savefig(full_path, dpi=dpi)


def boxplots(
    export_path,
    features_trials,
    title,
    xlabel,
    ylabel,
    meanline=False,
    plot_trials=True,
    scale='linear',
    y_range=None,
    rotate_x=None,
    showfliers=False,
    maximum_chars_per_line=50,
    title_line_height=0.055):
    """
    Compare different histories: e.g., compare 2 configuration, which one has the best results for a given
    measure?

    :param export_path: where to export the figure
    :param features_trials: a dictionary of list. Each list representing a feature
    :param title: the title of the plot
    :param ylabel: the label for axis y
    :param xlabel: the label for axis x
    :param meanline: if True, draw a line from the center of the plot for each history name to the next
    :param maximum_chars_per_line: the maximum of characters allowed per line of title. If exceeded, newline will be created.
    :param plot_trials: if True, each trial of a feature will be plotted
    :param scale: the axis scale to be used
    :param y_range: if not None, the (min, max) of the y-axis
    :param rotate_x: if not None, the rotation of the x axis labels in degree
    :param showfliers: if True, plot the outliers
    :param maximum_chars_per_line: the maximum number of characters of the title per line
    :param title_line_height: the height of the title lines
    """
    assert isinstance(features_trials, collections.Mapping), 'must be a dictionary of list'
    assert isinstance(next(iter(features_trials.keys())), collections.Iterable), 'each feature must be iterable'

    labels = []
    series = []
    for features_name, trials in features_trials.items():
        labels.append(features_name)
        series.append(trials)

    # for very long title, split it!
    title_lines = list(wrap(title, maximum_chars_per_line))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(series,
               labels=labels,
               positions=range(0, len(features_trials)),
               showfliers=showfliers,
               widths=0.4)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_title('\n'.join(wrap(title, maximum_chars_per_line)), fontsize=20)

    ax.set_yscale(scale)
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.grid(which='both', axis='y', linestyle='--')

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
        if rotate_x is not None:
            tick.label1.set_rotation(rotate_x)

    if plot_trials:
        for index, values in enumerate(series):
            y = values
            # Add some random "jitter" to the x-axis
            x = np.random.normal(index, 0.01, size=len(y))
            plt.plot(x, y, 'r.')

    if meanline:
        means = [np.mean(values) for values in features_trials.values()]
        lines_x = []
        lines_y = []
        for index in range(len(means) - 1):
            lines_x.append(index)
            lines_y.append(means[index])
            lines_x.append(index + 1)
            lines_y.append(means[index + 1])
        ax.plot(lines_x, lines_y)

    fig_tight_layout(fig)
    fig.subplots_adjust(top=1.0 - len(title_lines) * title_line_height)

    export_figure(export_path, title)
    plt.close()


def plot_roc(export_path, trues, found_scores_1, title, label_name=None, colors=None):
    """
    Calculate the ROC and AUC of a binary classifier

    Supports multiple ROC curves.

    :param export_path: the folder where the plot will be exported
    :param trues: the expected class. Can be a list for multiple ROC curves
    :param found_scores_1: the score found for the prediction of class `1`. Must be a numpy array of floats. Can be a list for multiple ROC curves
    :param title: the title of the ROC
    :param label_name: the name of the ROC curve. Can be a list for multiple ROC curves
    :param colors: if None use default colors. Else, a numpy array of dim (Nx3) where `N` is the number of colors. Must be in [0..1] range
    """
    
    if not isinstance(trues, list):
        trues = [trues]
        found_scores_1 = [found_scores_1]
        assert not isinstance(label_name, list), 'must nobe be a list! only if `found` is a list'
        label_name = [label_name]
    else:
        assert len(trues) == len(found_scores_1), 'must have 1 true for 1 found'
        if label_name is not None:
            assert isinstance(label_name, list)
            assert len(label_name) == len(found_scores_1)
        else:
            label_name = [None] * len(found_scores_1)
            
    if np.min(trues[0]) == np.max(trues[0]):
        logger.error('`trues` has only one class! We can\'t have a meaningful ROC')
        return

    fig = plt.figure()
    if colors is None:
        colors = utilities.make_unique_colors()
        colors = np.asarray(colors) / 255
    assert len(colors) >= len(found_scores_1), 'TODO define more colors!'
    cutoff = None
    for curve_n, found_values in enumerate(found_scores_1):
        true_values = trues[curve_n]

        assert len(true_values) == len(found_values)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=true_values, y_score=found_values, pos_label=None)
        roc_auc = sklearn.metrics.auc(fpr, tpr)

        def cutoff_youdens_j(fpr, tpr, thresholds):
            j_scores = tpr - fpr
            j_ordered = sorted(zip(j_scores, thresholds))
            return j_ordered[-1][1]

        cutoff = cutoff_youdens_j(fpr, tpr, thresholds)
        if cutoff != cutoff_youdens_j(fpr, tpr, thresholds):
            print('WARNING! manual threshold!')

        y_cutoff = found_values >= cutoff
        accuracy_cutoff = sum(true_values == y_cutoff) / len(true_values)
        cm_cutoff = sklearn.metrics.confusion_matrix(true_values, y_cutoff)
        tn, fp, fn, tp = cm_cutoff.ravel()
        print('title=%s, accuracy_cutoff=%f, cutoff=%f' % (title, accuracy_cutoff, cutoff))
        print('Confusion_matrix cutoff:\n', cm_cutoff)
        print('specificity cutoff:\n', tn / (tn + fp))
        print('sensitivity cutoff:\n', tp / (tp + fn))

        name = label_name[curve_n]
        if name is None:
            name = ''
        color = colors[curve_n]
        plt.plot(fpr, tpr, color=color, lw=2, label='ROC %s (AUC = %0.3f)' % (name, roc_auc))

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    fig_tight_layout(fig)
    export_figure(export_path, title)
    plt.close()
    return cutoff


def list_classes_from_mapping(mappinginv: collections.Mapping, default_name='unknown'):
    """
    Create a contiguous list of label names ordered from 0..N from the class mapping

    :param mappinginv: a dictionary like structure encoded as (class id, class_name)
    :param default_name: if there is no class name, use this as default
    :return: a list of class names ordered from class id = 0 to class id = N. If `mappinginv` is None,
        returns None
    """
    if mappinginv is None:
        return None
    nb_classes = max(mappinginv.keys()) + 1
    classes = [default_name] * nb_classes
    for class_id, name in mappinginv.items():
        classes[class_id] = name
    return classes


def classification_report(
        prediction_scores: np.ndarray,
        trues: collections.Sequence,
        class_mapping: collections.Mapping=None):
    """
    Summarizes the important statistics for a classification problem
    :param prediction_scores: the scores for each, for each sample
    :param trues: the true class for each sample
    :param class_mapping: the class mapping (class id, class name)
    :return: a dictionary of statistics or sub-report
    """
    assert isinstance(trues[0], numbers.Integral), 'must be a list of classes'
    predictions = np.argmax(prediction_scores, axis=1)

    cm = sklearn.metrics.confusion_matrix(y_pred=predictions, y_true=trues)
    labels = list_classes_from_mapping(class_mapping)
    try:
        report_str = sklearn.metrics.classification_report(y_true=trues, y_pred=predictions, target_names=labels)
    except ValueError as e:
        report_str = 'Report failed. Exception={}'.format(e)
        
    accuracy = float(np.sum(predictions == trues)) / len(trues)

    d = collections.OrderedDict()
    d['accuracty'] = accuracy
    d['sklearn_report'] = report_str

    print_options = np.get_printoptions()
    np.set_printoptions(threshold=1000000, linewidth=1000)
    d['confusion_matrix'] = str(cm)
    np.set_printoptions(**print_options)

    if len(cm) == 2:
        # special case: binary classification
        tn, fp, fn, tp = cm.ravel()

        # Sensitivity, hit rate, recall, or true positive rate
        d['tpr'] = tp / (tp + fn)
        # Specificity or true negative rate
        d['tnr'] = tn / (tn + fp)
        # Precision or positive predictive value
        d['ppv'] = tp / (tp + fp)
        # Negative predictive value
        d['npv'] = tn / (tn + fn)
        # Fall out or false positive rate
        d['fpr'] = fp / (fp + tn)
        # False negative rate
        d['fnr'] = fn / (tp + fn)
        # False discovery rate
        d['pdr'] = fp / (tp + fp)

        d['sensitivity'] = tp / (tp + fn)
        d['specificity'] = tn / (fp + tn)

        prediction_1_scores = prediction_scores[:, 1]
        d['auroc'] = auroc(trues=trues, found_1_scores=prediction_1_scores)

    # calculate the most common errors
    error_by_class = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for found, groundtruth in zip(predictions, trues):
        if found != groundtruth:
            error_by_class[groundtruth][found] += 1

    for error_id, (groundtruth, errors) in enumerate(error_by_class.items()):
        sorted_errors = [(k, v) for k, v in errors.items()]
        sorted_errors = list(reversed(sorted(sorted_errors, key=lambda v: v[1])))

        if class_mapping is not None:
            c = class_mapping[groundtruth]
        else:
            c = groundtruth
        largest_errors = []
        for key, nb in sorted_errors[:5]:
            if class_mapping is not None:
                key_name = class_mapping[key]
            else:
                key_name = key
            largest_errors.append((key_name, nb))

        class_str = 'class (total=%d)=%s, errors=%s\n' % (np.sum(trues == groundtruth), str(c), str(largest_errors))
        d['error_%d' % error_id] = class_str
    return d


def plot_group_histories(root, history_values, title, xlabel, ylabel, max_nb_plots_per_group=5, colors=utilities.make_unique_colors_f()):
    """
    Plot groups of histories
    :param root: the directory where the plot will be exported
    :param history_values: a map of list of list of (epoch, value)
    :param title: the title of the graph
    :param xlabel: the x label
    :param ylabel: the y label
    :param max_nb_plots_per_group: the maximum number of plots per group
    :param colors: the colors to be used
    """
    if len(history_values) == 0:
        return
    assert isinstance(history_values, collections.Mapping), 'must be a dictionary of lists of values'
    assert isinstance(next(iter(history_values.items()))[1], list), 'must be a dictionary of lists of values'
    assert len(history_values) <= len(colors), 'not enough colors!'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    patches = []
    for group_index, (name, values_list) in enumerate(history_values.items()):
        color = colors[group_index]
        patches.append(matplotlib.patches.Patch(color=color, label=name))

        for list_index, values in enumerate(values_list):
            i = []
            v = []
            for index, value in values:
                i.append(index)
                v.append(value)

            ax.plot(i, v, color=color)
            if index >= max_nb_plots_per_group:
                break

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.legend(handles=patches, loc='upper right')
    fig_tight_layout(fig)
    export_figure(root, title)
    plt.close()


def confusion_matrix(
        export_path,
        classes_predictions,
        classes_trues,
        classes: list=None,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.plasma,
        display_numbers=True,
        maximum_chars_per_line=50,
        rotate_x=None,
        rotate_y=None,
        display_names_x=True,
        sort_by_decreasing_sample_size=True,
        excludes_classes_with_samples_less_than=None,
        main_font_size=16,
        sub_font_size=8,
        normalize_unit_percentage=False):
    """
    Plot the confusion matrix of a predicted class versus the true class

    :param export_path: the folder where the confusion matrix will be exported
    :param classes_predictions: the classes that were predicted by the classifier
    :param classes_trues: the true classes
    :param classes: a list of labels. Label 0 for class 0, label 1 for class 1...
    :param normalize: if True, the confusion matrix will be normalized to 1.0 per row
    :param title: the title of the plot
    :param cmap: the color map to use
    :param display_numbers: if True, display the numbers within each cell of the confusion matrix
    :param maximum_chars_per_line: the title will be split every `maximum_chars_per_line` characters to avoid display issues
    :param rotate_x: if not None, indicates the rotation of the label on x axis
    :param rotate_y: if not None, indicates the rotation of the label on y axis
    :param display_names_x: if True, the class name, if specified, will also be displayed on the x axis
    :param sort_by_decreasing_sample_size: if True, the confusion matrix will be sorted by decreasing number of samples. This can
        be useful to show if the errors may be due to low number of samples
    :param excludes_classes_with_samples_less_than: if not None, the classes with less than `excludes_classes_with_samples_less_than` samples will be excluded
    :param normalize_unit_percentage: if True, use 100% base as unit instead of 1.0
    :param main_font_size: the font size of the text
    :param sub_font_size: the font size of the sub-elements (e.g., ticks)
    """
    if classes is not None:
        assert max(classes_trues) <= len(classes), 'there are more classes than class names!'

    if sort_by_decreasing_sample_size:
        def remap(class_ids, mapping):
            return [mapping[class_id] for class_id in class_ids]

        # first, calculate the most classes with the highest number of samples
        class_samples = collections.Counter(classes_trues)
        sorted_classes = sorted(list(class_samples.items()), key=lambda t: t[1], reverse=True)
        new_mappings = {}
        for new_mapping, (old_mapping, nb_samples) in enumerate(sorted_classes):
            new_mappings[old_mapping] = new_mapping

        if classes is not None:
            new_classes = []
            for new_mapping, (old_mapping, nb_samples) in enumerate(sorted_classes):
                new_classes.append(classes[old_mapping])
            classes = new_classes

        # now re-map the original classes
        classes_predictions = remap(classes_predictions, new_mappings)
        classes_trues = remap(classes_trues, new_mappings)

    classes_predictions = np.asarray(classes_predictions)
    classes_trues = np.asarray(classes_trues)

    if sort_by_decreasing_sample_size and excludes_classes_with_samples_less_than is not None:
        # IMPORTANT: we must have sorted by size before excluding the classes with low number of samples!
        # else the `classes` will not be consitent with the class id
        class_samples = collections.Counter(classes_trues)
        indices_to_keep = set()
        classes_to_keep = []
        for class_id, num_samples in class_samples.items():
            if num_samples >= excludes_classes_with_samples_less_than:
                indices = np.where(classes_trues == class_id)
                if len(indices) != 0:
                    indices_to_keep = indices_to_keep.union(indices[0])
                classes_to_keep.append(class_id)

        # keep only tha classes that satisfy the criteria
        indices_to_keep = list(indices_to_keep)
        classes_predictions = classes_predictions[indices_to_keep]
        classes_trues = classes_trues[indices_to_keep]
        if classes is not None:
            classes = np.asarray(classes)[classes_to_keep]

    cm = sklearn.metrics.confusion_matrix(y_pred=classes_predictions, y_true=classes_trues)
    if normalize:
        unit = 1.0
        if normalize_unit_percentage:
            unit = 100.0
        cm = unit * cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-3)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmax=np.max(cm), vmin=0.0000001)
    ax.set_title('\n'.join(wrap(title, maximum_chars_per_line)), fontsize=main_font_size)
    fig.colorbar(cax)

    if classes is not None:
        tick_marks = np.arange(len(classes))
        if display_names_x:
            plt.xticks(tick_marks, classes, rotation=rotate_x, fontsize=sub_font_size)
        plt.yticks(tick_marks, classes, rotation=rotate_y, fontsize=sub_font_size)

    if display_numbers:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('Predicted label', fontsize=main_font_size)
    ax.set_xlabel('True label', fontsize=main_font_size)
    fig_tight_layout(fig)
    export_figure(export_path, title)
    plt.close()
