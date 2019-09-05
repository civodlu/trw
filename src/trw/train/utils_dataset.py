import numpy as np
import collections


def calculate_weight_by_class(split, output_classification_name, max_weight=None, normalize_by_max_weight=False):
    """
    Calculate the counts for each class and add calculate weights that compensate for class imbalances

    :param split: the data to be used
    :param output_classification_name: the classification name to be used to calculate the class frequencies
    :param max_weight: the maximum weight possible for a class (unnormalized). This is to avoid the case where
    we have an outlier (i.e., a class with several order of magnitude smaller that the other), we don't want to
    have an "infinite weight" for this class but an acceptable maximum instead. If None, there is no maximum
    weight
    :param normalize_by_max_weight: if False, the class with the highest number of instances will have a weight of 1.0,
    the others > 1.0. if True, same weight ratio as if False, but the class with minimum instances will have
    weight = 1.0, while the others will have < 1.0. Rational: it is safer to have weight < 1.0 so that we don't blow up
    the gradient
    :return: a dictionary for each class with a weight for the training (less counts means higher weight
    for the training to compensate)
    """
    output = split.get(output_classification_name)
    assert output is not None, 'output=%s could not be found in the split!' % output_classification_name
    assert not split.get('sample_weights'), 'there is already a weight defined for this dataset!'

    counts = collections.defaultdict(lambda: 0)
    assert len(output.shape) == 1
    for c in output:
        counts[c] += 1

    # the class with the maximum number of instances will be set to 1.0
    # the other classes will all have > 1.0, indicating how much emphasis required
    # for this class. Note that if we have many classes, but just a few have almost
    # no sample, we probably don't want to have a "crazy" strength for this class
    # hence max_weight
    assert len(counts) != 0, 'dataset split is empty!'
    max_counts = np.max(list(counts.values()))

    weights_by_class = {}
    for c, count in counts.items():
        weight = max_counts / count
        if max_weight is not None:
            weight = min(weight, max_weight)
        weights_by_class[c] = weight

    if normalize_by_max_weight:
        max_weight = max(list(weights_by_class.values()))
        assert max_weight > 0
        for classname, weight in weights_by_class.items():
            normalized_weight = weight / max_weight
            weights_by_class[classname] = normalized_weight

    return weights_by_class


def set_weight_scaled_by_inverse_class_frequency(
        split,
        output_classification_name,
        weights_by_class=None,
        max_weight=5,
        normalize_by_max_weight=False):
    """
    In classification tasks, we may not have a perfectly balanced dataset. This is problematic,
    in particular in highly unbalanced datasets, since the classifier may just end up learning
    the ratio of the classes and not what we really care about (i.e., how to discriminate).

    A possible solution is to weight the sample according to the inverse class distribution.

    :param max_weight: the maximum weight a class may have relative to the other classes
    :param split: the split to use
    :param output_classification_name: the name of the classification feature (i.e, must be an 1D array of class)
    :param weights_by_class: if specified, use the provided weight. If not, automatically calculate the weights
    using `calculate_weight_by_class`
    :param normalize_by_max_weight: if False, the class with the highest number of instances will have a weight of 1.0,
    the others > 1.0. if True, same weight ratio as if False, but the class with minimum instances will have
    weight = 1.0, while the others will have < 1.0. Rational: it is safer to have weight < 1.0 so that we don't blow up
    the gradient
    :return: the weights by class
    """
    assert not split.get('sample_weights'), 'there is already a weight defined for this dataset!'
    output = split.get(output_classification_name)
    assert output is not None, 'output=%s could not be found in the split!' % output_classification_name

    if weights_by_class is None:
        weights_by_class = calculate_weight_by_class(
            split,
            output_classification_name,
            max_weight=max_weight,
            normalize_by_max_weight=normalize_by_max_weight)
    weights = np.ndarray(shape=[len(output)], dtype=np.float32)
    for index, c in enumerate(output):
        weights[index] = weights_by_class[c]
    split['sample_weights'] = weights
    return weights_by_class
