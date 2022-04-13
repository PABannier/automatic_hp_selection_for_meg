import numpy as np

from sklearn.utils import assert_all_finite
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics._ranking import _binary_clf_curve


def delta_precision_recall_curve(y_true, probas_pred, y_true_extended, pos_label=None,
                                 sample_weight=None):
    """Compute delta-precision-recall curve."""
    _, tps, _ = _binary_clf_curve(y_true, probas_pred, pos_label=pos_label,
                                  sample_weight=sample_weight)
    recall = tps / tps[-1]
    delta_fps, delta_tps, thresholds = _binary_clf_curve(y_true_extended, probas_pred,
                                                         pos_label=pos_label,
                                                         sample_weight=sample_weight)
    delta_precision = delta_tps / (delta_tps + delta_fps)
    delta_precision[np.isnan(delta_precision)] = 0
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[delta_precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def _signed_class_clf_curve(y_true, y_score):
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not y_type == "multiclass":
        raise ValueError("{0} format is not supported".format(y_type))
    check_consistent_length(y_true, y_score)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)
    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if not (np.array_equal(classes, [-1, 0, 1])):
        raise ValueError("Classes should be [-1, 0, 1]")
    # make y_true a boolean vector
    y_true_abs = np.logical_or(y_true == -1, y_true == 1)
    y_score_abs = np.abs(y_score)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score_abs, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    y_score_abs = y_score_abs[desc_score_indices]
    y_true_abs = y_true_abs[desc_score_indices]
    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score_abs))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = []
    tp_last = 0
    false_sign = 0
    for i in range(y_true.size):
        if y_true_abs[i] == 0:
            pass
        elif y_true[i] == 1 and y_score[i] >= 0:
            tp_last += 1
        elif y_true[i] == -1 and y_score[i] <= 0:
            tp_last += 1
        else:
            false_sign += 1
        tps.append(tp_last)

    tps = np.asarray(tps)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = np.copy(y_score_abs[threshold_idxs])

    fps = np.append(fps, fps[-1])
    tps = np.append(tps, tps[-1] + false_sign)
    thresholds = np.append(thresholds, 0)

    return fps, tps, thresholds


def signed_class_precision_recall(y_true, signed_probas_pred):
    """Compute signed class precision recall at different threshold levels."""
    fps, tps, thresholds = _signed_class_clf_curve(y_true, signed_probas_pred)
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
