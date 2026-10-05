import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    zero_one_loss
)


def compute_error(func, truth, prediction, ytrain, custom=False):
    try:
        if custom:
            error = func(truth, prediction, ytrain)
        else:
            error = func(truth, prediction)
    except ValueError:
        error = None
    return error


def my_r2_score(truth, prediction, ytrain):
    return 1.0 - np.mean((truth - prediction) ** 2) / np.mean(
        (truth - ytrain.mean()) ** 2
    )


def my_mse_baseline(truth, prediction, ytrain):
    """MSE of baseline model"""
    return np.mean((truth - ytrain.mean()) ** 2)


def errormetrics(truth, prediction, ytrain, is_classification=False):
    if not is_classification:
        names = [
            ("r2_score", True),
            ("MSE_baseline", True),
            ("MSE", False),
            ("MAE", False),
            ("explained_variance", False),
        ]
        funcs = [
            my_r2_score,
            my_mse_baseline,
            mean_squared_error,
            mean_absolute_error,
            explained_variance_score,
        ]
    else:
        names = ["accuracy_score", "balanced accuracy", "f1_score", "zero_one_loss"]
        funcs = [accuracy_score, balanced_accuracy_score, f1_score, zero_one_loss]
    ret = []
    for (metricname, custom), func in zip(names, funcs):
        ret.append((metricname, compute_error(func, truth, prediction, ytrain, custom)))
    return ret
