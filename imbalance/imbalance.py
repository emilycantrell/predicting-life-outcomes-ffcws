#!/usr/bin/env python3
import pandas as pd
from imblearn.over_sampling import SMOTE
from utils import rng


def resample_train(x_train, y_train, outcome_type, imbalance_opt):
    """Return a resampled version of the training data"""
    
    def wrap_df(x_res, y_res):
        return pd.DataFrame(x_res), pd.Series(y_res)

    if imbalance_opt["mode"] == "skip" or outcome_type == "Continuous":
        return False, wrap_df(x_train, y_train)
    nnbrs = imbalance_opt["param"]["k_neighbors"]
    if not all(y_train.value_counts() > nnbrs):
        # if n_samples <= nnbrs then SMOTE throws an error
        return False, wrap_df(x_train, y_train)
    sm = SMOTE(random_state=rng, k_neighbors=nnbrs)
    x_res, y_res = sm.fit_resample(x_train, y_train)
    return True, wrap_df(x_res, y_res)
