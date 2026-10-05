#!/usr/bin/env python3
import contextlib
import hashlib
import itertools as it
import json
import logging
import os
import pathlib
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyreadstat
from pyod.models.copod import COPOD
from sklearn.preprocessing import KBinsDiscretizer

from .track_shapes import TrackInfo, add

RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)


@dataclass(frozen=True, eq=True)
class DataStats:
    num_samples: int
    num_unique: int
    num_without_duplicates: int
    num_copod_outliers: int


def get_data_stats(X_train, X_test, outlier_score=100):
    """Get number of unique samples, outliers in training and testing data"""
    df_train = pd.DataFrame(X_train)
    df_test = pd.DataFrame(X_test)
    # COPOD cannot use parallel scoring (n_jobs != 1) on single-feature data.
    # When there is only one feature (e.g., k-best-1), force single-threaded mode.
    # For multi-feature data, enable parallelism.
    n_features = df_train.shape[1]
    if n_features == 1:
        clf = COPOD(n_jobs=1)
    else:
        clf = COPOD(n_jobs=-1)
    clf.fit(df_train.drop_duplicates(keep="first"))
    train_stats = DataStats(
        num_samples=len(df_train),
        num_unique=len(df_train.drop_duplicates(keep="first")),
        num_without_duplicates=len(df_train.drop_duplicates(keep=False)),
        num_copod_outliers=int((clf.decision_scores_ > outlier_score).sum()),
    )
    test_stats = DataStats(
        num_samples=len(df_test),
        num_unique=len(df_test.drop_duplicates(keep="first")),
        num_without_duplicates=len(df_test.drop_duplicates(keep=False)),
        num_copod_outliers=int(
            (
                clf.decision_function(df_test.drop_duplicates(keep="first"))
                > outlier_score
            ).sum()
        ),
    )
    return train_stats.__dict__, test_stats.__dict__


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")


def get_git_revision_short_hash():
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode("ascii")
    )


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


DICT_FIELD_SEP = "."


def conf_get(conf, nested_field_name):
    cp = conf.copy()
    sp = nested_field_name.split(DICT_FIELD_SEP)
    for field in sp:
        cp = cp[field]
    return cp


def zip_list(A, B):
    return zip(A, cycle(B)) if len(A) > len(B) else zip(cycle(A), B)


def is_all_enabled(item):
    return (isinstance(item, str) and item == "all") or (
        isinstance(item, list) and "all" in item
    )


def to_categorical(y, bin_method="sturges", **kwargs):
    """Convert an array with continuous values to categorical"""
    if bin_method == "quantile":
        dis = KBinsDiscretizer(
            n_bins=kwargs["param"], encode="ordinal", strategy="quantile"
        )
        y_cat = dis.fit_transform(pd.DataFrame(y))
        edges = dis.bin_edges_
    else:
        edges = np.histogram_bin_edges(y, bins=bin_method)
        y_cat = np.digitize(y, edges)
    return y_cat, edges


def get_logger(name, level=20):
    """Return a logger instance without setting handlers

    Levels:

    Critical, error, warning, info, debug, notset : range(50, 0, -10)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def read_csv(fpath, sep):
    return pd.read_csv(fpath, low_memory=False, sep=sep, engine="c")


def read(fpath: pathlib.Path, read_as_str: bool = False):
    """Read file based on the extension"""
    if fpath.suffix == ".csv":
        df = (
            read_csv(fpath, ",")
            if not read_as_str
            else pd.read_csv(fpath, sep=",", dtype=str)
        )
        return df, None
    elif fpath.suffix == ".dta":
        df, meta = pyreadstat.read_dta(str(fpath), encoding="LATIN1")
        return df, meta


def get_pobj(opt):
    project_root = Path(__file__).parent.parent
    path, md5sum = opt
    pobj = project_root.joinpath("data", path)
    return pobj, md5sum


def get_concept_json(concept_opt):
    if concept_opt:
        pobj, _ = get_pobj(concept_opt)
        js = json.loads(open(pobj).read())
        return js


def all_disjoint(x):
    return all((set(p0).isdisjoint(set(p1))) for p0, p1 in it.combinations(x, 2))


"""
1. Following is mapping from warning messages in the metadata file to warning codes.
It doesn't contain warning messages that occur only in restricted variables.
2. Warning string corresponding to code 1 has a typo in the v07 of the metadata file.
To handle the case where this gets fixed in the future versions of the metadata file,
we map two versions (with and without typo) of the warning string to same code (1)
"""
warning_strings = {
    "Variable uses Stata extended missing values, these may display differently in other programs": 0,
    "Misordered Categorical (outcomes do not have a contstant scale)": 1,
    "Misordered Categorical (outcomes do not have a constant scale)": 1,
    "Variable has outcomes which override a continuous answer set": 2,
    "A unique outcome is coded as a negative value": 4,
    "Missing data is coded as something other than the default": 5,
    "A survey Yes/No variable that has 'No' coded to 0 instead of 2": 6,
    "Variable contains values rounded into buckets, see value label": 7,
}


def get_warning_cols(meta, collist, codes):
    """Returns a subset of given columns for which metadata file has a non empty string in the warning field"""
    warning_cols = (
        meta.loc[collist]
        .warning.apply(lambda x: x in warning_strings and warning_strings[x] in codes)
        .where(lambda x: x)
        .dropna()
        .index
    )
    return warning_cols


def is_mode_hyperparameter_tuning(opt):
    return opt == "hyperparameter_tuning"


def is_model_tunable(model):
    return hasattr(model, "is_tunable") and model.is_tunable()


def is_imbalanced_enabled(opt):
    return opt != "skip"


def dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def roundoff_zeros(mat, inplace=True):
    rmat = mat * (~np.isclose(mat, 0))
    if inplace:
        mat = rmat
    return rmat


DROPPED_COLS = defaultdict(list)


def track_dropped_cols(col_type, category, outcome_list, msg):
    DROPPED_COLS[category].append((col_type, outcome_list, msg))


def get_dropped_cols():
    return DROPPED_COLS


def get_best_index(df, prefix):
    iters = [
        int(x.split("_")[-1]) if x.startswith(f"{prefix}_best_iter_") else -1
        for x in df.columns
    ]
    iters.sort()
    max_iter = f"{prefix}_best_iter_{iters[-1]}"
    best_index = df[max_iter].dropna().values
    return best_index


_UNORDERD_CATEGORICAL_ENCODING_METHODS = ["one-hot", "dummy_largest"]


def is_encoding_opt_valid(opt):
    return opt in _UNORDERD_CATEGORICAL_ENCODING_METHODS


def encoding_get_first(df):
    return df.groupby(lambda x: x.rsplit("_", 1)[0], axis=1).apply(
        lambda x: x.columns[0]
    )


def encoding_get_largest(df):
    return df.groupby(lambda x: x.rsplit("_", 1)[0], axis=1).apply(
        lambda x: x.sum().idxmax()
    )


def get_cvfold_ystats(y, train_index, test_index):
    return {
        "train": y.iloc[train_index].agg(["mean", "var"]).to_dict(),
        "test": y.iloc[test_index].agg(["mean", "var"]).to_dict(),
    }


def indicator_standard_missingness(x):
    """Return true if input is a negative integer in the interval [-9, -1]"""
    return x in [-9, -8, -7, -6, -5, -4, -3, -2, -1]


def drop_and_track(mat, cols, module, mat_type, col_type, msg):
    mat = drop_cols(mat, cols, mat_type, col_type, msg)
    add(TrackInfo(module, mat_type, mat.shape, msg))
    return mat


def drop_cols(df, cols, col_type, cat, msg):
    common_cols = df.columns.intersection(cols)
    new_df = df.drop(common_cols, axis=1)
    track_dropped_cols(col_type, cat, common_cols, msg)
    return new_df
