"""Generates test datasets (features, outcomes and metadata)
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


@dataclass
class Varmeta:
    new_name: str = None
    old_name: str = None
    type: str = None
    topics: str = None
    factors: str = 1
    wave: int = 1
    n_cities_asked: int = 20
    warning: str = ""


def gen_metadata():
    """Hardcoded function that generates metadata for variables"""
    return [
        Varmeta(new_name="c", type="Continuous", topics="foo", factors=1),
        Varmeta(new_name="o", type="Ordered Categorical", topics="bar", factors=3),
        Varmeta(new_name="u", type="Unordered Categorical", topics="bar", factors=3),
        Varmeta(new_name="b", type="Binary", topics="bar", factors=2),
    ]


def gen_data(nsamps, nfeatures, ntargets):
    return make_regression(
        n_samples=nsamps,
        n_features=nfeatures,
        n_informative=nfeatures,
        n_targets=2,
        coef=True,
    )


def char_range(num):
    for c in range(ord("a"), ord("a") + num):
        yield chr(c)


def create_df(arr, prefix="var_"):
    return pd.DataFrame(
        arr, columns=[prefix + "_" + x for x in char_range(arr.shape[1])]
    )


def add_index(df, key="ChallengeID", reverse=False):
    df[key] = range(len(df)) if not reverse else range(len(df))[::-1]
    return df


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def insert_na(X, y, xp=0.2, yp=0.1):
    """Insert NAs at random locations

    Args:
        X ([type]): [description]
        y ([type]): [description]
    """
    xmask = np.random.choice([True, False], size=X.shape, p=[xp, 1.0 - xp])
    ymask = np.random.choice([True, False], size=y.shape, p=[yp, 1.0 - yp])
    X = X.mask(xmask)
    y = y.mask(ymask)


def create_regression_dataset(args, path):
    X, y, coeff = gen_data(*[int(x) for x in args])
    # convert y vals to positive
    key = "ChallengeID"
    create_if_not_exists(path)
    X_df, y_df = (
        add_index(create_df(X, "ind"), key=key),
        add_index(create_df(y, "dep"), key=key),
    )
    metarows = []
    for var in X_df.columns.append(y_df.columns):
        if var != key:
            metarows.append(
                Varmeta(
                    new_name=var,
                    old_name=var,
                    type="Continuous",
                    topics="foo",
                    factors=1,
                )
            )
    meta_df = pd.DataFrame(metarows)
    insert_na(X_df, y_df)
    X_df.to_csv(os.path.join(path, "features.csv"), index=None, sep=",")
    y_df.to_csv(os.path.join(path, "outcomes.csv"), index=None, sep=",")
    meta_df.to_csv(os.path.join(path, "metadata.csv"), index=None, sep=",")
    coeff_df = pd.DataFrame(coeff)
    coeff_df.to_csv(os.path.join(path, "coeff.csv"), index=None, sep=",")


if __name__ == "__main__":
    create_regression_dataset(sys.argv[1:-1], sys.argv[-1])
