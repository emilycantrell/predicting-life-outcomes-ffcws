#!/usr/bin/env python3

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features_funcs import *


@dataclass
class MetaArgs:
    x: np.ndarray
    xbeforefs: pd.DataFrame
    y: pd.Series
    selected_features: pd.Index
    outcome: str
    outcome_type: str
    meta: pd.DataFrame


landmarks = {"1nn": at_1nn, "dt": at_dt}

statistical = {
    "corr": at_corr,
    "outliers": at_outlier,
    "outcomevariation": at_variation,
    "dispersion": at_dispersion,
}
general = {"numsamples": at_numsamples, "featdist": at_featdist}

complexity = {
    "dimension": at_dimension,
    "entropy": at_entropy,
    "condition": at_condition,
}

million_monkeys = {
    "wave": at_wavedist,
    "fp": at_fpdist,  # focal person distribution
    "topic": at_topicdist,
    "imputations": at_imputation,
    "source": at_source,
    "missing": at_missing,
}

attribute_classes = {
    "landmarks": landmarks,
    "general": general,
    "statistical": statistical,
    "complexity": complexity,
    "mm": million_monkeys,
}


def extract(args: MetaArgs):
    dct = args.__dict__
    attrs = {}
    for attr_class, funcs in attribute_classes.items():
        for fname, func in funcs.items():
            r = func(**dct)
            if isinstance(r, dict):
                for key, val in r.items():
                    attrs[f"{attr_class}_{fname}_{key}"] = val
            else:
                attrs[f"{attr_class}_{fname}"] = r
    # round features to 6 decimal and points and convert to floats for inserting
    # metafeature document into mongo database
    return dict([(k, float(round(v, 6))) for k, v in attrs.items() if v is not None])


if __name__ == "__main__":
    pass
