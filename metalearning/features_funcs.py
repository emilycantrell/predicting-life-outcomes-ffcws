#!/usr/bin/env python3
from collections import defaultdict

import numpy as np
import pandas as pd
import skdim
from scipy.stats import entropy, pearsonr
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

summary_funcs = {"max": np.max, "mean": np.mean, "min": np.min}
missing_codes = {"not-in-wave": -9, "dont-know": -2, "skip": -6, "missing": -3}
UNDERSCORE = "_"


def _get_meta(**kwargs):
    return kwargs["meta"]


def _is_mask(x):
    return x.startswith("mask_")


def _is_not_level(x, metadata):
    base = _get_base_name(x, metadata)
    return x.endswith(base)


def get_r2(x, y, model):
    model.fit(x, y)
    predictions = model.predict(x)
    score = r2_score(y, predictions)
    return score


def at_1nn(**kwargs):
    neigh = KNeighborsRegressor(n_neighbors=2)
    return {"r2": get_r2(kwargs["x"], kwargs["y"], neigh)}


def at_dimension(**kwargs):
    dim = skdim.id.lPCA()
    dim.fit(kwargs["x"])
    return {"x": dim.dimension_}


def at_condition(**kwargs):
    design_matrix = kwargs["x"]
    is_singular = False
    try:
        np.linalg.inv(np.dot(design_matrix.T, design_matrix))
    except np.linalg.linalg.LinAlgError:
        is_singular = True
    cond_num = np.linalg.cond(design_matrix)
    return {"number": cond_num, "singular": is_singular}


def at_entropy(**kwargs):
    return {"y": entropy(kwargs["y"].value_counts(normalize=True))}


def at_dt(**kwargs):
    regressor = DecisionTreeRegressor(max_depth=1)
    return {"r2": get_r2(kwargs["x"], kwargs["y"], regressor)}


def at_corr(**kwargs):
    corr = lambda x, y: abs(pearsonr(x, y)[0])
    feat_corr = [corr(kwargs["x"][col], kwargs["y"]) for col in kwargs["x"].columns]
    return dict(
        [(f"{funcname}", func(feat_corr)) for funcname, func in summary_funcs.items()]
    )


def at_numsamples(**kwargs):
    return kwargs["x"].shape[0]


def _get_feat(x, metadata):
    if _is_mask(x):
        return x.split("_")[1]
    elif "_" in x:
        return "_".join(x.split("_")[:-1])
    return x


def _get_base_name(x, metadata):
    if "_" not in x:
        return x
    elif _is_mask(x):
        return _get_base_name(x.split("_", 1)[1], metadata)
    elif "_" in x:
        base = "_".join(x.split("_")[:-1])
        return base if base in metadata.index else x


def _get_unord_code(x, metadata):
    """Get one hot encoding level"""
    base = _get_base_name(x, metadata)
    if base == x or _is_mask(x):  # this is never true for unordered categorical
        return None
    try:
        return float(x.split(UNDERSCORE)[-1])
    except ValueError:
        return None


def _get_feature_meta(**kwargs):
    metadata = _get_meta(**kwargs)
    feats = [_get_base_name(x, metadata) for x in kwargs["selected_features"]]
    submeta = metadata.loc[
        [_get_base_name(x, metadata) for x in kwargs["selected_features"]]
    ]
    return submeta, feats


def at_wavedist(**kwargs):
    submeta, feats = _get_feature_meta(**kwargs)
    return submeta.wave.value_counts(normalize=True).to_dict()


def at_fpdist(**kwargs):
    submeta, feats = _get_feature_meta(**kwargs)
    return (submeta.filter(regex="fp_", axis=1).sum() / len(submeta)).to_dict()


def at_topicdist(**kwargs):
    submeta, feats = _get_feature_meta(**kwargs)
    return submeta.topics.value_counts(normalize=True).to_dict()


def at_featdist(**kwargs):
    submeta, _ = _get_feature_meta(**kwargs)
    return dict(
        [
            ("".join(k.split()), v)
            for k, v in submeta.type.value_counts(normalize=True).to_dict().items()
        ]
    )


def _get_num_imputations(metadata, mask_df, prefix):
    num_imputations = mask_df.sum() / len(mask_df)
    types = metadata.loc[mask_df.columns].type
    cdf = pd.concat({"imputations": num_imputations, "types": types}, axis=1)
    imp_counts = (
        cdf.groupby("types")
        .apply(
            lambda x: (x.imputations.min(), x.imputations.mean(), x.imputations.max())
        )
        .to_dict()
    )
    keys = ["min", "mean", "max"]

    def f():
        for t, counts in imp_counts.items():
            for k, v in zip(keys, counts):
                yield (f"{prefix}{t}_{k}", v)

    dct_mask = dict(x for x in f())
    return dct_mask


def _get_raw_sub(**kwargs):
    xbeforefs = kwargs["xbeforefs"]
    selected_features = kwargs["selected_features"]
    xraw_sub = xbeforefs[[x for x in selected_features if not x.startswith("mask_")]]
    return xraw_sub


def _outlier(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).mean()


def at_outlier(**kwargs):
    xraw_sub = _get_raw_sub(**kwargs)
    metadata = _get_meta(**kwargs)
    notna = xraw_sub.mask((xraw_sub <= -1) & (xraw_sub >= -9), np.nan)
    outliers = _outlier(notna)
    outliers = outliers[[x for x in outliers.index if _is_not_level(x, metadata)]]
    return dict(
        [
            (f"{funcname}", func(outliers) if len(outliers) > 0 else 0)
            for funcname, func in summary_funcs.items()
        ]
    )


def at_dispersion(**kwargs):
    xraw_sub = _get_raw_sub(**kwargs)
    notna = xraw_sub.mask((xraw_sub <= -1) & (xraw_sub >= -9), np.nan)
    metadata = _get_meta(**kwargs)
    notna = notna[[x for x in notna.columns if _is_not_level(x, metadata)]]
    notna_std = dict(
        [
            (f"std{funcname}", func(notna.std()))
            for funcname, func in summary_funcs.items()
        ]
    )
    notna_var = dict(
        [
            (f"var{funcname}", func(notna.var()))
            for funcname, func in summary_funcs.items()
        ]
    )
    return {**notna_std, **notna_var}


def at_variation(**kwargs):
    y = kwargs["y"]
    return y.std() / y.mean()


def at_imputation(**kwargs):
    xraw_sub = _get_raw_sub(**kwargs)
    xbeforefs = kwargs["xbeforefs"]
    selected_features = kwargs["selected_features"]
    metadata = _get_meta(**kwargs)
    mask_df = (
        xbeforefs[selected_features]
        .filter(regex="mask_", axis=1)
        .rename(lambda x: x.split("_", 1)[1], axis=1)
    )
    mask_dct = _get_num_imputations(metadata, mask_df, "mask")

    # number of missing for non mask variables
    na_counts = xraw_sub.mask((xraw_sub <= -1) & (xraw_sub >= -9), np.nan).isna()
    unmask_dct = _get_num_imputations(metadata, na_counts, "unmask")
    return {**mask_dct, **unmask_dct}


def at_source(**kwargs):
    metadata = _get_meta(**kwargs)
    selected_features = kwargs["selected_features"]
    if "source" in metadata:
        return (
            metadata.loc[[_get_base_name(x, metadata) for x in selected_features]]
            .source.value_counts(normalize=True)
            .to_dict()
        )


def at_missing(**kwargs):
    xraw_sub = _get_raw_sub(**kwargs)
    metadata = _get_meta(**kwargs)
    counts = defaultdict(list)
    for codename, code in missing_codes.items():
        for col in xraw_sub.columns:
            varcode = _get_unord_code(col, metadata)
            if not varcode and not _is_mask(col):
                cnt = (xraw_sub[col] == code).sum()
            elif varcode == code:
                cnt = xraw_sub[col].sum()
            else:
                cnt = 0
            if cnt > 0:
                counts[codename].append(cnt)
    dct = {}
    for summary_name, func in summary_funcs.items():
        for codename, cnts in counts.items():
            dct[f"{codename}{summary_name}"] = (
                func(cnts) / len(xraw_sub) if len(cnts) > 0 else 0
            )
    return dct
