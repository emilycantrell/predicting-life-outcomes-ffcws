import logging
import types
import warnings
from exceptions import FeatureSelectionException
from itertools import product

import numpy as np
import pandas as pd
import sympy
from preprocessing import get_scaler_by_key
from scipy.stats import spearmanr
from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.feature_selection import (
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
)
from utils import roundoff_zeros

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

_num_best = [5, 10, 15, 20, 50]
_selection_modes = {
    "percentile": SelectPercentile,
    "k_best": SelectKBest,
    "fpr": SelectFpr,
    "fdr": SelectFdr,
    "fwe": SelectFwe,
}


def corr(X, y):
    try:
        return np.apply_along_axis(lambda k: spearmanr(k, y).correlation, 0, X)
    except np.AxisError:
        with open("test.log", "w") as w:
            w.write(str(X.shape) + str(y.shape))
            np.save("xmat", X, allow_pickle=False)
            np.save("ymat", y, allow_pickle=False)


_methods = {
    "f_regression": f_regression,
    "spearmanr": corr,
    "chi2": chi2,
    "f_classif": f_classif,
}


def get_selector_by_key(method, mode, param):
    selector = _selection_modes[mode](
        _methods[method] if not isinstance(method, types.FunctionType) else method
    )
    possible_params = selector._get_param_names()
    possible_params.remove("score_func")
    selector.set_params(**{possible_params[0]: param})
    return selector


def truncated_kbest(nfeatures, num_best):
    return [min(x, nfeatures) for x in num_best]


def fselection_classification(nfeatures):
    kbest = [
        x
        for x in product(["chi2", "f_classif"], ["k_best"], truncated_kbest(nfeatures))
    ]
    percentilebest = [
        x for x in product(["chi2", "f_classif"], ["percentile"], [0.1, 0.2, 0.5])
    ]
    return kbest + percentilebest


def fselection(confitem, nfeatures, is_classification=False):
    if is_classification:
        return fselection_classification(nfeatures)
    else:
        methods = []
        for mode in confitem["mode"]:
            param = confitem["param"][mode]
            if mode == "manual":
                manual_methods = [
                    x
                    for x in product(
                        ["manual"], [None], confitem["param"]["manual"].values()
                    )
                ]
                methods.extend(manual_methods)
            if mode == "pca":
                pca_methods = [
                    x for x in product(["pca"], ["n_components"], param["arg"])
                ]
                methods.extend(pca_methods)
            elif mode == "kbest":
                arg = param["arg"]
                metric = param["metric"]
                kbest_methods = [
                    x
                    for x in product(
                        metric,
                        ["k_best"],
                        truncated_kbest(nfeatures, arg),
                    )
                ]
                methods.extend(kbest_methods)
            elif mode == "percentile":
                arg = param["arg"]
                metric = param["metric"]
                percentile_method = [x for x in product(metric, ["percentile"], arg)]
                methods.extend(percentile_method)
            else:
                pass
    return methods


class ManualFeatureSelector:
    def __init__(self, features):
        self._feats = features

    def fit(self, train_df, test_df):
        pass

    def transform(self, df):
        return df[:, self._feats]

    def fit_transform(self, train_df, test_df):
        return self.transform(train_df)


def verify_and_get_feature_selection_stats(
    x_train, selection, selected_features, support_indices, inds
):
    """Return a dict containing remaining features, dropped features, selected features and their scores"""
    ncols = len(selected_features)
    dropped_indices = np.setdiff1d(np.arange(ncols), inds)
    remaining_features = selected_features[list(inds)].tolist()
    remaining_features_scores = selection.scores_[support_indices[list(inds)]]
    dropped_features = selected_features[list(dropped_indices)].tolist()
    dropped_features_scores = selection.scores_[
        support_indices[list(dropped_indices)].tolist()
    ]
    assert ncols == len(remaining_features) + len(dropped_features)
    assert selection.get_support().sum() == len(remaining_features)

    def expand_zip_object(z):
        return [x for x in z]

    dct = {
        "remaining_features": expand_zip_object(
            zip(remaining_features, remaining_features_scores)
        ),
        "dropped_features": expand_zip_object(
            zip(dropped_features, dropped_features_scores)
        ),
        "selected_features": expand_zip_object(
            zip(selected_features, selection.scores_[support_indices])
        ),
    }
    return dct


def run_feature_selection(data, scaler_name: str, selector_name):
    # run univariate feature selection with all columns
    (
        x_train,
        feature_names,
        selected_features,
        selection,
        scaler,
    ) = _run_feature_selection(data, scaler_name, selector_name)
    # truncate imputed values to zero
    # these values need to set to zero because the data is serialized and then loaded again
    x_train = x_train * (~np.isclose(x_train, 0))
    roundoff_zeros(x_train, inplace=True)
    # perform reduced row echelon form of the matrix
    # inds is a tuple containing indices of columns that are linear combinations of other columns
    _, ind_col_indices = sympy.Matrix(x_train).rref()
    # Given the set of indices of independent columns (inds), update the feature selection object
    # to drop non-independent columns. The method for dropping depends on the type of feature selection
    # method either manual (if the selection object has an attribute named _feats) or
    # k-best (if the selection object has an attribute named scores_).
    ncols = x_train.shape[1]
    dep_col_indices = np.setdiff1d(np.arange(ncols), ind_col_indices)
    if hasattr(selection, "_feats"):  # Manual feature selection
        # create a dict with selected, dropped and remaining features
        fs_dct = {
            "remaining_features": selected_features[list(ind_col_indices)].tolist(),
            "dropped_features": selected_features[list(dep_col_indices)].tolist(),
            "selected_features": selected_features.tolist(),
        }
        assert len(fs_dct["selected_features"]) == len(
            fs_dct["remaining_features"]
        ) + len(fs_dct["dropped_features"])
        assert len(fs_dct["remaining_features"]) >= 1
        selection._feats = selection._feats[list(ind_col_indices)]
    elif hasattr(selection, "scores_"):  # kbest feature selection
        # To modify selection object, set scores of non-independent columns to min(score) -1
        # and reduce k by number of dependent columns
        # this ensures that dependent columns are not selected
        # NOTE: Selection object doesn't maintain an array of indices of selected columns. Instead it stores
        # the scores of all columns.
        support_indices = selection.get_support(True)
        selection.scores_[support_indices[dep_col_indices]] = min(selection.scores_) - 1
        selection.k = selection.k - len(dep_col_indices)
        assert selection.k >= 1
        fs_dct = verify_and_get_feature_selection_stats(
            x_train, selection, selected_features, support_indices, ind_col_indices
        )
    fs_dct["num_features"] = (
        ncols,
        len(ind_col_indices),
        len(fs_dct["dropped_features"]),
    )
    # variance of all selected features
    fs_dct["variance"] = pd.Series(x_train.var(axis=0)).describe().to_dict()
    return (
        x_train[:, ind_col_indices],
        feature_names,
        selected_features[list(ind_col_indices)],
        fs_dct,
        selection,
        scaler,
    )


def _run_feature_selection(data, scaler_name: str, selector_name):
    """Selects features using a scoring function

    Given the names of data scaling and feature selection methods, the function
    returns the best set of features. The names uniquely identify the parameters
    for the feature selection methods. See `featureselection` module for more details.
    Args:
        data_filename_memmap: `str` memorymap file for data
        scaler_name: `str` name for data scaling method
        selector_name: `str` name for feature selection method
    """
    scaler = get_scaler_by_key(scaler_name)
    method, mode, param = selector_name
    feature_list = data.xtrain.columns

    if method == "manual":
        reg_exp = "|".join(param)
        features_in_data = data.xtrain.columns[
            data.xtrain.columns.str.contains(reg_exp)
        ]
        col_ids = data.xtrain.columns.get_indexer(features_in_data)
        # col_id is -1 for missing columns
        missing_feats = [k for k, v in zip(param, col_ids) if v < 0]
        if len(missing_feats) > 0:
            logger.debug(
                f"Missing columns during manual feature selection {missing_feats}"
            )
        if len(missing_feats) == len(param):
            # all features are missing
            raise FeatureSelectionException(f"Features {param} not found in data")

        mfs = ManualFeatureSelector(col_ids)
        X_transform = mfs.fit_transform(
            scaler.fit_transform(data.xtrain, data.ytrain), data.ytrain
        )
        return X_transform, feature_list, features_in_data, mfs, scaler
    elif method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=param)
        X_transform = pca.fit_transform(
            scaler.fit_transform(data.xtrain, data.ytrain), data.ytrain
        )
        return (
            X_transform,
            feature_list,
            tuple(pca.singular_values_.tolist()),
            pca,
            scaler,
        )
    else:
        feature_selector = get_selector_by_key(method, mode, param)
        X_scaled = scaler.fit_transform(data.xtrain, data.ytrain)
        feature_selector.fit(X_scaled, data.ytrain)
        selected_features = feature_list[feature_selector.get_support()]
        return (
            feature_selector.transform(X_scaled),
            feature_list,
            selected_features,
            feature_selector,
            scaler,
        )
