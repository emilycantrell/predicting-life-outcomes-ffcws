import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.core.common import SettingWithCopyWarning
from utils import conf_get, indicator_standard_missingness
from utils import track_shapes as Tracker

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# Note: This function assumes that negative outcome values represent missing data and therefore
# removes all observations with outcome values below zero. Although this convention generally
# holds in the FFCWS data, some variables (e.g., z-scores) can have valid, non-missing negative
# values. The function is therefore not appropriate for such outcomes. For this reason, we
# exclude z-score outcomes from our analyses, as described in the appendix.
def remove_missing(X, y, outcome):
    if outcome not in y:
        raise KeyError("Outcome {} not in the data".format(outcome))
    y_subset = y[[outcome]]
    # samples for which outcome is not null and greater than 0
    y_non = y_subset[(y_subset[outcome].notnull()) & (y_subset[outcome] >= 0)]
    df = pd.merge(
        X,
        y_non.dropna(),
        left_index=True,
        right_index=True,
    )
    X1, y1 = df[X.columns], df[outcome]
    X1.sort_index(inplace=True)
    y1.sort_index(inplace=True)
    return X1, y1


def mark_na_missing(X, inplace=True):
    X.mask((X <= -1) & (X >= -9), np.nan, inplace=inplace)


def cols_with_indicator(data, cols):
    # get columns that have atleast one na and one notna
    #
    nan_values = data.xtrain.isna()
    nan_columns = nan_values.any()
    all_nan_columns = nan_values.all()
    cols_ind = (
        data.xtrain.columns[nan_columns]
        .difference(data.xtrain.columns[all_nan_columns])
        .intersection(cols)
    )
    if len(cols_ind) > 0:
        train_mask_data = data.xtrain[cols_ind].isna().astype(int).add_prefix("mask_")
        test_mask_data = data.xtest[cols_ind].isna().astype(int).add_prefix("mask_")
        return train_mask_data, test_mask_data
    else:
        return None, None


def inplace_mean_impute(data, cols):
    train_mask_data, test_mask_data = cols_with_indicator(data, cols)
    data.xtrain.fillna(dict(data.xtrain[cols].mean()), inplace=True)
    data.xtest.fillna(dict(data.xtrain[cols].mean()), inplace=True)
    return train_mask_data, test_mask_data


def inplace_median_impute(data, cols):
    train_mask_data, test_mask_data = cols_with_indicator(data, cols)
    data.xtrain.fillna(dict(data.xtrain[cols].median()), inplace=True)
    data.xtest.fillna(dict(data.xtrain[cols].median()), inplace=True)
    return train_mask_data, test_mask_data


def inplace_mode_impute(data, cols):
    train_mask_data, test_mask_data = cols_with_indicator(data, cols)
    data.xtrain.fillna(dict(data.xtrain[cols].mode().iloc[0]), inplace=True)
    data.xtest.fillna(dict(data.xtrain[cols].mode().iloc[0]), inplace=True)
    return train_mask_data, test_mask_data


def inplace_dropna_columns(X, y):
    X.dropna(axis=1, how="all", inplace=True)


def inplace_drop_notcategorical(X, y, nunique=25):
    # remove columns for which number nunique greater than threshold
    ax = X.nunique() > nunique
    X.drop(ax[ax].index, axis=1, inplace=True)


_methods = [
    ("mean", inplace_mean_impute),
    ("median", inplace_median_impute),
    ("mode", inplace_mode_impute),
    ("dropna columns", inplace_dropna_columns),
    ("drop noncategorical", inplace_drop_notcategorical),
]


def mask_runner(data, col_types, vartype, method):
    method_dict = dict(_methods)
    cols = [
        k for k, v in col_types.items() if v == vartype and k in data.xtrain.columns
    ]
    if not cols:
        return None
    return method_dict[method](data, cols)


def impute(data, meta, methods, add_mask_cols=True):
    col_types = dict(zip(meta.index, meta.type))
    mask_dfs = {}
    impute_mask_tasks = [
        (data, col_types, vartype, method) for vartype, method in methods.items()
    ]
    with Parallel(n_jobs=1, verbose=10) as parallel:
        results = parallel(delayed(mask_runner)(*args) for args in impute_mask_tasks)

    for result, (_, _, vartype, method) in zip(results, impute_mask_tasks):
        if result is None:
            continue
        train_mask_data, test_mask_data = result
        if train_mask_data is not None and test_mask_data is not None:
            mask_dfs[method] = (train_mask_data, test_mask_data)

    if add_mask_cols:
        for train_mask_df, test_mask_df in mask_dfs.values():
            data.xtrain[train_mask_df.columns] = train_mask_df
            data.xtest[test_mask_df.columns] = test_mask_df

# This function takes a column and checks if it has too few nonmodal 
# observations            
def check_nonmodal_observations(x, nonmodal_threshold = 10, floating_point_threshold = .000001):
    # The following code makes sure that we are treating values within 10^-6 
    # of each other as the same. We find the longest chains of mergable unique  
    # values and count the number of observations with these values.
    # Tabulate the counts of each unique value. Sort the table by the unique 
    # values--not the counts--in ascending order. The unique values are the 
    # indices.
    value_counts = x.value_counts().sort_index()
    # For each row until the second to last row...
    i = 0
    while i < len(value_counts) - 1:
        # Check if the index for the next row--the smallest unique value
        # larger than the one in this row--is within 10^-6 of the index of
        # of this row
        # If yes...
        if value_counts.index[i + 1] - value_counts.index[i] < floating_point_threshold:
            # Add the count of the unique value for this row to the count in
            # the next row
            value_counts.iloc[i + 1] = value_counts.iloc[i] + value_counts.iloc[i + 1]
            # Delete this row to avoid double counting. We delete this row, 
            # not the next row because the index for the NEXT NEXT row may be 
            # about .000002 apart from this row but still grouped together 
            # with this row. If there are three values like 0.9^10-6, 
            # 1.8^10-6, and 2.7^10^-6, we will want to merge the counts.
            value_counts.drop(value_counts.index[i], inplace = True)
            # After the deletion of this row, the next row is the new this 
            # row, and every single following row also move up one. We tell
            # the code to look at this row (the old next row) again to avoid
            # skipping over a row
            i = i - 1
        # Move on to the next row
        i = i + 1
    # Sort the table by the counts--not the values--of unique values in
    # ascending order. The last row is the mode. Add up the counts in all
    # other rows to check if the sum is less than the nonmodal value threshold
    return value_counts.sort_values().iloc[:-1].sum() < nonmodal_threshold

# This function returns a list of columns with too few nonmodal observations
def get_features_with_too_few_nonmodal_observations(xtrain):
    return xtrain.columns[xtrain.apply(check_nonmodal_observations)]


def drop_cols(X, cols):
    X.drop(cols, axis=1, inplace=True)


def preprocess(data, meta, conf):
    imputation_methods = conf_get(conf, "pipeline.imputation")
    # replace all negative with missing
    if conf_get(
        conf,
        "data.preprocessing.common.convert_standard_ff_missing_values_to_na",
    ):
        mark_na_missing(data.xtrain, inplace=True)
        mark_na_missing(data.xtest, inplace=True)
        # check that there no standard missing values
        assert not data.xtrain.applymap(indicator_standard_missingness).any().any()
        assert not data.xtest.applymap(indicator_standard_missingness).any().any()
    
    # Get a list of columns with too few nonmodal observations before 
    # imputation
    features_with_too_few_nonmodal_observations = get_features_with_too_few_nonmodal_observations(data.xtrain)
    # Drop columns that had too few nonmodal observations
    drop_cols(data.xtrain, features_with_too_few_nonmodal_observations)
    drop_cols(data.xtest, features_with_too_few_nonmodal_observations)
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PIPELINE,
            Tracker.MatrixType.PREDICTORS_TRAINING,
            data.xtrain.shape,
            "Drop predictors that are below our threshold for number of non-modal observations",
        )
    )
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PIPELINE,
            Tracker.MatrixType.PREDICTORS_TESTING,
            data.xtest.shape,
            "Drop predictors that are below our threshold for number of non-modal observations",
        )
    )
    
    # imputation
    impute(data, meta, imputation_methods)
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PIPELINE,
            Tracker.MatrixType.PREDICTORS_TRAINING,
            data.xtrain.shape,
            "Imputing missing values and adding indicator columns",
        )
    )
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PIPELINE,
            Tracker.MatrixType.PREDICTORS_TESTING,
            data.xtest.shape,
            "Imputing missing values ande adding indicator columns",
        )
    )
    
    # It's possible that for some predictors the training fold has only NAs (Non-missing values are only in the testing fold)
    # Drop such predictors from both training and testing folds.
    # Note : It's impossible to have NAs in the testing fold because of the above imputation step.
    all_na_cols = data.xtrain.columns[data.xtrain.isna().all()]
    drop_cols(data.xtrain, all_na_cols)
    drop_cols(data.xtest, all_na_cols)
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PIPELINE,
            Tracker.MatrixType.PREDICTORS_TRAINING,
            data.xtrain.shape,
            "Drop predictors that are all NAs in the training fold",
        )
    )
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PIPELINE,
            Tracker.MatrixType.PREDICTORS_TESTING,
            data.xtest.shape,
            "Drop predictors that are all NAs in the training fold",
        )
    )
    assert data.xtrain.isna().sum().sum() == 0
