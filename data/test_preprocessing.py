import pathlib

import metadata
import numpy as np
import pandas as pd
from utils import (
    encoding_get_first,
    encoding_get_largest,
    indicator_standard_missingness
)

from preprocessing import (
    get_features_with_too_few_nonmodal_observations
)

from .load import get_all_missing, get_lowvariance, get_recode_cols, read

def test_preprocessing():
    id_col = "ChallengeID"
    x_path = pathlib.Path(
        "./data/datasets/testdata/preprocessing_outcomes/features.csv"
    )
    y_path = pathlib.Path(
        "./data/datasets/testdata/preprocessing_outcomes/outcomes.csv"
    )
    meta_path = pathlib.Path(
        "./data/datasets/testdata/preprocessing_outcomes/metadata.csv"
    )
    metadata.load(meta_path)
    meta = metadata.get_snapshot()
    ((X, _), (y, _)) = (read(x_path), read(y_path))
    # set index for X and y
    X.set_index(id_col, inplace=True)
    y.set_index(id_col, inplace=True)
    cols = get_recode_cols(y, meta)
    recode_col = "dep_a"
    assert len(cols) == 1 and recode_col in cols
    y.mask(y[cols] == 2, other=0, inplace=True)
    assert not (y[recode_col] == 2).any()
    not_recode_col = "dep_c"
    assert (y[not_recode_col] == 2).any()


def test_drop_lowvariance():
    """
    var_1: 1, 1, 1, 1, 1 [drop because 0 variance among non-missing values]
    var_2: 0, 0, 0, 0, 0 [drop because 0 variance among non-missing values]
    var_3: 1, -6, -6, -6, -6 [drop because 0 variance among non-missing values, note that variance of a single number is undefined]
    var_4: 1, 1, 1, -6, -9 [drop because 0 variance among non-missing values]
    var_5: -6, -6, -6, -6, -6 [drop because all missing]
    var_6: -6, -6, -6, -6, -9 [drop because all missing]
    var_7: 1, 0, -6, -6, -6 [keep]
    var_8: -6, -9, 1, 0, -6 [keep]
    """
    df = pd.DataFrame(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, -6, -6, -6, -6],
            [1, 1, 1, -6, -9],
            [-6, -6, -6, -6, -6],
            [-6, -6, -6, -6, -9],
            [1, 0, -6, -6, -6],
            [-6, -9, 1, 0, -6],
        ],
    ).T
    df.columns = [
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_7",
        "var_8",
    ]
    df = df.applymap(lambda x: np.nan if indicator_standard_missingness(x) else x) 
    all_missing = get_all_missing(df)
    low_var = get_lowvariance(df, 0)
    assert all_missing.symmetric_difference(["var_5", "var_6"]).empty
    assert low_var.symmetric_difference(
        ["var_1", "var_2", "var_3", "var_4", "var_5", "var_6"]
    ).empty


def test_drop_features_with_few_nonmodal_observations():
    """
    var_1: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 [drop because there are zero non-modal values]
    var_2: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2 [drop because fewer than 10 non-modal values]
    var_3: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3 [drop because fewer than 10 non-modal values (even though some of those non-modal values differ from each other)]
    var_4: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, -6, -6, -6, -6, -6, -6, -9, -9, -9, -9, -9 [drop because fewer than 10 non-modal values among non-missing values]
    var_5: .9999999999, .9999999999, .9999999999, .9999999999, .9999999999, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0000000001, 1.0000000001, 1.0000000001, 1.0000000001, 1.0000000001, 1.0000000001 [drop because these values are so close together that they might differ due only to a floating point issue, and therefore they should be considered the same value for our purposes]
    var_6: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 [keep because there are exactly 10 non-modal values]
    var_7: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 [keep because there are greater than 10 non-modal values]
    var_8: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, -9 [keep because there are greater than 10 non-modal values (checking that this still works when a negative value is present)]
    var_9: 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3 [keep because there are greater than 10 non-modal values, when two values are tied for the mode]
    var_10: 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, -9, -9, -9, -9, -9, -9, -9 [drop because there are fewer than 10 non-modal values, when two values are tied for the mode]  
    """
    df = pd.DataFrame(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, -6, -6, -6, -6, -6, -6, -9, -9, -9, -9, -9],
            [.9999999999, .9999999999, .9999999999, .9999999999, .9999999999, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0000000001, 1.0000000001, 1.0000000001, 1.0000000001, 1.0000000001, 1.0000000001],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, -9],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, -9, -9, -9, -9, -9, -9, -9],
        ],
    ).T
    df.columns = [
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_7",
        "var_8",
        "var_9",
        "var_10",
    ]
    df = df.applymap(lambda x: np.nan if indicator_standard_missingness(x) else x) # This line assumes we will mark missing values NA before running the function to find rare features
    features_below_threshold_for_nonmodal_observations = get_features_with_too_few_nonmodal_observations(df)
    assert features_below_threshold_for_nonmodal_observations.symmetric_difference(
        ["var_1", "var_2", "var_3", "var_4", "var_5", "var_10"]
    ).empty

    
def test_drop_encoding_levels():
    df = pd.DataFrame(
        [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
        columns=["var_1.0", "var_2.0", "var_nan"],
    )
    assert all(encoding_get_largest(df) == "var_2.0")
    assert all(encoding_get_first(df) == "var_1.0")
