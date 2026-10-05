# flake8: noqa
#
#!/usr/bin/env python3

# reads features and outcomes files and prepares X and Y matrices for running piplines

import pathlib
from collections import namedtuple

import metadata
import numpy as np
import pandas as pd
from scales import get_scale_data
from sklearn.impute import SimpleImputer
from utils import (
    conf_get,
    drop_and_track,
    encoding_get_largest,
    get_concept_json,
    get_logger,
    get_outcomes,
    get_warning_cols,
    indicator_standard_missingness,
    read,
)
from utils import track_shapes as Tracker

logger = get_logger(__name__)
DataSplit = namedtuple("DataSplit", ["xtrain", "xtest", "ytrain", "ytest", "meta"])


def get_recode_cols(y, meta):
    return meta.query(
        'type == "Binary" and warning != "A yes/no variable that has “no” coded to 0 instead of 2"'
    ).index.intersection(y.columns)


def recode_cols(y, meta):
    cols = get_recode_cols(y, meta)
    y.mask(y[cols] == 2, other=0, inplace=True)


def encode_unordered_categorical(df, meta, conf):
    encoding_method = conf_get(
        conf, "data.preprocessing.predictors.encoding_method_unordered_categorical"
    )
    logger.info("Encoding unordered categorical variables")
    cols = meta.loc[df.columns].query('type == "Unordered Categorical"').index
    logger.info(f"Number of unord. cat. cols = {len(cols)} out of {len(df.columns)}")
    if len(cols) == 0:
        return df
    if conf_get(
        conf,
        "data.preprocessing.predictors.impute_mode_before_encoding_unordered_categorical",
    ):
        imputer = SimpleImputer(strategy="most_frequent")
        df[cols] = imputer.fit_transform(df[cols])
    # encoding_method in the job file can be one of one-hot or dummy_largest.
    # 1. If one-hot then create k levels for an unordered categorical with k different values
    # 2. If dummy_largest then the largest level is dropped
    # By default, pd.get_dummies implements one-hot encoding
    new_df = pd.get_dummies(
        df[cols],
        prefix=cols,
        dummy_na=True,
        columns=cols,
    )
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PREPROCESSING_ENCODING,
            Tracker.MatrixType.PREDICTORS_DUMMIES,
            df[cols].shape,
            "Unordered categorical variables for encoding",
        )
    )
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PREPROCESSING_ENCODING,
            Tracker.MatrixType.PREDICTORS_DUMMIES,
            new_df.shape,
            f"Dummies created for {len(cols)} unordered categorial variables",
        )
    )
    dummy_cols_to_drop = []
    if encoding_method == "one-hot":
        pass
    elif encoding_method == "dummy_largest":
        dummy_cols_to_drop = encoding_get_largest(new_df)
    new_df = drop_and_track(
        new_df,
        dummy_cols_to_drop,
        Tracker.ModuleType.PREPROCESSING_ENCODING,
        Tracker.MatrixType.PREDICTORS_DUMMIES,
        "dummies",
        "Drop levels of unordered categoricals based on encoding_method = {opt['encoding_method']}",
    )
    # When dummy_na=True in pd.get_dummies, the encoded data contains _nan columns (with all zeros) even when the original
    # unordered categorical variable has no missing values. Identify such columns and drop them!
    new_df_na_cols = (
        new_df.iloc[:, new_df.columns.str.endswith("_nan")]
        .apply(lambda x: all(x == 0))
        .where(lambda x: x)
        .dropna()
        .index
    )
    new_df = drop_and_track(
        new_df,
        new_df_na_cols,
        Tracker.ModuleType.PREPROCESSING_ENCODING,
        Tracker.MatrixType.PREDICTORS_DUMMIES,
        "encoding_nan_levels",
        "Drop nan levels that are all zeros",
    )
    # Drop low_variance predictors after encoding
    low_var_threshold = conf_get(
        conf, "data.preprocessing.predictors.remove_predictors_below_variance_threshold"
    )
    low_var_cols = get_lowvariance(new_df, low_var_threshold)
    new_df = drop_and_track(
        new_df,
        low_var_cols,
        Tracker.ModuleType.PREPROCESSING_ENCODING,
        Tracker.MatrixType.PREDICTORS_DUMMIES,
        "lowvariance",
        f"Removing low variance(={low_var_threshold}) levels of unordered categorical vars",
    )
    # add metadata information for new columns
    meta_rows = []
    for colname in new_df.columns:
        prefix = colname.rsplit("_", 1)[0]
        mrow = meta.loc[prefix]
        meta_rows.append(
            (
                colname,
                mrow.type,
                mrow.topics,
                mrow.subtopics,
                mrow.warning,
                mrow.n_cities_asked,
            )
        )
    meta_df = pd.DataFrame(
        meta_rows,
        columns=[
            "new_name",
            "type",
            "topics",
            "subtopics",
            "warning",
            "n_cities_asked",
        ],
    )
    metadata.append(meta_df)
    df = df.join(new_df)
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PREPROCESSING_ENCODING,
            Tracker.MatrixType.PREDICTORS,
            df.shape,
            "Appending dummies to predictor matrix",
        )
    )
    df = drop_and_track(
        df,
        cols,
        Tracker.ModuleType.PREPROCESSING_ENCODING,
        Tracker.MatrixType.PREDICTORS,
        "unord_cat",
        "Removing unordered categoricals after encoding",
    )
    return df


def filter_participants(X, y, basepath, conf, id_col):
    # filter on training participant ids
    pids = []
    if (
        conf_get(conf, "data.files.training_participant_ids") != "all"
        and len(conf_get(conf, "data.files.training_participant_ids")) > 0
    ):
        pid, _ = read(
            basepath.joinpath(conf_get(conf, "data.files.training_participant_ids")[0]),
            read_as_str=True,
        )
        pids.append([_ for _ in pid[id_col]])
    if not pids:
        return X, y
    common_pids = pd.Index(
        set.intersection(*[set(pidlist) for pidlist in pids])
    ).sort_values()
    X = X.loc[common_pids]
    y = y.loc[common_pids]
    assert (
        X.shape[0] == y.shape[0]
    ), "Number of rows in the feature matrix is different from the number of rows in outcome matrix"
    return X, y


def filter_Columbia_participants(X, y, basepath, conf, id_col):
    pids = []
    if (
        conf_get(
            conf,
            "data.participants.remove_participants_collected_in_nonresponse_followup_Columbia",
        )
        and "cp6source" in y
    ):
        pid = (y.cp6source != 2).where(lambda x: x).dropna().index
        logger.info(
            f"Removing participants in Columbia followup. Number of participants = {len(y.index.difference(pid))}"
        )
        pids.append(pid.tolist())
    if not pids:
        return X, y
    common_pids = set.intersection(*[set(pidlist) for pidlist in pids])
    X = X.loc[common_pids]
    y = y.loc[common_pids]
    assert (
        X.shape[0] == y.shape[0]
    ), "Number of rows in the feature matrix is different from the number of rows in outcome matrix"
    return X, y


def get_xy(basepath, conf, id_col):
    features_file, outcomes_file = (
        basepath.joinpath(conf_get(conf, "data.files.predictors")[0]),
        basepath.joinpath(conf_get(conf, "data.files.outcomes")[0]),
    )
    logger.info(
        f"Reading predictors file {features_file} and outcomes file {outcomes_file} "
    )
    ((X, dta_meta_x), (y, _)) = (
        read(features_file),
        read(outcomes_file),
    )
    # set index for X and y
    X.set_index(id_col, inplace=True)
    y.set_index(id_col, inplace=True)
    return X, y, dta_meta_x


def scale_computation(conf, df, stata_meta):
    if conf_get(conf, "data.preprocessing.outcomes.add_scales_to_outcomes") or conf_get(
        conf, "data.preprocessing.predictors.add_scales_to_predictors"
    ):
        path = pathlib.Path(__file__).parent.absolute()
        scales_instructions = path.joinpath(
            conf_get(conf, "data.files.scales_instructions")[0]
        )
        return get_scale_data(scales_instructions, df, stata_meta)
    return None, None, None


def append_scales(X, y, conf, dta_meta_x):
    if conf_get(conf, "data.preprocessing.outcomes.add_scales_to_outcomes") or conf_get(
        conf, "data.preprocessing.predictors.add_scales_to_predictors"
    ):
        # cols is the list of scales
        # intermediatecols is the list of variables used in constructing scales
        scale_data, scale_meta, cols, intermediate_cols = scale_computation(
            conf, X, dta_meta_x
        )
        if scale_meta is not None:
            metadata.append(scale_meta)
            # y is unappended during scale construction. Add cols.
            y = pd.merge(y, scale_data[cols], left_index=True, right_index=True)
            # X contains both cols and intermediate_cols. Drop intermediate_cols.
            X.drop(intermediate_cols, axis=1, inplace=True)
    return X, y


def get_weight_columns(df, meta):
    """Returns column names for which the subtopics field contains survey weights (case insensitive)"""
    return (
        meta.subtopics.str.contains("survey weights", case=False)
        .where(lambda x: x)
        .dropna()
        .index.intersection(df.columns)
        if "topics" in meta and "subtopics" in meta
        else pd.Index([])
    )


def get_paradata_columns(df, meta):
    """Returns column names for which the subtopics field contains paradata (case insensitive)"""
    return (
        meta.subtopics.str.contains("paradata", case=False)
        .where(lambda x: x)
        .dropna()
        .index.intersection(df.columns)
        if "topics" in meta and "subtopics" in meta
        else pd.Index([])
    )


def filter_outcomes(y, conf, meta):
    waves_to_use = conf_get(conf, "data.preprocessing.outcomes.waves_to_use")
    vars_not_in_waves = meta.loc[~meta.wave.isin(waves_to_use)].index
    y = drop_and_track(
        y,
        vars_not_in_waves,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.OUTCOMES,
        "not_in_wave",
        f"Removing outcomes that are not in {waves_to_use}",
    )
    outcomes_not_in_job_file = y.columns.difference(get_outcomes(y, meta, conf))
    y = drop_and_track(
        y,
        outcomes_not_in_job_file,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.OUTCOMES,
        "outcomes",
        "Removing outcomes not specified in job file",
    )
    # remove outcomes of type
    outcome_vartype = conf_get(conf, "data.preprocessing.outcomes.vartypes_to_use")
    rm_outcomes = y.columns.difference(meta[meta.type.isin(outcome_vartype)].index)
    y = drop_and_track(
        y,
        rm_outcomes,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.OUTCOMES,
        "vartype",
        f"Removing outcomes NOT of vartype {outcome_vartype}",
    )
    min_cities = conf_get(conf, "data.preprocessing.outcomes.min_cities_outcomes")
    vars_fewer_than_min_cities = meta[meta.n_cities_asked < min_cities].index
    y = drop_and_track(
        y,
        vars_fewer_than_min_cities,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.OUTCOMES,
        "not_in_min_cities",
        f"Removing outcomes with min cities < {min_cities}",
    )
    if conf_get(conf, "data.preprocessing.outcomes.recode2to0"):
        recode_cols(y, meta)
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PREPROCESSING,
            Tracker.MatrixType.OUTCOMES,
            y.shape,
            "Recode2to0",
        )
    )
    if conf_get(conf, "data.preprocessing.outcomes.remove_skipped_outcomes"):
        skipped_outcomes = (y == -6).any(axis=0).where(lambda x: x).dropna().index
        y = drop_and_track(
            y,
            skipped_outcomes,
            Tracker.ModuleType.PREPROCESSING,
            Tracker.MatrixType.OUTCOMES,
            "skipped",
            "Removing skipped outcomes",
        )
    if (
        conf_get(conf, "data.preprocessing.outcomes.remove_months_from_outcomes")
        is not False
    ):
        y = drop_and_track(
            y,
            metadata.get_month_vars(),
            Tracker.ModuleType.PREPROCESSING,
            Tracker.MatrixType.OUTCOMES,
            "months",
            "Removing month outcomes",
        )
    # remove weights from outcomes
    if conf_get(conf, "data.preprocessing.outcomes.remove_weights_from_outcomes"):
        y = drop_and_track(
            y,
            get_weight_columns(y, meta),
            Tracker.ModuleType.PREPROCESSING,
            Tracker.MatrixType.OUTCOMES,
            "weights",
            "Removing weights from outcomes",
        )
    # remove paradata from outcomes
    if conf_get(conf, "data.preprocessing.outcomes.remove_paradata_from_outcomes"):
        y = drop_and_track(
            y,
            get_paradata_columns(y, meta),
            Tracker.ModuleType.PREPROCESSING,
            Tracker.MatrixType.OUTCOMES,
            "paradata",
            "Removing paradata from outcomes",
        )
    warning_opt = conf_get(
        conf, "data.preprocessing.outcomes.remove_outcomes_with_these_warnings"
    )
    warning_cols = get_warning_cols(meta, y.columns, warning_opt)
    y = drop_and_track(
        y,
        warning_cols,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.OUTCOMES,
        "warning",
        f"Removing outcomes with warning codes {warning_opt}",
    )
    concept_opt = conf_get(conf, "data.files.million_monkeys_concepts_metadata")
    concept_json = get_concept_json(concept_opt)
    if concept_json:
        rm_outcomes = conf_get(
            conf, "data.preprocessing.outcomes.remove_outcomes_of_concept"
        )
        for concept in rm_outcomes:
            # get concept dict from json
            for dct in concept_json:
                if dct["concept"] == concept:
                    y = drop_and_track(
                        y,
                        dct["varnames"],
                        Tracker.ModuleType.PREPROCESSING,
                        Tracker.MatrixType.OUTCOMES,
                        f"{concept}",
                        f"Removing outcomes of concept {concept}",
                    )
    # If the outcome does not contain "NA" for any observations in the input file
    # then convert negative outcome values to NA and drop homogeneous outcomes
    outcomes_without_na = (~y.isna().any()).where(lambda x: x).dropna().index
    y[outcomes_without_na] = y[outcomes_without_na].applymap(
        lambda x: np.nan if indicator_standard_missingness(x) else x
    )
    zero_variance_outcomes = get_lowvariance(y, 0)
    y = drop_and_track(
        y,
        zero_variance_outcomes,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.OUTCOMES,
        "zero_variance",
        "Removing outcomes where the non-missing values are all same (zero variance)",
    )
    # Finally, drop outcomes that don't have minimum number of non-negative values
    min_samples = conf_get(conf, "data.preprocessing.outcomes.min_valid_outcomes")
    less_than_min_sample_outcomes = (
        ((y >= 0).sum() < min_samples).where(lambda x: x).dropna().index
    )  # >=0 returns false for NaNs
    y = drop_and_track(
        y,
        less_than_min_sample_outcomes,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.OUTCOMES,
        "min_samples",
        f"Removing outcomes where the number of non-missing values is less than min_valid_outcomes(={min_samples}) option",
    )
    return y


def get_lowvariance(df, threshold):
    """Return variables for which the variance among non-missing is less than or equal to threshold"""
    # Unbiased var estimator = \Sum_ (x_i - xhat) ** 2 / (N-1)
    cols = ((df.var() <= threshold) | df.var().isna()).where(lambda x: x).dropna().index
    return cols


def get_all_missing(df):
    """Return variables that are all missing"""
    all_missing_vars = df.isna().all().where(lambda x: x).dropna().index
    return all_missing_vars


def filter_predictors(X, conf, meta):
    waves_to_use = conf_get(conf, "data.preprocessing.predictors.waves_to_use")
    vars_not_in_waves = meta.loc[~meta.wave.isin(waves_to_use)].index
    X = drop_and_track(
        X,
        vars_not_in_waves,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.PREDICTORS,
        "not_in_wave",
        f"Removing predictors that are not in {waves_to_use}",
    )
    vartypes = conf_get(conf, "data.preprocessing.predictors.vartypes_to_use")
    vars_not_of_type = meta.loc[~meta.type.isin(vartypes)].index
    X = drop_and_track(
        X,
        vars_not_of_type,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.PREDICTORS,
        "not_of_vartype",
        f"Removing predictors that are not of vartype {vartypes}",
    )
    min_cities = conf_get(conf, "data.preprocessing.predictors.min_cities_predictors")
    vars_fewer_than_min_cities = meta[meta.n_cities_asked < min_cities].index
    X = drop_and_track(
        X,
        vars_fewer_than_min_cities,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.PREDICTORS,
        "not_in_min_cities",
        f"Removing predictors with min cities < {min_cities}",
    )
    # remove weights from predictors
    if conf_get(conf, "data.preprocessing.predictors.remove_weights_from_predictors"):
        X = drop_and_track(
            X,
            get_weight_columns(X, meta),
            Tracker.ModuleType.PREPROCESSING,
            Tracker.MatrixType.PREDICTORS,
            "weights",
            "Removing weights from predictors",
        )
    # remove paradata from predictors
    if conf_get(conf, "data.preprocessing.predictors.remove_paradata_from_predictors"):
        X = drop_and_track(
            X,
            get_paradata_columns(X, meta),
            Tracker.ModuleType.PREPROCESSING,
            Tracker.MatrixType.PREDICTORS,
            "paradata",
            "Removing paradata from predictors",
        )
    # remove metadata warnings from predictors
    warning_opt = conf_get(
        conf, "data.preprocessing.predictors.remove_predictors_with_these_warnings"
    )
    warning_cols = get_warning_cols(meta, X.columns, warning_opt)
    X = drop_and_track(
        X,
        warning_cols,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.PREDICTORS,
        "warning",
        f"Removing predictors with warning codes {warning_opt}",
    )
    # remove heterogeneous units
    concept_opt = conf_get(conf, "data.files.million_monkeys_concepts_metadata")
    concept_json = get_concept_json(concept_opt)
    if concept_json:
        # remove predictors
        rm_predictors = conf_get(
            conf, "data.preprocessing.predictors.remove_predictors_of_concept"
        )
        for concept in rm_predictors:
            # get concept dict from json
            for dct in concept_json:
                if dct["concept"] == concept:
                    X = drop_and_track(
                        X,
                        dct["varnames"],
                        Tracker.ModuleType.PREPROCESSING,
                        Tracker.MatrixType.PREDICTORS,
                        f"{concept}",
                        f"Removing predictors of concept {concept}",
                    )
    # If the predictors does not contain "NA" for any observations in the input file
    # then convert negative predictor values to NA and drop homogeneous outcomes
    predictors_without_na = (~X.isna().any()).where(lambda x: x).dropna().index
    X[predictors_without_na] = X[predictors_without_na].applymap(
        lambda x: np.nan if indicator_standard_missingness(x) else x
    )
    # remove all missing predictors
    all_missing_cols = get_all_missing(X)
    X = drop_and_track(
        X,
        all_missing_cols,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.PREDICTORS,
        "allmissing",
        "Removing predictors with all missing",
    )
    # remove low variance
    low_var_threshold = conf_get(
        conf, "data.preprocessing.predictors.remove_predictors_below_variance_threshold"
    )
    low_var_cols = get_lowvariance(X, low_var_threshold)
    X = drop_and_track(
        X,
        low_var_cols,
        Tracker.ModuleType.PREPROCESSING,
        Tracker.MatrixType.PREDICTORS,
        "lowvariance",
        f"Removing predictors with low variance(={low_var_threshold})",
    )
    return X


def track_n(matrices, msg_prefix):
    """Supports tracking similar preprocessing steps of X, y or X, y, metadata"""
    mtypes = [
        Tracker.MatrixType.PREDICTORS,
        Tracker.MatrixType.OUTCOMES,
        Tracker.MatrixType.METADATA,
    ]
    for index, item in enumerate(matrices):
        if item is None:
            continue
        Tracker.add(
            Tracker.TrackInfo(
                Tracker.ModuleType.PREPROCESSING,
                mtypes[index],
                item.shape,
                f"{msg_prefix} {mtypes[index]}",
            )
        )


def get_data(conf):
    """Returns X, Y and Metadata matrices
    :param conf: jobfile in json format
    :return: X, Y and meta matrices
    """
    # 1. Read raw data
    basepath = pathlib.Path(__file__).parent.absolute()
    id_col = conf_get(conf, "data.participants.participant_id_column")
    X, y, dta_meta_x = get_xy(basepath, conf, id_col)
    track_n([X, y], "Loading")
    # 2. Filter participants
    X, y = filter_participants(X, y, basepath, conf, id_col)
    track_n([X, y], "Filter training participants in")
    # 2.1 Filter Columbia participants
    X, y = filter_Columbia_participants(X, y, basepath, conf, id_col)
    track_n([X, y], "Filter Columbia participants in")
    # 3.1 read metadata
    metadata.load(basepath.joinpath(conf_get(conf, "data.files.metadata")[0]))
    track_n([None, None, metadata.get_snapshot()], "Loading")
    # 3.2 check for metadata edits
    if conf_get(conf, "data.files.metadata_edit_instructions") is not None:
        logger.info("Reading metadata edit instructions")
        meta_edit_ins, _ = read(
            basepath.joinpath(
                conf_get(conf, "data.files.metadata_edit_instructions")[0]
            )
        )
        meta_edit_ins.columns = meta_edit_ins.columns.str.strip()
        var = "Variable(s)"
        field = "Field"
        val = "New value"
        for _, row in meta_edit_ins.iterrows():
            if row["Monkeys Auto-Fix"]:
                for varname in row[var].split(","):
                    varname = varname.strip()
                    metadata.edit(varname, row[field], row[val])
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PREPROCESSING,
            Tracker.MatrixType.METADATA,
            metadata.get_shape(),
            "Metadata edits",
        )
    )
    # 4. Compute scales
    X, y = append_scales(X, y, conf, dta_meta_x)
    track_n(
        [X, None, None],
        "Computing scales, variables with units standardized, variables with units combined outcomes and",
    )
    track_n(
        [None, y, None],
        "Computing scales, variables with units standardized, variables with units combined predictors and",
    )
    track_n(
        [None, None, metadata.get_snapshot()],
        "Computing scales, variables with units standardized, variables with units combined predictors and outcomes",
    )
    # 5. Get a snapshot of the metadata for next set of operations
    meta = metadata.get_snapshot()
    # 6. Filter outcomes
    y = filter_outcomes(y, conf, meta)
    # 7. Filter predictors
    X = filter_predictors(X, conf, meta)
    # 8. Encoding unordered categoricals
    X = encode_unordered_categorical(X, meta, conf)
    logger.info(
        f"Number of columns after encoding unordered categorical {len(X.columns)}"
    )
    meta = metadata.get_snapshot()
    Tracker.add(
        Tracker.TrackInfo(
            Tracker.ModuleType.PREPROCESSING,
            Tracker.MatrixType.METADATA,
            metadata.get_shape(),
            "Encoding unordered categoricals",
        )
    )
    return X, y, meta
