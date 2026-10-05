import math
import numbers
import pathlib
import warnings
from itertools import chain

import metadata
import numpy as np
import pandas as pd
import pytest
from pandas.core.common import SettingWithCopyWarning
from pandas.errors import DtypeWarning
from run.run import myrun, parse_job_toml
from schema import SchemaError
from utils import conf_get, get_concept_json

warnings.filterwarnings(action="error", category=FutureWarning)
warnings.filterwarnings(action="ignore", message=".*", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", message=".*", category=DtypeWarning)
warnings.filterwarnings(action="ignore", message=".*", category=SettingWithCopyWarning)


def check_pipelines(db):
    """Checks if each cv fold has same number of pipelines in the result table"""
    result_df = pd.DataFrame(db.posts, columns=["resid", "result"])
    result_df["dataid"] = result_df.result.apply(lambda x: x.__dict__["dataid"])
    meta_df = pd.DataFrame(db.meta, columns=["id", "metadata"])
    assert (
        len(
            np.unique(
                meta_df.merge(
                    left_on="id",
                    right_on="dataid",
                    right=result_df,
                    validate="one_to_many",
                ).id.value_counts()
            )
        )
        == 1
    )


def check_desc_stats(db):
    # check if Result object has both train and test descriptive stats
    assert all(hasattr(resobj, "model_train_stats") for _, resobj in db.posts)
    assert all(hasattr(resobj, "model_test_stats") for _, resobj in db.posts)
    # check if num_samples >= num_unique
    assert all(
        resobj.model_train_stats["num_samples"]
        >= resobj.model_train_stats["num_unique"]
        for _, resobj in db.posts
    )
    assert all(
        resobj.model_test_stats["num_samples"] >= resobj.model_test_stats["num_unique"]
        for _, resobj in db.posts
    )


def test_reproducibility(reproducibility_toml):
    """Tests if two consecutive runs produce same folds and results"""
    conf = parse_job_toml(reproducibility_toml)
    exp_id = conf_get(conf, "summary.name")
    db = myrun(conf, exp_id)
    db2 = myrun(conf, exp_id)
    assert db.meta == db2.meta
    assert len(db.posts) == len(db2.posts)
    # compare the hash of the result object
    assert all([x[0] == y[0] for (x, _), (y, _) in zip(db.posts, db2.posts)])


def test_metalearning(metalearning_toml):
    conf = parse_job_toml(metalearning_toml)
    exp_id = conf_get(conf, "summary.name")
    db = myrun(conf, exp_id)
    results = db.get_results()
    try:
        metafeatures = db.get_metafeatures()
        # create dataframe from metafeatures
        pd.DataFrame([x[1] for x in metafeatures])
    except Exception as e:
        raise AssertionError(f"Invalid metafeatures. Can't create dataframe {e}")
    assert all("metafeatureid" in x.__dict__ for x in results)


def test_linear_regression(regression_toml):
    conf = parse_job_toml(regression_toml)
    exp_id = conf_get(conf, "summary.name")
    db = myrun(conf, exp_id)
    r2_scores = db.get_scores(key="r2_score")
    MSE_baseline = db.get_scores(key="MSE_baseline")
    mae_scores = db.get_scores(key="MAE")
    check_pipelines(db)
    check_desc_stats(db)
    assert (
        len(r2_scores)
        % conf_get(conf, "pipeline.cv.pipeline_performance_estimate.nfolds")
        == 0
    )
    assert np.all(r2_scores == 1)
    assert np.all(np.isclose(mae_scores, 0))
    assert np.all(MSE_baseline >= 0)


def test_imperfect_linear_regression(imperfect_regression_toml):
    conf = parse_job_toml(imperfect_regression_toml)
    exp_id = conf_get(conf, "summary.name")
    db = myrun(conf, exp_id)
    check_pipelines(db)
    check_desc_stats(db)
    r2_scores = db.get_scores(key="r2_score")
    MSE_baseline = db.get_scores(key="MSE_baseline")
    assert (
        len(r2_scores)
        % conf_get(conf, "pipeline.cv.pipeline_performance_estimate.nfolds")
        == 0
    )
    assert np.max(r2_scores < 1) and np.max(r2_scores > 0) and np.all(MSE_baseline >= 0)


def test_metadata_warning(metadata_warning_toml):
    conf = parse_job_toml(metadata_warning_toml)
    exp_id = conf_get(conf, "summary.name")
    db = myrun(conf, exp_id)
    results = db.get_results()
    path = pathlib.Path(__file__).parent.parent.joinpath("data")
    metadata.load(path.joinpath(conf_get(conf, "data.files.metadata")[0]))
    meta = metadata.get_snapshot()
    vars_with_warning = meta[~meta.warning.isna()]
    outcomes_in_results = set(x.outcome for x in results)
    assert (
        len((diff := outcomes_in_results.intersection(vars_with_warning.index))) == 0
    ), f"Invalid outcome(s) {diff} in the result"
    assert all(
        len(vars_with_warning.index.intersection(x.features)) == 0 for x in results
    )


def test_metadata_concepts(metadata_concepts_toml):
    conf = parse_job_toml(metadata_concepts_toml)
    exp_id = conf_get(conf, "summary.name")
    db = myrun(conf, exp_id)
    results = db.get_results()
    outcomes_in_results = set(x.outcome for x in results)
    concept_opt = conf_get(conf, "data.files.million_monkeys_concepts_metadata")
    js = get_concept_json(concept_opt)
    assert js is not None
    concept_outcomes = conf_get(
        conf, "data.preprocessing.outcomes.remove_outcomes_of_concept"
    )
    var_drop_set = set(
        chain.from_iterable(
            [x["varnames"] for x in js if x["concept"] in concept_outcomes]
        )
    )
    assert len(outcomes_in_results.intersection(var_drop_set)) == 0
    assert all(len(var_drop_set.intersection(x.features)) == 0 for x in results)


def test_incorrect(incorrect_toml):
    with pytest.raises(SchemaError):
        conf = parse_job_toml(incorrect_toml)
        exp_id = conf_get(conf, "summary.name")
        myrun(conf, exp_id)


# @patch("run.joblibrun.raise_not_enough_valid_outcome_data_exception")
# def test_data_exceptions(mockexception, not_enough_data_toml):
# conf = parse_job_toml(not_enough_data_toml)
# exp_id = conf_get(conf, "summary.name")
# myrun(conf, exp_id)
# assert mockexception.called
