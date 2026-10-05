# flake8: noqa
#!/usr/bin/env python3

import hashlib
import json
import sys
from exceptions import UsingColumbiaFollowupDataException

import click
import toml
from colorama import Fore, Style
from jsonschema import validate
from models import get_model_names
from preprocessing import scalers
from schema import And, Optional, Or, Schema, SchemaError, Use
from utils import (
    all_disjoint,
    conf_get,
    get_concept_json,
    get_logger,
    get_pobj,
    is_all_enabled,
    is_encoding_opt_valid,
)

logger = get_logger(__name__)

var_types = ["Continuous", "Binary", "Ordered Categorical", "Unordered Categorical"]
ALL_FLAG = "all"
imputation_methods = ["mean", "median", "mode"]
feature_selection_mode = ["pca", "manual", "kbest", "percentile"]
valid_waves = ["Baseline", "Year 1", "Year 3", "Year 5", "Year 9", "Year 15"]
model_names = get_model_names("regressors")
scaler_names = scalers()
model_tuning_modes = ["randomized_grid_search"]
histogram_methods = ["fd", "sturges", "quantile"]


warning_prompt = (
    Fore.RED
    + """
    drop_warning_code_predictors and/or drop_warning_code_outcomes are not as expected.
    The code not designed to handle predictors or outcomes with these warnings.
    If you want to do it, delete this check and proceed at your own risk. Good luck!
    """
)
valid_warning_codes = [1, 2, 4, 5]


def handle_warning_codes(opt):
    if not isinstance(opt, list):
        return False
    else:
        if sorted(opt) != valid_warning_codes:
            raise SchemaError(warning_prompt)
        return True


def validate_regression_outcomes(opt):
    is_valid = (
        "topics" in opt
        and "outcome_names" in opt
        and (is_all_enabled(opt["topics"]) or is_all_enabled(opt["outcome_names"]))
    )
    if not is_valid:
        raise SchemaError(
            f"Error in regression_outcomes section {opt} of job file. At least one of topics/outcomes must be all"
        )
    return is_valid


mm_metadata_schema = {
    "type": "array",
    "items": {
        "properties": {
            "concept": {
                "description": "Name of the concept",
                "type": "string",
            },
            "varnames": {
                "description": "List of variable names",
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "uniqueItems": True,
            },
        },
        "required": ["concept", "varnames"],
    },
}


def verify_varname_uniqueness(js):
    """Verify that varnames are unique across concepts"""
    li = [x["varnames"] for x in js]
    assert all_disjoint(
        li
    ), "Varnames are not unique across concepts. Check the metadata json file."


def validate_imbalancelearn(opt):
    """validate imbalance learn options"""
    if opt["mode"] == "skip":
        return True
    else:
        # raise error if imbalancelearn is enabled
        raise SchemaError(
            Fore.RED
            + """
    This is an experimental feature and is not allowed in this version.
    """
        )


def validate_json(opt):
    # verify that json file exists and md5sum matches
    if not verify_md5sum(opt):
        return False
    try:
        pobj, md5sum = get_pobj(opt)
        with open(pobj) as r:
            js = json.loads(r.read())
        validate(js, schema=mm_metadata_schema)
        verify_varname_uniqueness(js)
        return True
    except Exception as e:
        print(
            f"Error in parsing the million monkeys metadata file {pobj}. Check the json schema. {e}"
        )


def verify_md5sum(opt):
    try:
        pobj, md5sum = get_pobj(opt)
        return pobj.exists() and hashlib.md5(pobj.read_bytes()).hexdigest() == md5sum
    except Exception:
        print("Parsing error : %s" % opt)


columbia_prompt = (
    Fore.RED
    + """
The option 'remove_participants_collected_in_nonresponse_followup_Columbia' is set to False in the current job file.
The data in this set differs from standard FFCWS data. Please read the following user guide (starting on page 89)
https://fragilefamilies.princeton.edu/sites/fragilefamilies/files/year_15_guide_2021.01.13.pdf
Using Columbia data is currently not supported.
"""
    + Fore.GREEN
    + """
Please update the job file by setting the above option to true.
"""
)


def gen_schema(skip_data_check=False):
    conf_schema = Schema(
        {
            "data": {
                "participants": {
                    "participant_id_column": str,
                    "remove_participants_collected_in_nonresponse_followup_Columbia": bool,
                },
                "files": {
                    "predictors": lambda x: skip_data_check or verify_md5sum(x),
                    "outcomes": lambda x: skip_data_check or verify_md5sum(x),
                    "metadata": lambda x: skip_data_check or verify_md5sum(x),
                    Optional(
                        "metadata_edit_instructions", default=None
                    ): lambda x: skip_data_check
                    or verify_md5sum(x),
                    Optional(
                        "million_monkeys_concepts_metadata", default=None
                    ): lambda x: skip_data_check
                    or validate_json(x),
                    Optional(
                        "scales_instructions", default=None
                    ): lambda x: skip_data_check
                    or verify_md5sum(x),
                    "training_participant_ids": lambda x: skip_data_check
                    or x == "all"
                    or verify_md5sum(x),
                },
                "preprocessing": {
                    "common": {
                        "convert_standard_ff_missing_values_to_na": bool,
                    },
                    "outcomes": {
                        "recode2to0": bool,
                        "min_cities_outcomes": int,
                        Optional(
                            "remove_outcomes_with_these_warnings", default=[]
                        ): handle_warning_codes,
                        Optional("remove_weights_from_outcomes", default=False): bool,
                        Optional("remove_months_from_outcomes", default=False): bool,
                        Optional("remove_paradata_from_outcomes", default=False): bool,
                        Optional("remove_outcomes_of_concept", default=[]): list,
                        Optional("remove_skipped_outcomes", default=False): bool,
                        "min_valid_outcomes": int,
                        "vartypes_to_use": [Or(*var_types)],
                        Optional("add_scales_to_outcomes", default=False): bool,
                        Optional(
                            "remove_outcomes_with_these_warnings", default=[]
                        ): handle_warning_codes,
                        "waves_to_use": lambda lst: all(x in valid_waves for x in lst),
                    },
                    "predictors": {
                        Optional(
                            "remove_predictors_with_these_warnings", default=[]
                        ): handle_warning_codes,
                        Optional("remove_weights_from_predictors", default=False): bool,
                        Optional(
                            "remove_paradata_from_predictors", default=False
                        ): bool,
                        Optional("remove_predictors_of_concept", default=[]): list,
                        "vartypes_to_use": [Or(*var_types)],
                        "waves_to_use": lambda lst: all(x in valid_waves for x in lst),
                        "min_cities_predictors": int,
                        "remove_predictors_below_variance_threshold": float,
                        Optional("remove_all_missing_predictors", default=True): bool,
                        Optional(
                            "encoding_method_unordered_categorical",
                            default="dummy_largest",
                        ): is_encoding_opt_valid,
                        Optional(
                            "impute_mode_before_encoding_unordered_categorical",
                            default=False,
                        ): bool,
                        Optional("add_scales_to_predictors", default=False): bool,
                    },
                },
                "outcomes": {
                    "regression_outcomes": validate_regression_outcomes,
                    Optional("sample_fraction", default=1.0): lambda opt: isinstance(
                        opt, float
                    )
                    and (opt > 0 and opt <= 1),
                },
            },
            "pipeline": {
                "imputation": {
                    Or(*var_types): Or(*imputation_methods),
                },
                Optional(
                    "imbalancelearn", default={"mode": "skip"}
                ): validate_imbalancelearn,
                Optional("evaluation", default={"mode": "accuracy"}): lambda opt: "mode"
                in opt
                and opt["mode"] in ["accuracy", "hyperparameter_tuning"],
                "postprocessing": {"truncate_predictions": bool},
                "cv": {
                    "pipeline_performance_estimate": {
                        "nfolds": int,
                        "continuous_to_categorical_bin_method": lambda x: x
                        in histogram_methods,
                    },
                    Optional("model_hyperparameter_tuning", default={}): lambda x: x
                    == {}
                    or (
                        "mode" in x
                        and x["mode"] in model_tuning_modes
                        and isinstance(x["param"][x["mode"]], dict)
                    ),
                },
                "scaling": {
                    "scaler_method": lambda opt: opt == "all"
                    or (
                        isinstance(opt, list)
                        and all(x in scaler_names for x in opt)
                        and len(opt) > 0
                    ),
                },
                "feature_selection": {
                    "mode": lambda opt: all(x in feature_selection_mode for x in opt),
                    "param": {Or(*feature_selection_mode): Or(list, dict)},
                },
                "models": {
                    Or(*var_types): {
                        "methods": lambda opt: opt == ALL_FLAG
                        or (
                            isinstance(opt, list) and all(x in model_names for x in opt)
                        ),
                        Optional("ignore", default=[]): lambda opt: isinstance(
                            opt, list
                        )
                        and all(x in model_names for x in opt),
                    }
                },
            },
            "execution": {
                "parallel": bool,
                Optional("batch_size_multiplier", default=8): int
            },
            "results": {
                "database_name": str,
                "database_type": str,
                Optional("database_delete_before_run", default=True): bool,
            },
            "summary": {"name": str, "description": str},
        }
    )
    # post parsing checks
    return conf_schema


def parse_server_toml(fpath):
    options = toml.loads(open(fpath).read())
    return options


def parse_job_toml(fpath):
    options = toml.loads(open(fpath).read())
    conf = gen_schema(False).validate(options)
    model_dct = conf_get(conf, "pipeline.models")
    # check mm concept

    def check_mm_concept_consistency():
        concept_opt = conf_get(conf, "data.files.million_monkeys_concepts_metadata")
        js = get_concept_json(concept_opt)
        if js:
            concepts = [x["concept"] for x in js]
            rm_predictors = conf_get(
                conf, "data.preprocessing.predictors.remove_predictors_of_concept"
            )
            rm_outcomes = conf_get(
                conf, "data.preprocessing.outcomes.remove_outcomes_of_concept"
            )
            if not all(x in concepts for x in rm_predictors):
                raise SchemaError(
                    f"Predictor concept {rm_predictors} not in the metadata file"
                )
            if not all(x in concepts for x in rm_outcomes):
                raise SchemaError(
                    f"Outcome concept {rm_outcomes} not in the metadata file"
                )

    check_mm_concept_consistency()
    # columbia followup
    columbia_opt = conf_get(
        conf,
        "data.participants.remove_participants_collected_in_nonresponse_followup_Columbia",
    )
    if not columbia_opt:  # Columbia followup is not supported
        try:
            print(columbia_prompt)
            raise UsingColumbiaFollowupDataException(columbia_prompt)
        finally:
            print(Style.RESET_ALL)

    def check_model_spec_consistency(methods, ignore):
        if len(set(methods).intersection(ignore)) > 0 and (methods != ALL_FLAG):
            raise SchemaError(
                f"Invalid model specification {methods} and ignore {ignore}"
            )

    def check_feature_selection_spec_consistency(mode, param):
        # check that param keys match the list of modes
        if set(mode) != set(param.keys()):
            raise SchemaError(
                f"Feature selection modes {mode} and parameters {param} don't match"
            )
        for m in mode:
            if m in ["kbest", "pca", "percentile"]:
                if "arg" not in param[m]:
                    raise SchemaError(
                        f"Missing argument for mode {m} in the param dict {param[m]}"
                    )
            if m == "manual":
                manual_param = param[m]
                if not isinstance(manual_param, dict) or any(
                    not isinstance(v, list) for v in manual_param.values()
                ):
                    raise SchemaError(
                        "Manual feature selection param must be a dict with each value being a list of features."
                    )

    check_feature_selection_spec_consistency(
        conf_get(conf, "pipeline.feature_selection.mode"),
        conf_get(conf, "pipeline.feature_selection.param"),
    )
    # check that the model spec is compatible with remove_outcome_type
    # d1 = set(model_dct.keys()) # outcome types in model dict
    # d2 = set(conf_get(conf, "pipeline.outcomes.preprocessing.remove_outcomes_of_vartype"))
    # if len(d1.intersection(d2)) > 0 or len(set(var_types).difference(d1.union(d2))) > 0:
    # raise SchemaError(f"Model spec {model_dct} is incompatible with pipeline remove outcome type {d2}")

    for outcome_type, dct in model_dct.items():
        check_model_spec_consistency(dct["methods"], dct.get("ignore", None))
    if (
        len(
            set(conf_get(conf, "pipeline.imputation").keys()).difference(
                set(conf_get(conf, "data.preprocessing.predictors.vartypes_to_use"))
            )
        )
        > 0
    ):
        raise SchemaError("Incompatible imputation speficiation")
    if conf_get(conf, "pipeline.cv.pipeline_performance_estimate.nfolds") > conf_get(
        conf, "data.preprocessing.outcomes.min_valid_outcomes"
    ):
        raise SchemaError("nfolds more than min_valid_outcomes")
    if conf_get(conf, "data.preprocessing.outcomes.add_scales_to_outcomes"):
        if len(conf_get(conf, "data.files.scales_instructions")) == 0:
            raise SchemaError(
                "add_scales_to_outcomes option enabled but scales data files are missing"
            )
    return conf


if __name__ == "__main__":
    parse_job_toml(sys.argv[1])
