#!/usr/bin/env python3

from utils import conf_get, is_all_enabled


def get_outcomes(df_y, meta, job):
    """
    1. If topic opt == all and outcome_names == all then return all columns in y
    2. If topic opt == all and outcome_names == [outcome1, outcome2,...] then return intersection of outcome_names and columns in df_y
    3. If topic opt != all and outcome_names == all then return columns in df_y belonging to given topics
    4. If topic opt != all and outcome_names == [outcome1, outcome2,...] then return intersection of outcome_names and columns in df_y in given topics
    """
    topic_opt = conf_get(job, "data.outcomes.regression_outcomes.topics")
    outcome_name_opt = conf_get(job, "data.outcomes.regression_outcomes.outcome_names")
    if is_all_enabled(topic_opt):
        if is_all_enabled(outcome_name_opt):
            return df_y.columns.tolist()
        else:
            return df_y.columns.intersection(outcome_name_opt).tolist()
    else:
        if is_all_enabled(outcome_name_opt):
            return (
                meta.query("topics in @topic_opt")
                .index.intersection(df_y.columns)
                .tolist()
            )
        else:
            return (
                meta.query("topics in @topic_opt")
                .index.intersection(outcome_name_opt)
                .intersection(df_y.columns)
                .tolist()
            )
