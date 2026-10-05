"""Datbase mockup
"""
from dataclasses import dataclass
from hashlib import sha1
from typing import Any

import numpy as np
from utils import track_shapes as Tracker


@dataclass(frozen=True, eq=True)
class Metadata:
    index: int
    outcome: str
    data: Any
    ystats: dict
    conf_id: str


@dataclass(frozen=True, eq=True)
class Result:
    dataid: str
    outcome: str
    scaler: str
    index: 0
    feature_selection: tuple
    model: str
    scores: dict
    features: tuple
    predictions: np.array
    predictions_raw: np.array
    model_train_stats: dict
    model_test_stats: dict
    metafeatureid: str


@dataclass(frozen=True, eq=True)
class DropsVariables:
    vartype: str
    dropstype: str
    variables: list
    msg: str


class MemoryDB:
    def __init__(self, database):
        self.conf = []
        self.meta = []
        self.posts = []
        self.metalearning = []
        self.drops = []
        self.matrix_shapes = []
        self.timing = None

    def insert_drops(self, vartype, dropstype, variables, msg):
        dropsobj = DropsVariables(vartype, dropstype, variables, msg)
        dropsid = sha1(repr(dropsobj).encode()).hexdigest()
        self.drops.append((dropsid, dropsobj))
        return dropsid

    def insert_exp_conf(self, conf):
        objid = sha1(repr(sorted(conf.items())).encode()).hexdigest()
        self.conf.append((objid, conf))
        return objid

    def delete_database(self):
        self.conf = []
        self.meta = []
        self.posts = []

    def insert_timing(self, dct):
        self.timing = dct.copy()

    def insert_cv(self, index, outcome, data, ystats, conf_id, overwrite):
        train_outcomes = data.ytrain.to_json()
        test_outcomes = data.ytest.to_json()
        metaobj = Metadata(
            index, outcome, (train_outcomes, test_outcomes), ystats, conf_id
        )
        metaid = sha1(repr(metaobj).encode()).hexdigest()
        self.meta.append((metaid, metaobj))
        return metaid

    def insert_metalearning(self, metafeature):
        featid = sha1(repr(metafeature).encode()).hexdigest()
        self.metalearning.append((featid, metafeature))
        return featid

    def insert_results(self, successes, failures, delete=False):
        for item in successes:
            if ("failed" in item and item["failed"]) or (
                "skipped" in item and item["skipped"]
            ):
                continue
            resobj = Result(
                item["dataid"],
                item["outcome"],
                item["scaler"],
                item["index"],
                item["feature_selection"],
                item["model"],
                item["scores"],
                item["features"],
                item["predictions"],
                item["predictions_raw"],
                item["model_train_stats"],
                item["model_test_stats"],
                item["metafeatureid"],
            )
            resid = sha1(repr(resobj).encode()).hexdigest()
            self.posts.append((resid, resobj))

    def insert_matrix_shape(self):
        shape_json = Tracker.to_json()
        shapeid = sha1(repr(shape_json).encode()).hexdigest()
        self.matrix_shapes.append((shapeid, shape_json))
        return shapeid

    def get_metafeatures(self):
        return self.metalearning

    def get_results(self):
        return [result for (_, result) in self.posts]

    def get_tuning_results(self):
        return [post.hyperparameter_tuning for _, post in self.posts]

    def get_scores(self, key="r2_score"):
        return np.array([post.scores[key] for _, post in self.posts])
