#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Tuple

import pandas as pd


class ModuleType(str, Enum):
    PREPROCESSING = "Preprocessing"
    PREPROCESSING_ENCODING = "Preprocessing_encoding"
    PIPELINE = "Pipeline"

    def __str__(self):
        return self.name


class MatrixType(str, Enum):
    PREDICTORS = "Predictors"
    OUTCOMES = "Outcomes"
    METADATA = "Metadata"
    PREDICTORS_DUMMIES = "Predictors_dummies"
    PREDICTORS_TRAINING = "Predictors_training"
    PREDICTORS_TESTING = "Predictors_testing"

    def __str__(self):
        return self.name


@dataclass(frozen=True, eq=True)
class TrackInfo:
    module: ModuleType
    matrix: MatrixType
    shape: Tuple[int, int]
    step: str


class ShapeTracker:
    def __init__(self):
        self.tracker = []

    def add(self, info):
        self.tracker.append((info, str(datetime.now())))

    def to_df(self):
        df = pd.DataFrame([{**x.__dict__, **{"time": t}} for x, t in self.tracker])
        df.index.name = "global_stepid"
        df["group_stepid"] = df.groupby(["module", "matrix"]).cumcount() + 1
        df.reset_index(inplace=True)
        df.global_stepid = df.global_stepid + 1
        df[["nrows", "ncols"]] = df.apply(
            lambda row: pd.Series(
                [row["shape"][0], 1 if len(row["shape"]) == 1 else row["shape"][1]]
            ),
            axis=1,
        )
        df.drop("shape", axis=1, inplace=True)
        df = df.reindex(sorted(df.columns), axis=1)
        return df.sort_values(["global_stepid"])

    def to_json(self):
        return self.to_df().to_json()


_shapetracker = ShapeTracker()
add = _shapetracker.add
to_df = _shapetracker.to_df
to_json = _shapetracker.to_json
