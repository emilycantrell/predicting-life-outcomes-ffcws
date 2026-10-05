#!/usr/bin/env python3
import time
import traceback
import warnings
from exceptions import FeatureSelectionException

import metalearning
import numpy as np
import pandas as pd
from evaluation import errormetrics
from featureselection import run_feature_selection
from imbalance import resample_train
from joblib import load
from models import get_model_instance
from utils import (
    get_best_index,
    get_data_stats,
    get_logger,
    is_mode_hyperparameter_tuning,
    is_model_tunable,
    roundoff_zeros
)

logger = get_logger(__name__)


def add_data_desc_stats(dct, x_train, x_test):
    train_stats, test_stats = get_data_stats(x_train, x_test)
    dct["model_train_stats"] = train_stats
    dct["model_test_stats"] = test_stats


class Worker:
    """Class encapsulating pipeline execution functions"""

    def __init__(self, args):
        (
            (self.index, self.data_id, self.data_filename_memmap, _),
            self.outcome,
            self.outcome_type,
            (self.scaler_name),
            (self.selector_name),
            self.model,
            self.is_classification,
            self.truncate,
            (self.imbalance_opt, self.mode),
        ) = args

    def _feature_selection(self):
        self.data = load(self.data_filename_memmap, mmap_mode="r")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                (
                    self.x_train,
                    self.feature_names,
                    self.selected_features,
                    self.fs_info,
                    self.selection,
                    self.scaler,
                ) = run_feature_selection(
                    self.data, self.scaler_name, self.selector_name
                )
            except FeatureSelectionException as e:
                return [
                    {
                        "error": str(e),
                        "dataid": self.data_id,
                        "scaler": self.scaler_name,
                        "outcome": self.outcome,
                        "index": self.index,
                        "feature_selection": self.selector_name,
                        "failed": True,
                        "num_models_skipped": len(self.models),
                    }
                ]

    def _get_base_result_dict(self):
        dct = {
            "scaler": self.scaler_name,
            "outcome": self.outcome,
            "index": self.index,
            "feature_selection": self.selector_name,
            "feature_selection_stats": self.fs_info,
            "dataid": self.data_id,
            "features": self.selected_features.tolist(),
            "failed": False,
        }
        return dct

    def _is_tuning_mode_enabled(self, model):
        return is_mode_hyperparameter_tuning(self.mode) and is_model_tunable(model)

    def _fit_transform_train(self, model_name, model_class, tune_args):
        model = get_model_instance(
            model_name, model_class, self.outcome_type, self.truncate, self.scaler_name, self.selector_name, self.data.meta, tune_args
        )

        # use imbalance module to oversample minority data. Resampling can be turned on and off in jobfile. If resampling is off, this just returns the datasets fed into the function
        # x_res_inner has not gone through outerfold feature selection and will be used for innerfold tuning, where innerfold feature selection will happen
        is_resampled, (x_res_inner, y_res_inner) = resample_train(
            self.data.xtrain,
            self.data.ytrain,
            self.outcome_type,
            self.imbalance_opt,
        )
        # x_res_outer has gone through outer fold feature selection and will be used for outerfold evaluations
        is_resampled, (x_res_outer, y_res_outer) = resample_train(
            self.x_train,
            self.data.ytrain,
            self.outcome_type,
            self.imbalance_opt,
        )
        if self._is_tuning_mode_enabled(model):
            # model.fit(self.x_train, self.data.ytrain)
            # Innerfold tuning happens here:
            model.fit(x_res_inner, y_res_inner)
        else:
            model.fit(x_res_outer, y_res_outer)
        method, _, _ = self.selector_name
        # test fold is not resampled
        x_test = (
            # Use scaler and feature selection objects that were fit on training data
            # See : ../docs/notes/peeking-testing-data.md
            self.selection.transform(self.scaler.transform(self.data.xtest))
            if not isinstance(method, list)
            # old code: model.predict(self.scaler.transform(self.data.xtest[method]))
            else self.scaler.transform(self.data.xtest[method])
        )
        roundoff_zeros(x_test, inplace=True)
        prediction_function = (
            model.predict_tuned
            if self._is_tuning_mode_enabled(model)
            else model.predict
        )
        return (
            is_resampled,
            x_res_outer,
            y_res_outer,
            model,
            prediction_function,
            pd.DataFrame(x_test, columns=self.selected_features),
        )

    def _eval_model(self, y_pred_raw):
        if self.truncate:
            y_pred = np.clip(y_pred_raw, min(self.data.ytrain), max(self.data.ytrain))
        else:
            y_pred = y_pred_raw
        metrics = {}
        for metric_name, error in errormetrics(
            self.data.ytest, y_pred, self.data.ytrain, self.is_classification
        ):
            metrics[metric_name] = error
        return metrics, y_pred

    def run(self):
        """Code to be executed in worker process"""

        def _get_tune_scores(df):
            """Get scores for the hyperparameter with best inner r2_score"""
            max_iter = max(
                df.columns,
                key=lambda x: int(x.split("_")[-1]) if "best_iter_" in x else -1,
            )
            best_index = df[max_iter][0]
            scores = df.loc[best_index].filter(regex=".*holdout.*")
            return pd.Series(
                dict(
                    zip(
                        scores.index.map(lambda x: "_".join(x.split("_")[5:])),
                        scores.values,
                    )
                )
            ).to_dict()
        
        self._feature_selection()

        (model_name, model_class), tune_args = self.model
        try:
            t1 = time.time()
            dct = self._get_base_result_dict()
            dct["model"] = model_name
            (
                is_resampled,
                x_res_outer,
                y_res_outer,
                model,
                prediction_function,
                x_test,
            ) = self._fit_transform_train(model_name, model_class, tune_args)
            if is_resampled:
                dct["resampling"] = {
                    "mode": self.imbalance_opt,
                    "predictors_shape_after_resampling": x_res_outer.shape,
                    "outcome_shape_after_resampling": y_res_outer.shape,
                    "predictors_shape_before_resampling": self.x_train.shape,
                    "outcome_shape_before_resampling": self.data.ytrain.shape,
                    "outcome_dist_after_resampling": y_res_outer.value_counts().to_json(),
                    "outcome_dist_before_resampling": self.data.ytrain.value_counts().to_json(),
                }
            dct["is_tuned"] = is_model_tunable(model)
            dct["tuning"] = {}
            # add_data_desc_stats(dct, self.x_train, x_test)
            add_data_desc_stats(dct, x_res_outer, x_test)
            if self._is_tuning_mode_enabled(model):
                temp_dct = {}
                predictions_dct = {}
                for rv_index, predictions_raw in prediction_function(
                    # x_test, self.x_train, self.data.ytrain
                    x_test,
                    x_res_outer,
                    y_res_outer,
                ):
                    temp_dct[rv_index], predictions = self._eval_model(predictions_raw)
                    predictions_dct[rv_index] = (predictions, predictions_raw)

                def add_hyperparameter_tuning_results():
                    prefix = "hyperparameter_tuning"
                    df_best = pd.DataFrame(model.best_params).add_prefix(
                        f"{prefix}_best_"
                    )
                    df_scores = pd.DataFrame(temp_dct).T.add_prefix(
                        f"{prefix}_holdout_cvfold_{self.index}_"
                    )
                    df_nested = model.score_df.add_prefix(f"{prefix}_")
                    df_params = model.param_df.add_prefix(f"{prefix}_param_")
                    df_time = pd.DataFrame(model.fit_time).T.add_prefix(f"{prefix}_")
                    df_tune = pd.concat(
                        [df_best, df_scores, df_nested, df_params, df_time], axis=1
                    )
                    dct[f"{prefix}"] = df_tune.to_json()
                    dct[f"{prefix}_fs_info"] = model.cvfold_fs_info.T.to_json()
                    dct[f"{prefix}_cvfolds"] = model.cvfold_outcome_stats.T.to_json()
                    dct["scores"] = _get_tune_scores(df_tune)
                    best_index = get_best_index(df_best, prefix)[0]
                    return best_index

                best_index = add_hyperparameter_tuning_results()
                predictions, predictions_raw = predictions_dct[best_index]
            else:
                predictions_raw = prediction_function(x_test)
                dct["scores"], predictions = self._eval_model(predictions_raw)
            dct["predictions_raw"] = pd.Series(
                predictions_raw, index=self.data.ytest.index
            ).to_json()
            dct["predictions"] = pd.Series(
                predictions, index=self.data.ytest.index
            ).to_json()
            t2 = time.time()
            dct["pipeline_model_time"] = t2 - t1
            # extract metafeatures for the data that went into ML models
            meta_args = metalearning.MetaArgs(
                **{
                    "x": x_res_outer,
                    "xbeforefs": self.data.xtrain,
                    "y": y_res_outer,
                    "selected_features": self.selected_features,
                    "outcome": self.outcome,
                    "outcome_type": self.outcome_type,
                    "meta": self.data.meta,
                }
            )
            try:
                meta_dct = metalearning.extract(meta_args)
            except Exception as ex:
                meta_dct = {"error": str(ex), "failed": True}

        except Exception as ex:
            dct["scores"] = {}
            dct["error"] = str(ex)
            dct["stack_trace"] = "".join(
                traceback.TracebackException.from_exception(ex).format()
            )
            dct["failed"] = True
            meta_dct = {"error": str(ex), "failed": True}
            logger.error(ex)

        # set data to None so as to release memory in the next gc cycle
        #
        self.data = None
        return (dct, meta_dct)
