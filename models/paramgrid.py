#!/usr/bin/env python3
# Grid version 2024-03-27
from utils import constants, get_cvfold_ystats, get_logger, to_categorical, rng
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats import expon, uniform
import pandas as pd
from data import DataSplit
import numpy as np
from collections import defaultdict
import warnings
import types
import time
import random
import math
from featureselection import run_feature_selection


def warn(*args, **kwargs):
    pass


# hyperparameter grids for models
warnings.warn = warn


logger = get_logger(__name__)


def generate_hidden_layer_param(arg, niter):
    """Generate hidden layer params"""
    nlayers, max_in_each = arg
    ret = []
    np.random.seed(42)
    for itr in range(niter):
        itr_layers = np.random.randint(1, nlayers + 1)
        itr_neurons = tuple(
            [np.random.randint(2, max_in_each + 1) for _ in range(itr_layers)]
        )
        ret.append(itr_neurons)
    return ret


param_grid = {
    "epsilon support vector regressor": {
        "C": expon(scale=1),  # p(x) = 1/scale * exp(-x/scale); Hastie et al. p421
        "kernel": ["linear", "poly", "rbf", "sigmoid"], # Hastie et al. p424
        "degree": [2, 3, 5, 10], # Hastie et al. p431
        "gamma": [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5], # Hastie et al. p432
        "max_iter": [50, 100, 500, 1000, 2000],
    },
    "nu support vector regressor": {
        "nu": uniform(loc=0, scale=1),  # random [0, 1]
        "C": expon(scale=1), # p(x) = 1/scale * exp(-x/scale); Hastie et al. p421
        "kernel": ["linear", "poly", "rbf", "sigmoid"], # Hastie et al. p424
        "degree": [2, 3, 5, 10], # Hastie et al. p431
        "gamma": [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5], # Hastie et al. p432
        "max_iter": [100, 500, 1000, 2000, 3000],
    },
    "linear support vector regressor": {
        "C": expon(scale=1),  # p(x) = 1/scale * exp(-x/scale); Hastie et al. p421
        "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
    },
    "linear regressor": {
        "fit_intercept": [True],
    },
     "lasso": {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10], # Hastie et al. p663
    },
    "ridge": {
        "alpha": [0.3, 1, 2, 3, 10, 20, 30], 
    },
    "sgd regressor": {
        "penalty": ["l1", "l2", "elasticnet"], # Hastie et al. p62
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10], # Hastie et al. p663
        "l1_ratio": [0.15, 0.3, 0.5, 0.7, 0.85], # Hastie et al. p663
    },
    "logistic regressor": {
        "penalty": ["l1", "l2", "elasticnet", 'none'], # Hastie et al. p62
        "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], # Hastie et al. p663
        "l1_ratio": [0.15, 0.3, 0.5, 0.7, 0.85], # Hastie et al. p663
        "solver": ["saga"],
    },
    "bayesian ridge": {
        "alpha_init": [0.001, 0.01, 0.1, 1, None], # sklearn curve fitting with bayesian ridge regression
        "lambda_init": [0.001, 0.01, 0.1, 1, None],
    },
    "decision tree regressor": {
        "criterion": ["squared_error", "absolute_error"],
        "ccp_alpha": [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5], # Hastie et al. p308 # see sklearn
    },
    "random forest regressor": {
        "n_estimators": [10, 50, 100, 200], # Hastie et al. p589
        "criterion": ["squared_error", "absolute_error"],
        "min_samples_split": [2, 5, 10, 20, 30, 50], # Hastie et al. 598
        "max_features": [None, 1, 2, 5, 10, "sqrt", "log2"], # Hastie et al. p591
    },
    "extra trees regressor": {
        "n_estimators": [10, 50, 100, 200],  # Hastie et al. p589
        "criterion": ["squared_error", "absolute_error"],
        "min_samples_split": [2, 5, 10, 20, 30, 50, 80], # Hastie et al. 598
        "max_features": [None, 1, 2, 5, 10, "sqrt", "log2"], # Hastie et al. p591
    },
    "adaboost regressor": {
        "n_estimators": [10, 50, 100, 200], # Hastie et al. p589
        "base_estimator__criterion": ["squared_error", "absolute_error"],
        "base_estimator__max_leaf_nodes": [2, 10, 100, None], # Hastie et al. p363
        "base_estimator__max_depth": [3, None],
        "base_estimator__max_features": [None, 1, 2, 5, 10, "sqrt", "log2"], # Hastie et al. p591
        "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1], # Hastie et al. 365
        "loss": ["linear", "square", "exponential"], # Hastie et al. p345
    },
    "xgboost regressor": {
        "n_estimators": [10, 50, 100, 200], # Hastie et al. p589
        "max_depth": [1, 2, 3, 4, 5, 6], # Hastie et al. p591
        "colsample_bytree": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1], # Hastie et al. p591
        "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1], # Hastie et al. 365
    },
    "mlp regressor": {
        "hidden_layer_sizes": (generate_hidden_layer_param, (6, 6)), # Hastie et al. p400
        "activation": ["identity", "logistic", "tanh", "relu"],
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10], 
    },
    "knn regressor": {
        "n_neighbors": [1, 2, 5, 10, 20, 50, 100, 150], # Hastie et al. p17
        "weights": ["uniform", "distance"],
        "p": [1, 2, 3, 5, 10],
    },
    "dummy regressor": {
        "strategy": ["mean"]
    },
}


class MyRandomizedSearchCV:
    def __init__(self, clf, params, n_iter, n_splits, bin_method, output_type, truncate_option, scaler_name, selector_name, meta):
        self.model = clf
        self.params = params
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.bin_method = bin_method
        self.output_type = output_type
        self.fit_time = defaultdict(lambda: defaultdict(float))
        self.truncate = truncate_option
        self.scaler_name = scaler_name
        self.selector_name = selector_name
        self.meta = meta

    def is_tunable(self):
        return self.params != {}

    def fit(self, X, y):
        np.random.seed(42)
        if self.is_tunable():
            switch = False
            if self.output_type == constants.CONTINUOUS:
                y1, _ = to_categorical(
                    y, bin_method=self.bin_method, param=self.n_splits
                )
                switch = True
            kf = StratifiedKFold(n_splits=self.n_splits,
                                 shuffle=True, random_state=42)
            params = {}
            for k, v in self.params.items():
                if isinstance(v, rv_frozen):
                    params[f"{k}"] = v.rvs(self.n_iter, random_state=42)
                elif isinstance(v, list):
                    params[f"{k}"] = np.random.choice(v, self.n_iter)
                elif isinstance(v, tuple) and isinstance(v[0], types.FunctionType):
                    func, func_params = v
                    params[f"{k}"] = func(func_params, self.n_iter)
                # can't set None values in int integer array; convert to list
                params[f"{k}"] = [_ for _ in params[f"{k}"]]
                # set the first value to model default parameter
                params[f"{k}"][0] = self.model.get_params()[k]

            self.param_df = pd.DataFrame(
                params, index=[f"rv_{i}" for i in range(self.n_iter)]
            )
            # Pandas converts columns with ints and None into Floats. convert_dtypes restores the types
            self.param_df = self.param_df.convert_dtypes()
            # replace nan with None
            self.param_df.replace({np.nan: None}, inplace=True)
            # remove duplicates in parameter grid - always keep the first (default parameter)
            new_index = (
                self.param_df.drop_duplicates(keep="first")
                .reset_index(drop=True)
                .index.map(lambda x: f"rv_{x}")
            )
            self.param_df = self.param_df.drop_duplicates().set_index(new_index)
            self.n_iter = len(self.param_df)
            self.cvfold_fs_info = pd.Series(name="cvfold_fs_info")
            self.cvfold_outcome_stats = pd.Series(name="cvfold_outcome_stats")

            # if model has n_jobs then set it to 1
            if hasattr(self.model, "n_jobs"):
                self.param_df["n_jobs"] = 1
            scores = defaultdict(lambda: defaultdict(float))
            for index, (train_index, test_index) in enumerate(
                kf.split(X, y1 if switch else y)
            ):
                # Prepare the data so we can run feature selection on inner folds
                data = DataSplit(
                    xtrain=X.iloc[train_index, :],
                    xtest=X.iloc[test_index, :],
                    ytrain=y.iloc[train_index],
                    ytest=y.iloc[test_index],
                    meta=self.meta,
                )
                (
                    x_train,
                    feature_names,
                    selected_features,
                    fs_info,
                    selection,
                    scaler,
                ) = run_feature_selection(
                    data, self.scaler_name, self.selector_name
                )
                method, _, _ = self.selector_name
                # Reduce innerfold test set to the set of features selected based on the innerfold training set
                x_test = (
                    # Use scaler and feature selection objects that were fit on training data
                    selection.transform(scaler.transform(data.xtest))
                    if not isinstance(method, list)
                    else scaler.transform(data.xtest[method])
                )
                self.cvfold_fs_info.at[f"fold_{index}"] = fs_info
                self.cvfold_outcome_stats.at[f"fold_{index}"] = get_cvfold_ystats(
                    y, train_index, test_index
                )
                # track variance
                for i in range(self.n_iter):
                    try:
                        self.model.set_params(
                            **dict(
                                zip(self.param_df.columns,
                                    self.param_df.loc[f"rv_{i}"])
                            )
                        )
                        t1 = time.time()
                        # fit with above parameters
                        self.model.fit(x_train,
                                       y.iloc[train_index])
                        if self.truncate:
                            predictions = np.clip(self.model.predict(x_test), y.iloc[train_index].min(), y.iloc[train_index].max())
                        else:
                            predictions = self.model.predict(x_test)
                        baseline = y.iloc[train_index].mean()
                        tse_predictions = ((predictions - y.iloc[test_index]) ** 2).sum()
                        tse_baseline = ((baseline - y.iloc[test_index]) ** 2).sum()
                        r2_score = 1.0 - (tse_predictions / tse_baseline)
                        scores[f"fold_{index}_training_mean"][f"rv_{i}"] = baseline
                        scores[f"fold_{index}_tse_predictions"][f"rv_{i}"] = tse_predictions
                        scores[f"fold_{index}_tse_baseline"][f"rv_{i}"] = tse_baseline
                        scores[f"fold_{index}_r2_score"][f"rv_{i}"] = r2_score
                        less_mean = (predictions < y.iloc[train_index].min()) & (
                            predictions <= y.iloc[train_index].mean()
                        )
                        more_mean = (predictions > y.iloc[train_index].max()) & (
                            predictions >= y.iloc[train_index].mean()
                        )
                        scores[f"fold_{index}_num_extreme"][f"rv_{i}"] = (
                            (predictions < y.iloc[train_index].min())
                            | (predictions > y.iloc[train_index].max())
                        ).sum()
                        scores[f"fold_{index}_num_predictions"][f"rv_{i}"] = len(
                            predictions
                        )
                        scores[f"fold_{index}_num_extreme_less_than_mean"][
                            f"rv_{i}"
                        ] = less_mean.sum()
                        scores[f"fold_{index}_num_extreme_more_than_mean"][
                            f"rv_{i}"
                        ] = more_mean.sum()
                        scores[f"fold_{index}_num_extreme_same_side_of_mean"][
                            f"rv_{i}"
                        ] = (less_mean | more_mean).sum()
                        scores[f"fold_{index}_coef"][f"rv_{i}"] = (
                            self.model.coef_ if hasattr(
                                self.model, "coef_") else []
                        )
                        assert (
                            scores[f"fold_{index}_num_extreme"][f"rv_{i}"]
                            >= scores[f"fold_{index}_num_extreme_same_side_of_mean"][
                                f"rv_{i}"
                            ]
                        )
                        t2 = time.time()
                        self.fit_time[f"rv_{i}"][f"fold_{index}_time"] = t2 - t1
                    except Exception as e:
                        logger.error(
                            f"Base model raised an exception {self.model} {self.params} {self.output_type} {e}"
                        )
                        raise
                
            # normalize inner fold fs into
            self.cvfold_fs_info = pd.json_normalize(
                self.cvfold_fs_info)
            self.cvfold_fs_info.index = [
                f"inner_fold_{i}" for i in self.cvfold_fs_info.index
            ]
            # normalize inner fold outcome stats
            self.cvfold_outcome_stats = pd.json_normalize(
                self.cvfold_outcome_stats)
            self.cvfold_outcome_stats.index = [
                f"inner_fold_{i}" for i in self.cvfold_outcome_stats.index
            ]
            self.score_df = pd.DataFrame(scores)
            # get best params at each experiment size, key is exp size and value is dict with key = split id and value is index
            self.best_params = defaultdict(lambda: defaultdict(str))
            # for rv_size in [2 ** i for i in range(int(math.log(self.n_iter, 2)) + 1)]:
            try:
                for rv_size in [
                    min(2 ** i, self.n_iter)
                    for i in range(math.ceil(math.log(self.n_iter, 2)) + 1)
                ]:
                    start = 0
                    end = rv_size
                    index = 0
                    while end <= self.n_iter and start != end:
                        # self.best_params[rv_size][index] = self.score_df.iloc[range(start, end)].mean(axis=1).idxmax()
                        self.best_params[f"iter_{rv_size}"][f"rv_{index}"] = (
                            self.score_df[
                                self.score_df.columns[
                                    self.score_df.columns.str.endswith(
                                        "r2_score")
                                ]
                            ]
                            .iloc[range(start, end)]
                            .mean(axis=1)
                            .idxmax()
                        )
                        index += 1
                        start = end
                        end = min(start + rv_size, self.n_iter)
            except Exception as e:
                logger.error(
                    f"Hyper-parameter tuning raised an exception {self.model} {self.params} {self.output_type} {e}"
                )
                raise
        else:
            self.best_params = {}
            self.scores = {}
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_tuned(self, X, X_train, y_train):
        """predictions based on tuned model"""
        for rv_index, row in self.param_df.iterrows():
            self.model.set_params(**row)
            self.model.fit(X_train, y_train)
            # predictions based on this set of paramters
            predictions = self.model.predict(X)
            yield rv_index, predictions


def generate_cv_regressor(clf, params, n_iter, nfolds, bin_method):
    return RandomizedSearchCV(clf, params, n_iter=n_iter, n_jobs=1, random_state=rng)
