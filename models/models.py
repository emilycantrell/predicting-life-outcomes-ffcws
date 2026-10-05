from typing import List, Tuple

import xgboost as xgb
from sklearn.base import RegressorMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import (
    BayesianRidge,
    HuberRegressor,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
    SGDRegressor,
    Lasso,
    TheilSenRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVR, NuSVR
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.dummy import DummyRegressor

from .custom_models import LogisticRegressorWithProb
from .paramgrid import MyRandomizedSearchCV, param_grid

regression_models = {
    "linear regressor": LinearRegression,
    "logistic regressor": LogisticRegressorWithProb,
    "ridge": Ridge,
    "sgd regressor": SGDRegressor,
    "lasso": Lasso,
    "lassocv": LassoCV,
    "ridgecv": RidgeCV,
    "bayesian ridge": BayesianRidge,
    "theil regressor": TheilSenRegressor,
    "huber regressor": HuberRegressor,
    "epsilon support vector regressor": lambda: SVR(
        max_iter=1000
    ),
    "linear support vector regressor": LinearSVR,
    "nu support vector regressor": lambda: NuSVR(
        max_iter=1000
    ),
    "decision tree regressor": DecisionTreeRegressor,
    "extra trees regressor": ExtraTreesRegressor,
    "knn regressor": KNeighborsRegressor,
    "adaboost regressor": lambda: AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=3, random_state=42)
    ),
    "random forest regressor": RandomForestRegressor,
    "xgboost regressor": xgb.XGBRegressor,
    "mlp regressor": MLPRegressor,
    "dummy regressor": DummyRegressor
}


def get_model_obj(model_constructor):
    """
    Instantiates model class with a fixed random state if it exists
    """
    model = model_constructor()
    if "random_state" in model.get_params():
        model.set_params(random_state=42)
    return model


def get_model_instance(model_name, model_class, outcome_type, truncate_option, scaler_name, selector_name, meta, tune_args={}):
    if len(tune_args) == 0:
        return get_model_obj(model_class)
    params = param_grid[model_name]
    niter = tune_args["param"][tune_args["mode"]]["iterations"]
    tune_folds = tune_args["param"][tune_args["mode"]]["nfolds"]
    tune_cv_method = tune_args["param"][tune_args["mode"]][
        "continuous_to_categorical_bin_method"
    ]
    cv_model = MyRandomizedSearchCV(
        get_model_obj(model_class),
        params,
        niter,
        tune_folds,
        tune_cv_method,
        outcome_type,
        truncate_option,
        scaler_name,
        selector_name,
        meta,
    )
    return cv_model


def _models(tune={}) -> List[Tuple[str, RegressorMixin]]:
    return [(k, v) for k, v in regression_models.items()]


def ensemble():
    reg = StackingRegressor(
        estimators=_models(), final_estimator=GradientBoostingRegressor(random_state=42)
    )
    return [("stacking regressor", reg)]


def classifiers():
    return [
        ("knn", KNeighborsClassifier(3)),
        ("support vector linear kernel", SVC(kernel="linear", C=0.025)),
        ("svc", SVC(gamma=2, C=1)),
        ("gaussian process", GaussianProcessClassifier(1.0 * RBF(1.0))),
        ("decision tree", DecisionTreeClassifier(max_depth=5)),
        (
            "random forest",
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        ),
        ("mlp", MLPClassifier(alpha=1, max_iter=1000)),
        ("adaboost", AdaBoostClassifier()),
        ("gaussian nb", GaussianNB()),
        ("quadratic analysis", QuadraticDiscriminantAnalysis()),
    ]


def get_model_names(modeltype):
    if modeltype == "regressors":
        return [k for k, _ in regression_models.items()]
    elif modeltype == "classifiers":
        return [x[0] for x in classifiers()]
    else:
        return []


def models(methods, is_classification=False, ignore=[]):
    if not is_classification:
        return [
            x
            for x in _models()
            if ("all" in methods or x[0] in methods) and (x[0] not in ignore)
        ]
    return [x for x in classifiers() if "all" in methods or x[0] in methods]


if __name__ == "__main__":
    pass
