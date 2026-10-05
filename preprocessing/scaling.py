from typing import List, Union

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    OneHotEncoder
)

keys: List[str] = [
    "identity",
    "maxabs scaler",
    "minmax scaler",
    "quantile transformer normal",
    "quantile transformer uniform",
    "robust scaler",
    "standard scaler",
]


class IdentityTransformer:
    def fit_transform(self, X, y=None, **fit_params):
        return X

    def transform(self, X, **fit_params):
        return X


class MinMax_NoDivZero:
    """Implementation of MaxMax
    for columns with same values on all the rows, the values are replaced by
    1 if identical value != 0 else 0
    """

    def fit_transform(self, X, y=None, **fit_params):
        return X

    def transform(self, X, **fit_params):
        return X


def scalers() -> List[str]:
    return keys


def get_scaler_keys(confitem):
    return scalers() if confitem == "all" else confitem


def get_scaler_by_key(
    key: str,
) -> Union[IdentityTransformer, MaxAbsScaler, MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler]:
    if key == "identity":
        return IdentityTransformer()
    if key == "maxabs scaler":
        return MaxAbsScaler()
    elif key == "minmax scaler":
        return MinMaxScaler()
    elif key == "quantile transformer normal":
        return QuantileTransformer(output_distribution='normal')
    elif key == "quantile transformer uniform":
        return QuantileTransformer(output_distribution='uniform')
    elif key == "robust scaler":
        return RobustScaler()
    elif key == "standard scaler":
        return StandardScaler()
    elif key == "one hot encoding":
        return OneHotEncoder(handle_unknown="ignore")
    else:
        raise KeyError("Cannot find a data scaler with name : %s" % key)
