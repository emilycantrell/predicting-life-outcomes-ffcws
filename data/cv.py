import os
from collections import Counter
from dataclasses import dataclass
from exceptions import TrainTestSplitException

import pandas as pd
from data import DataSplit
from database import ResultsDB
from joblib import Parallel, delayed, dump, load
from preprocessing import preprocess
from scipy.stats import describe
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from utils import conf_get, constants, to_categorical, tqdm_joblib, rng


@dataclass
class SplitArgs:
    outcome: str
    outcome_type: str
    conf_id: str
    n_folds: int = "10"
    memmapdir: str = "./joblib_memmap"
    conf: dict = None


def cv(
    X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, db: ResultsDB, args: SplitArgs
):
    """Generate cross-validation splits

    :param X: predictor matrix
    :param y: outcome vector
    :param meta: metadata matrix
    :param db: database object
    :param args: arguments for cross-validation splits
    :raises TrainTestSplitException: Exception if cv splits can't be generated for a given type
    :return: None
    :yield: index of the split, objectid in the database, memmap file, data object
    """
    try:
        os.makedirs(args.memmapdir)
    except FileExistsError:
        pass

    def worker(index, train_index, test_index):
        data_filename_memmap = os.path.join(args.memmapdir, "data_memmap_%d" % index)
        data = DataSplit(
            xtrain=X.iloc[train_index, :],
            xtest=X.iloc[test_index, :],
            ytrain=y.iloc[train_index],
            ytest=y.iloc[test_index],
            meta=meta,
        )
        assert data.xtrain.shape[0] == data.ytrain.shape[0]
        assert data.xtest.shape[0] == data.ytest.shape[0]
        preprocess(data, meta, args.conf)
        dump(data, data_filename_memmap)
        return index, data_filename_memmap

    def get_ystats(data, outcome_type):
        y_train_stats = describe(data.ytrain)
        y_test_stats = describe(data.ytest)
        stats_df = pd.DataFrame(
            [y_train_stats, y_test_stats],
            columns=y_train_stats._fields,
            index=["train", "test"],
        )
        # Drop minmax and add min, max columns
        stats_df["min"] = stats_df.minmax.apply(lambda x: x[0])
        stats_df["max"] = stats_df.minmax.apply(lambda x: x[1])
        stats_df.drop("minmax", axis=1, inplace=True)
        if outcome_type != constants.CONTINUOUS:
            stats_df["freq"] = [
                [(k, v) for k, v in Counter(data.ytrain).items()],
                [(k, v) for k, v in Counter(data.ytest).items()],
            ]
        return stats_df.to_dict()

    enable_proxy = False
    if args.outcome_type == constants.CONTINUOUS:
        method = conf_get(
            args.conf,
            "pipeline.cv.pipeline_performance_estimate.continuous_to_categorical_bin_method",
        )
        y_proxy, _ = to_categorical(y, bin_method=method, param=args.n_folds)
        enable_proxy = True
    kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    try:
        worker_args = [
            (index, train_index, test_index)
            for index, (train_index, test_index) in enumerate(
                kf.split(X, y_proxy if enable_proxy else y)
            )
        ]
    except ValueError:
        raise TrainTestSplitException(
            f"Train test split error with method {kf} on {args.outcome} of type (from metadata) {args.outcome_type}"
        )
    with tqdm_joblib(tqdm(desc="Parallel cv splits", total=len(worker_args))):
        results = Parallel(n_jobs=-1)(delayed(worker)(*args) for args in worker_args)

    for index, data_filename_memmap in results:
        data = load(data_filename_memmap, mmap_mode="r")
        ystats = get_ystats(data, args.outcome_type)
        dataid = db.insert_cv(index, args.outcome, data, ystats, args.conf_id, False)
        if data.ytrain.var() == 0:
            dct = {
                "outcome": args.outcome,
                "index": index,
                "outcome_type": args.outcome_type,
                "failed": False,
                "dataid": dataid,
                "msg": "Zero variance in the training fold for outcome",
            }
            db.insert_drops("outcomes", "outcome_fold", [args.outcome], dct)
        else:
            yield index, str(dataid), data_filename_memmap, data
