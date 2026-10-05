# execute pipeline using joblib memory and parallel
#
import os
import os.path
import shutil
import sys
import traceback
import warnings
from exceptions import NotEnoughValidOutcomeDataException, TrainTestSplitException
from itertools import cycle, product

import pandas as pd
from data import SplitArgs, cv, get_data
from database import MemoryDB, ResultsDB
from featureselection import fselection
from joblib import Memory, Parallel, delayed
from models import models
from pandas.core.common import SettingWithCopyWarning
from pandas.errors import DtypeWarning
from pipelines import Worker
from preprocessing import get_scaler_keys, remove_missing
from scripts import parse_job_toml
from utils import (
    conf_get,
    dict_hash,
    get_dropped_cols,
    get_git_revision_short_hash,
    get_logger,
    timing,
    track_dropped_cols,
)

from tqdm import tqdm
from utils import track_shapes as Tracker, tqdm_joblib

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", message=".*", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", message=".*", category=DtypeWarning)
warnings.filterwarnings(action="ignore", message=".*", category=SettingWithCopyWarning)

logger = get_logger(__name__)


def gen_splits(
    X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, db: ResultsDB, args: SplitArgs
):
    splits = [x for x in cv(X, y, meta, db, args)]
    for index, _, _, data in splits:
        Tracker.add(
            Tracker.TrackInfo(
                Tracker.ModuleType.PIPELINE,
                Tracker.MatrixType.PREDICTORS,
                data.xtrain.shape,
                f"Outcome = {args.outcome}, outer training fold index = {index}",
            )
        )
        Tracker.add(
            Tracker.TrackInfo(
                Tracker.ModuleType.PIPELINE,
                Tracker.MatrixType.OUTCOMES,
                data.ytrain.shape,
                f"Outcome = {args.outcome}, outer training fold index = {index}",
            )
        )
        Tracker.add(
            Tracker.TrackInfo(
                Tracker.ModuleType.PIPELINE,
                Tracker.MatrixType.PREDICTORS,
                data.xtest.shape,
                f"Outcome = {args.outcome}, outer testing fold index = {index}",
            )
        )
        Tracker.add(
            Tracker.TrackInfo(
                Tracker.ModuleType.PIPELINE,
                Tracker.MatrixType.OUTCOMES,
                data.ytest.shape,
                f"Outcome = {args.outcome}, outer testing fold index = {index}",
            )
        )
    return splits


def prepare_run(conf, exp_id):
    def jobget(fieldname):
        return conf_get(conf, fieldname)

    KEY = jobget("data.participants.participant_id_column")
    dbname = jobget("results.database_name")
    dbtype = jobget("results.database_type")
    if dbtype == "mongo":
        db = ResultsDB(remote=False, database=dbname)
    elif dbtype == "inmemory":
        db = MemoryDB(database=dbname)
    else:
        logger.error("Invalid database")
        return
    if jobget("results.database_delete_before_run"):
        logger.info("Deleting database %s before running pipelines" % dbname)
        db.delete_database()
    logger.info("Reading dataset")
    with timing.MeasureBlockTime("data preprocessing"):
        X, y, meta = get_data(conf)
    conf["execution"]["execution_id"] = exp_id
    logger.info("Execution id : %s ", exp_id)
    conf["execution"]["git-rev-hash"] = get_git_revision_short_hash()
    logger.info("Git revision hash: %s ", get_git_revision_short_hash())
    conf_id = db.insert_exp_conf(conf)

    n_jobs = -1 if jobget("execution.parallel") else 1
    logger.info("Number of concurrent jobs : %d", n_jobs)
    return conf_id, jobget, db, X, y, meta, KEY, n_jobs


def run(wobj):
    return wobj.run()


def task_runner(parallel, grp_tasks):
    """Creates worker objects for each task and schedules it to run on parallel"""
    worker_objs = [Worker(args) for args in grp_tasks]
    with tqdm_joblib(tqdm(desc="Pipelines", total=len(worker_objs))):
        return parallel(delayed(run)(wobj) for wobj in worker_objs)


def get_temp_dir_prefix():
    # check if /datadrive/experiments exists or if we can create then return it
    # otherwise use local directory
    prefix = "/datadrive/experiments"
    try:
        os.makedirs(prefix)
    except FileExistsError:
        return prefix
    except (PermissionError, OSError):
        return "." + prefix


# flake8: noqa: C901
def myrun(conf, exp_id):
    """Runs prediction pipelines

    Run prediction pipeline for the given outcomes in the dataset
    This function spawns a worker task for each combination of data scaler,
    feature selection, and modeling method. These tasks are scheduled to
    execute on maximum number of cores.
    """

    temp_dir_prefix = get_temp_dir_prefix()
    with timing.MeasureBlockTime("prepare run"):
        conf_id, jobget, db, X, y, meta, KEY, n_jobs = prepare_run(conf, exp_id)

    max_tasks = os.cpu_count() * jobget("execution.batch_size_multiplier")
    tasks = []
    mmdirs = []

    def run_pipelines(outcomes, is_classification):

        nonlocal mmdirs, tasks

        def _run(parallel, db, mmdirs):
            nonlocal tasks
            with timing.MeasureBlockTime(f"pipeline group of size {len(tasks)}"):
                worker_results = task_runner(parallel, tasks)
                metafeature_hashes = {}
                for model_result, metafeature in worker_results:
                    # check if metafeature for this outcome, fold is already in the collection
                    # NOTE : It's enough to check in this batch of results because pipelines for an outcome
                    # aren't split across task batches.
                    metalearning_hash = dict_hash(metafeature)
                    outcome = model_result["outcome"]
                    index = model_result["index"]
                    if (
                        mhash := (outcome, index, metalearning_hash)
                    ) not in metafeature_hashes:
                        metaid = db.insert_metalearning(metafeature)
                        metafeature_hashes[mhash] = metaid
                    else:
                        metaid = metafeature_hashes[mhash]
                    results = []
                    model_result["metafeatureid"] = metaid
                    results.append(model_result)
                    db.insert_results(results, None, False)
            for mdir in mmdirs:
                logger.info("Deleting memorymap directory %s" % mdir)
                shutil.rmtree(mdir, ignore_errors=True)
            tasks = []
            mmdirs = []

        with Parallel(
            n_jobs=n_jobs,
            verbose=10,
            temp_folder="./joblib_worker_share",
            mmap_mode="r",
        ) as parallel:
            try:
                for index, outcome in enumerate(outcomes):
                    remaining = len(outcomes) - 1 - index
                    logger.info("Executing pipelines for outcome = %s", outcome)
                    logger.info("Removing missing outcomes and negative outcomes")
                    try:
                        X1, y1 = remove_missing(X, y, outcome)
                    except KeyError:
                        logger.error("Outcome dropped during preprocessing")
                        track_dropped_cols(
                            "outcomes",
                            "missing",
                            [outcome],
                            "Outcome missing in the data",
                        )
                        continue
                    Tracker.add(
                        Tracker.TrackInfo(
                            Tracker.ModuleType.PIPELINE,
                            Tracker.MatrixType.PREDICTORS,
                            X1.shape,
                            f"Outcome = {outcome} after dropping cases with missing outcome data",
                        )
                    )
                    Tracker.add(
                        Tracker.TrackInfo(
                            Tracker.ModuleType.PIPELINE,
                            Tracker.MatrixType.OUTCOMES,
                            y1.shape,
                            f"Outcome = {outcome}",
                        )
                    )
                    # get outcome type from metadata table
                    try:
                        outcome_type = meta.loc[outcome].type
                        model_dct = jobget("pipeline.models")
                        model_dct[outcome_type]["methods"]  # to check for errors
                    except IndexError:
                        logger.error(f"Cannot infer outcome type {outcome}")
                        track_dropped_cols(
                            "outcomes",
                            "outcome_type",
                            [outcome],
                            "Cannot infer outcome type",
                        )
                        continue
                    except KeyError:
                        logger.error(f"Missing model spec for type {outcome_type}")
                        continue
                    mem_map_dir = "%s/joblib_memmap_%s_%s" % (
                        temp_dir_prefix,
                        outcome,
                        is_classification,
                    )
                    mmdirs.append(mem_map_dir)
                    logger.info("Memmap directory : %s ", mem_map_dir)
                    with timing.MeasureBlockTime("generate cv splits"):
                        try:
                            nfolds = jobget(
                                "pipeline.cv.pipeline_performance_estimate.nfolds"
                            )
                            splits = gen_splits(
                                X1,
                                y1,
                                meta,
                                db,
                                SplitArgs(
                                    outcome,
                                    outcome_type,
                                    conf_id,
                                    nfolds,
                                    mem_map_dir,
                                    conf,
                                ),
                            )
                        except NotEnoughValidOutcomeDataException as e:
                            track_dropped_cols(
                                "outcomes", "data", [outcome], "not enough valid data"
                            )
                            logger.exception(e, exc_info=True)
                            continue
                        except TrainTestSplitException as e:
                            track_dropped_cols(
                                "outcomes",
                                "data",
                                [outcome],
                                "train test split exception",
                            )
                            logger.exception(e, exc_info=True)
                            continue
                    data_scale_keys = get_scaler_keys(
                        jobget("pipeline.scaling.scaler_method")
                    )
                    logger.info("Data scaling methods : %s", str(data_scale_keys))
                    selection = fselection(
                        jobget("pipeline.feature_selection"),
                        X1.shape[1],
                        is_classification=is_classification,
                    )
                    logger.info("Feature selection methods : %s", str(selection))
                    model_tune = (jobget("pipeline.cv.model_hyperparameter_tuning"),)
                    learning_models = [
                        x
                        for x in zip(
                            list(
                                models(
                                    methods=model_dct[outcome_type]["methods"],
                                    is_classification=is_classification,
                                    ignore=model_dct[outcome_type]["ignore"],
                                )
                            ),
                            cycle(model_tune),
                        )
                    ]
                    logger.info("Learning methods : %s", str(learning_models))
                    if not learning_models:
                        logger.warn(
                            "Based on pipeline.models.methods = %s and pipeline.models.ignore = %s there are no learning models"
                            % (
                                jobget("pipeline.models.methods"),
                                jobget("pipeline.models.ignore"),
                            )
                        )
                        continue
                    eval_mode = jobget("pipeline.evaluation.mode")
                    imbalance_mode = jobget("pipeline.imbalancelearn")
                    job_iter = product(
                        splits,
                        [outcome],
                        [outcome_type],
                        data_scale_keys,
                        selection,
                        learning_models,
                        [is_classification],
                        [
                            jobget(
                                "pipeline.postprocessing.truncate_predictions",
                            )
                        ],
                        [(imbalance_mode, eval_mode)]
                    )
                    tasks.extend([x for x in job_iter])
                    if (len(tasks) > max_tasks) or remaining == 0:
                        _run(parallel, db, mmdirs)
                if len(tasks) > 0:
                    _run(parallel, db, mmdirs)
            except Exception as e:
                traceback.print_exc()
                logger.exception(e, exc_info=True)

    regression_outcomes = y.columns

    def insert_drops_to_db():
        warnings = get_dropped_cols()
        for category, warning_list in warnings.items():
            for var_type, outcomes, msg in warning_list:
                db.insert_drops(
                    var_type, category, [outcome for outcome in outcomes], msg
                )

    logger.info(f"Running pipeline with outcomes : {regression_outcomes}")
    run_pipelines(regression_outcomes, False)
    # add warnings to database
    insert_drops_to_db()
    # add timing information to database
    timing_dct = timing.generate_dict_report()
    timing_dct["conf_id"] = conf_id
    db.insert_timing(timing_dct)
    # add matrix shape tracker
    db.insert_matrix_shape()
    return db


if __name__ == "__main__":
    conf = parse_job_toml(sys.argv[1])
    exp_id = conf["execution"]["name"]
    myrun(conf, exp_id)
