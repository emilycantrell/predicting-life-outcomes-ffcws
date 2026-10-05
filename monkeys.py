# flake8: noqa: E501
#
#!/usr/bin/env python
"""
Run : python monkeys.py --help

Key commands [CMD] :
 run : deploy pipelines locally
 remote : deploy pipelines on cluster
 export-to-json : Export mongo collections to json

 python monkeys.py [CMD] --help for more information about CMD
"""
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

import json
import logging
import os
import pathlib
import subprocess
import sys
from collections import defaultdict
from exceptions import (
    CVFoldError,
    ExportError,
    ExportFileError,
    IdLinkError,
    MetricsError,
    ParsingError,
    PipelineError,
)
from logging.handlers import RotatingFileHandler
from pathlib import Path

import click
import metadata
import numpy as np
import pandas as pd
from data import read
from pythonjsonlogger import jsonlogger
from run import myrun, parse_job_toml, parse_server_toml, remote_deploy, timing
from utils import conf_get


def setup_root_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler("./monkeys.log", maxBytes=20 * 1024)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    format_str = "%(message)%(levelname)%(name)%(asctime)"
    formatter = jsonlogger.JsonFormatter(format_str)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


@click.group()
def cli1():
    pass


@cli1.command()
@click.argument("jobfile", type=click.Path(exists=True))
@click.option("-e", "--exp_id", "exp_id")
def run(jobfile, exp_id):
    """Runs pipelines defined in job file on local machine

    Args:

        jobfile (str): job file path

        exp_id (str): A string id for the run. Defaults to the id in job file.
    """
    with timing.MeasureBlockTime("job"):
        conf = parse_job_toml(jobfile)
        if exp_id is None:
            exp_id = conf_get(conf, "summary.name")
        print("exp_id = ", exp_id)
        myrun(conf, exp_id)
    # export time report to csv
    pd.DataFrame(timing.generate_dict_report()).to_csv(f"{exp_id}.csv")


@cli1.command()
@click.argument("dirpath")
@click.argument("dbname")
def export_to_json(dirpath, dbname):
    """Exports pipeline results from mongo to json files

    Function exports three collections from the given mongodb database.
    Each collection will be exported to a json file.

    1. results.json : Exports dataid, scaler,index, outcome, feature_selection,
       model, features, scores, inner fold feature selection info, ystats from the results collection.

    2. cv.json : Exports cvid, confid, outcome from meta collection

    3. jobdesc.json : Exports the job file

    4. metalearning.json: Exports metalearning feature data to json file.

    5. drops.json: Exports skipped outcomes and outcome-folds

    Args:

        dirpath (str): Directory to store the json files. Creates if it doesn't exist.

        dbname (str): Mongodb database name.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    # use mongoexport commands to export collections into json files
    # 1. Results collection
    root = Path(dirpath)
    res_path = os.path.join(root, "results.json")
    cv_path = os.path.join(root, "cv.json")
    job_path = os.path.join(root, "jobdesc.json")
    metalearning_path = os.path.join(root, "metalearning.json")
    drops_path = os.path.join(root, "drops.json")
    matrix_shapes_path = os.path.join(root, "matrix_shapes.json")
    timing_path = os.path.join(root, "timing.json")
    res_cmd = (
        'mongoexport -d %s -c results --out=%s -f "dataid,scaler,index,outcome,feature_selection,model,features,scores,ystats,predictions,predictions_raw,hyperparameter_tuning,hyperparameter_tuning_fs_info,hyperparameter_tuning_cvfolds,failed,error,stack_trace,is_tuned,pipeline_model_time,model_train_stats.model_test_stats,metafeatureid,feature_selection_stats,resampling"'
        % (dbname, res_path)
    )
    # 2. CV data
    cv_cmd = 'mongoexport -d %s -c meta --out=%s -f "cvid,confid,outcome,ystats,train_outcomes,test_outcomes"' % (
        dbname,
        cv_path,
    )
    # 3. job description
    conf_cmd = "mongoexport -d %s -c conf --out=%s" % (dbname, job_path)
    # 4. metalearning
    meta_cmd = f"mongoexport -d {dbname} -c metalearning --out={metalearning_path}"
    # 5. Drops
    drops_cmd = f"mongoexport -d {dbname} -c drops  --out={drops_path}"
    # 6. Matrix shapes
    matrix_shapes_cmd = (
        f"mongoexport -d {dbname} -c matrix_shapes  --out={matrix_shapes_path}"
    )
    # 7. Timing info
    timing_cmd = f"mongoexport -d {dbname} -c timing --out={timing_path}"
    # execute commands
    for cmd in [
        res_cmd,
        cv_cmd,
        conf_cmd,
        meta_cmd,
        drops_cmd,
        matrix_shapes_cmd,
        timing_cmd,
    ]:
        print(cmd)
        subprocess.run(cmd.split(), check=True)


@cli1.command()
@click.argument("dirpath")
def verify_json_files(dirpath):
    """Run post-conditions checks on the json files"""
    # 1. Ensure the directory has all three files
    root = Path(dirpath)
    res_path, cv_path, job_path, errors = [
        os.path.join(root, fname)
        for fname in ["results.json", "cv.json", "jobdesc.json", "errors.txt"]
    ]

    def check_all_eq(items):
        arr = np.array(items)
        return np.all(arr == arr[0])

    def is_file_empty(file_path):
        """Check if file is empty by confirming if its size is 0 bytes"""
        # Check if file exist and it is empty
        return os.path.exists(file_path) and os.stat(file_path).st_size == 0

    def process_error(file_handler, msg, errorclass):
        file_handler.write(msg)
        raise errorclass(msg)

    with open(errors, "w") as w:
        for path in [res_path, cv_path, job_path]:
            if not os.path.exists(path):
                process_error(w, "%s doesn't exist\n" % path, ExportFileError)
        # All three files exist
        try:
            jobdesc = json.loads(open(job_path).read())
            jobid = jobdesc["_id"]["$oid"]
        except ValueError:
            process_error(w, "Can't parse job desc file %s" % job_path, ParsingError)
        # read cv data
        cvids = defaultdict(list)
        with open(cv_path) as cvfile:
            for cvline in cvfile:
                cv = json.loads(cvline)
                cv_job_id = cv["confid"]["$oid"]
                if cv_job_id != jobid:
                    process_error(
                        w, "cv jobid doesn't exists %s \n" % cv_job_id, IdLinkError
                    )
                cvids[cv["outcome"]].append(cv["_id"]["$oid"])
            # check equal number of folds for all outcomes
            num_folds = np.array([len(v) for v in cvids.values()])
            if not check_all_eq(num_folds):
                process_error(
                    w, "Unequal number of folds %s\n" % str(num_folds), CVFoldError
                )
            # now iterate over the results collection
            num_results = defaultdict(int)
            with open(res_path) as resfile:
                for resline in resfile:
                    res = json.loads(resline)
                    if res["outcome"] not in cvids:
                        process_error(w, "Missing cv fold %s\n" % resline, IdLinkError)
                    if res["dataid"] not in cvids[res["outcome"]]:
                        process_error(w, "Missing dataid %s\n" % resline, IdLinkError)
                    num_results[res["outcome"]] += 1
                    if 'r2_score' in res['scores'] and res["scores"]["r2_score"] > 1:
                        process_error(w, "R^2 score > 1 in %s" % str(res), MetricsError)
            # check if number of pipelines is a multiple of number of folds
            num_pipeline_per_outcome = list(num_results.values())[0]
            if list(num_results.values())[0] % num_folds[0] != 0:
                process_error(
                    w,
                    "Number of pipelines %d not a multiple of num folds %d \n"
                    % (num_pipeline_per_outcome, num_folds[0]),
                    PipelineError,
                )
    if not is_file_empty(errors):
        raise ExportError("Error in exported files. See %s for details" % errors)


def convert_to_str(df, cols):
    for col in cols:
        df[col] = df[col].apply(str)


@cli1.command()
@click.argument("jobfile", type=click.Path(exists=True))
@click.argument("servers", type=click.Path(exists=True))
def remote(jobfile, servers):
    """Deploys execution of pipelines across servers

    Distributes the outcomes defined in job file across the servers equally.
    To achieve this distributed execution, a new job file is created (from the
    input) for each machine. This "sub" job file contains a subset of outcomes
    listed in the original job description file. The function copies the newly
    created job file and runner scripts to the correspoding worker machine.
    Finally, the runner script is started in a tmux session. See
    run/exec_remote.py file for more details.

    Args:

        jobfile (str): Path to job description file

        servers (str): File containing list of worker machines and their ip addresses
    """
    servers = parse_server_toml(servers)
    job = parse_job_toml(jobfile)
    path = pathlib.Path(__file__).parent.absolute().joinpath("data")
    metadata.load(path.joinpath(conf_get(job, "data.files.metadata")[0]))
    meta = metadata.get_snapshot()
    depvars, _ = read(
        Path(os.path.join(Path("./data"), job["data"]["files"]["outcomes"][0]))
    )
    exp_name = job["summary"]["name"]
    remote_deploy(depvars, meta, servers, job, "./experiments", exp_name)


cli = click.CommandCollection(sources=[cli1])

if __name__ == "__main__":
    setup_root_logger()
    cli()
