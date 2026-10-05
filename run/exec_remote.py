# execute tasks on remote serverimport yaml
import copy
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import toml
from fabric import Connection
from utils import conf_get, get_outcomes, zip_list


def exec_server_job(cxn, cmd, fname):
    with open(fname, "w") as w:
        remote_cmd = 'tmux new-session -d -s "million-monkeys" "%s"' % cmd
        result = cxn.run(remote_cmd, pty=False, hide=True)
        print(result)
        dct = {"host": cxn.host, "user": cxn.user, "cmd": cmd}
        w.write(json.dumps(dct))


def deploy(cxn, local_path, remote_path):
    cxn.put(local_path, remote_path)


def assign_tasks(rawdata, meta, servers, job):
    """Get regression and classification outcomes and assign them to servers in roundrobin"""
    outcome_names = get_outcomes(rawdata, meta, job)
    if conf_get(job, "data.preprocessing.outcomes.add_scales_to_outcomes"):
        df = pd.read_csv(
            os.path.join(
                Path("./data"), Path(conf_get(job, "data.files.scales_instructions")[0])
            )
        )
        df = df[~df.metadata_new_name.str.contains("intermediate")]
        scale_names = df.metadata_new_name.tolist()
        # column that corresponds to variable name
        outcome_names.extend(scale_names)
    # remove any duplicate varnames; using Counter to maintain item order
    outcome_names = [k for k, v in Counter(outcome_names).items() if v >= 1]
    waves_to_use = conf_get(job, "data.preprocessing.outcomes.waves_to_use")
    vars_not_in_waves = meta.loc[~meta.wave.isin(waves_to_use)].index
    # sample outcomes
    # outcome_names = random.sample(
    # outcome_names,
    # math.ceil(
    # len(outcome_names)
    # * conf_get(job, "execution.remote.sample.outcome_fraction")
    # ),
    # )
    outcome_names = pd.Index(outcome_names).difference(vars_not_in_waves)
    return zip(zip_list(servers, outcome_names), [True] * len(outcome_names))


def remote_deploy(rawdata, metadata, servers, job, rootdir, prefix="million-monkeys"):
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    dirname = tempfile.mkdtemp(
        dir=rootdir, prefix="{}{:-%Y-%m-%dT%H-%M-%S}-".format(prefix, datetime.now())
    )
    print(dirname)
    exp_id = Path(dirname).name
    tasks = assign_tasks(
        rawdata, metadata, [server["name"] for server in servers["server"]], job
    )
    server_outcomes = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (server_key, outcome), is_regression in tasks:
        server_outcomes[server_key][
            "regression_outcomes" if is_regression else "classification_outcomes"
        ]["outcome_names"].append(outcome)
        # empty topic names in outcomes section
        server_outcomes[server_key]["regression_outcomes"]["topics"] = ["all"]
    # schema requires both regression and classification
    for server_info in servers["server"]:
        server_key = server_info["name"]
        # for outcome_class in ["regression_outcomes", "classification_outcomes"]:
        for outcome_class in ["regression_outcomes"]:
            if outcome_class not in server_outcomes[server_key]:
                server_outcomes[server_key][outcome_class]["outcome_names"] = []
    script_path = os.path.abspath(os.path.expanduser("./scripts/run.sh"))
    for server_info in servers["server"]:
        server_key = server_info["name"]
        server_job = copy.deepcopy(job)
        server_job["data"]["outcomes"] = dict(server_outcomes[server_key])
        connection = Connection(
            host="%s@%s" % (server_info["user"], server_info["host"]),
            connect_kwargs={
                "password": server_info["password"] if "password" in server_info else ""
            },
        )
        expdir = os.path.abspath(os.path.join("/datadrive", dirname))
        server_conf_path = os.path.join(expdir, "%s_conf.toml" % server_key)
        local_conf_path = os.path.join(dirname, "%s_conf.toml" % server_key)
        with open(local_conf_path, "w") as outfile:
            toml.dump(server_job, outfile)
        connection.run("mkdir -p %s && touch %s" % (expdir, server_conf_path))
        deploy(connection, script_path, os.path.basename(script_path))
        deploy(connection, local_conf_path, server_conf_path)
        remote_cmd = "./run.sh %s %s %s" % (
            server_conf_path,
            exp_id,
            conf_get(job, "results.database_name"),
        )
        print(remote_cmd)
        exec_server_job(
            connection, remote_cmd, os.path.join(dirname, "%s.json" % server_key)
        )
