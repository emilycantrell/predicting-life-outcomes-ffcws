import json
import pickle
from pymongo import MongoClient
from utils import track_shapes as Tracker

try:
    from secrets import host, password, port, username
except ImportError:
    pass


def get_remote_client(db_name):
    args = "ssl=true&ssl_cert_reqs=CERT_NONE&retrywrites=false"
    connection_uri = f"mongodb://{username}:{password}@{host}:{port}/{db_name}?{args}"
    client = MongoClient(connection_uri)
    return client


def get_binary_meta(data):
    return {
        "train_outcomes": data.ytrain.to_json(),
        "test_outcomes": data.ytest.to_json(),
    }


def divide_chunks(li, n):
    for i in range(0, len(li), n):
        yield li[i : i + n]


class ResultsDB:
    def __init__(self, remote=False, database="test-database"):
        if not remote:
            self.client = MongoClient("mongodb://localhost:27017/")
        else:
            # connect to mongo instance
            self.client = get_remote_client("test-database")
        self.dbname = database
        self.db = self.client[database]
        # create collections
        collections = [
            "results",
            "failures",
            "meta",
            "conf",
            "metalearning",
            "drops",
            "matrix_shapes",
            "timing",
        ]
        for col in collections:
            setattr(self, col, self.db[col])
        super().__init__()

    def insert_drops(self, vartype, key, variables, msg):
        self.drops.insert_one(
            {"vartype": vartype, "dropstype": key, "variables": variables, "msg": msg}
        )

    def delete_database(self):
        self.client.drop_database(self.dbname)

    def insert_metalearning(self, metafeature):
        res = self.metalearning.insert_one(metafeature)
        return str(res.inserted_id)

    def insert_successes(self, successes, delete=False):
        # if no results
        if not successes:
            return
        if delete:
            self.results.drop()
        resultids = []
        for chunk in divide_chunks(successes, 50):
            resobj = self.results.insert_many(
                chunk, ordered=False, bypass_document_validation=True
            )
            resultids.extend([str(resid) for resid in resobj.inserted_ids])
        return resultids

    def insert_failures(self, failures, delete=False):
        # if no failures
        if not failures:
            return
        if delete:
            self.results.drop()
        self.failures.insert_many(failures)

    def insert_results(self, successes, failures, delete=False):
        self.insert_failures(failures, delete)
        self.insert_successes(successes, delete)

    def insert_timing(self, timing_dct):
        self.timing.insert_one(timing_dct)

    def insert_cv(self, cvid, outcome, data, ystats, conf_id=None, delete=False):
        """insert cross validation metadata to database

        Arguments:
            cvid {int} -- split index
            data {Data} -- tuple containing training and testing data
        """
        if delete:
            self.meta.drop()
        insertresult = self.meta.insert_one(
            {
                "cvid": cvid,
                "confid": conf_id,
                "outcome": outcome,
                "ystats": ystats,
                **get_binary_meta(data),
            }
        )
        return insertresult.inserted_id

    def insert_matrix_shape(self):
        shape_dct = json.loads(Tracker.to_json())
        self.matrix_shapes.insert_one(shape_dct)

    def insert_exp_conf(self, conf):
        conf = self.conf.insert_one(conf)
        return conf.inserted_id
