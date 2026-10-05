"""feature selection using zassociation scores
"""
import sys
from collections import Counter

import numpy as np
from sklearn.feature_selection import GenericUnivariateSelect


def zassoc(feat, labs, ldist):
    cntr = Counter(zip(feat, labs))
    sub = 1.0
    tot = len(feat)
    itms = Counter(feat).items()
    if len(itms) > 10:
        return 0
    for j, cnt in itms:
        pj = cnt / float(tot)
        prod = 1.0
        for i, icnt in ldist.items():
            num = float(cnt - cntr[(j, i)]) / float(cnt)
            den = float(tot - icnt) / float(tot)
            prod *= float(num) / float(den)
        sub -= pj * prod
    return sub


def zassoc_score_function(X, y):
    vectorizer = np.vectorize(lambda x: "H" if x >= 3.25 else ("M" if x > 2.5 else "L"))
    train_labs = vectorizer(y)
    ldist = Counter(train_labs)
    scores = []
    for colData in X.T:
        z = zassoc(colData, train_labs, ldist)
        scores.append(z)
    return np.array(scores)


def zassoc_fs(params=[10]):
    for param in params:
        name = "%s-%s-%s" % ("zassociation", "k_best", str(param))
        yield name, GenericUnivariateSelect(
            zassoc_score_function, mode="k_best", param=param
        )


if __name__ == "__main__":
    xname, yname = sys.argv[1:]
    X = np.load(xname)
    y = np.load(yname)
    scores = zassoc_score_function(X, y)
