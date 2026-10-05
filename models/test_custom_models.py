#!/usr/bin/env python3

from sklearn.datasets import load_iris

from .custom_models import LogisticRegressorWithProb


def test_logistic_with_prob():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegressorWithProb(random_state=0).fit(X, y)
    predictions = clf.predict(X[:2, :])
    assert (predictions >= 0).all()
    assert (predictions <= 1).all()
