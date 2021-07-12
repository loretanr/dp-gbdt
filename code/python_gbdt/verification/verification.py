# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Verification of our DPGBDT implementation."""

import numpy as np

from sklearn import metrics, model_selection
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from data.parser.parser import Parser
from evaluation import estimator

DATASET = 'abalone'
PRIVACY_BUDGET = 0.1
# The number of CV folds
NB_SPLITS = 5
# Number of rows to use from the dataset
SAMPLES = 5000
NB_TREES_PER_ENSEMBLE = 50
NB_TREES = 50
MIN_SAMPLES_SPLIT = 2
LEARNING_RATE = 0.1
MAX_DEPTH = 6


if __name__ == '__main__':
    print("starting python verification")

    # Read the data
    parser = Parser(dataset=DATASET)
    X, y, cat_idx, num_idx, task = parser.Parse(n_rows=SAMPLES)

    rmse = make_scorer(metrics.mean_squared_error, squared=False)

    m = estimator.DPGBDT(  # type: ignore
        PRIVACY_BUDGET,
        NB_TREES,
        NB_TREES_PER_ENSEMBLE,
        MAX_DEPTH,
        LEARNING_RATE,
        n_classes=len(set(y)) if task == 'classification' else None,
        gradient_filtering=True,
        leaf_clipping=False,   # TODO implement cpp, false for now     truuuuuuuue ?????
        # max_leaves=model_params.get('max_leaves'),
        min_samples_split=MIN_SAMPLES_SPLIT,
        balance_partition=True,
        use_bfs=False,
        use_3_trees=False,
        cat_idx=cat_idx,
        num_idx=num_idx,
        verbosity=-1)  # type: ignore
    regressor = TransformedTargetRegressor(        # regressor = "all names of the variables 
        regressor=m,# transformer=RobustScaler())#,                               # that are used to predict the target"
        transformer=MinMaxScaler(feature_range=(-1, 1)))     # just to scale the features.
    validator = model_selection.KFold(n_splits=NB_SPLITS, shuffle=False)
    scores = cross_val_score(
        regressor, X, y, cv=validator, scoring=rmse, n_jobs=1) # was -1 for multithreading

    mean, std = scores.mean(), (scores.std() / 2)

    print(scores)
    print("====== PB {} SCORE {} ======".format(PRIVACY_BUDGET, mean))