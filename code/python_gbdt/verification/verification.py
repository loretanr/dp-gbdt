# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Verification of our DPGBDT implementation."""

import numpy as np

from DPGBDT import logging
import DPGBDT.model

from sklearn import metrics, model_selection
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from data.parser.parser import Parser
from evaluation import estimator

PRIVACY_BUDGET = 0.1
NB_SPLITS = 5 # number of CV folds
SAMPLES = [] # [300, 5000] # number of rows to use from the dataset
NB_TREES_PER_ENSEMBLE = 50
NB_TREES = 50
MIN_SAMPLES_SPLIT = 2
LEARNING_RATE = 0.1
MAX_DEPTH = 6


if __name__ == '__main__':

    DATASET = 'abalone'
    parser = Parser(dataset=DATASET)

    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = DATASET + "_small" if num_samples == 300 else DATASET + "_full"
        print(dataset_name)
        logging.SetupVerificationLogger(dataset_name)
        X, y, cat_idx, num_idx, task = parser.Parse(n_rows=num_samples)
        rmse = make_scorer(metrics.mean_squared_error, squared=False)

        m = estimator.DPGBDT(  # type: ignore
            PRIVACY_BUDGET, NB_TREES, NB_TREES_PER_ENSEMBLE, MAX_DEPTH, LEARNING_RATE,
            n_classes=len(set(y)) if task == 'classification' else None,
            gradient_filtering=True,
            leaf_clipping=True,
            min_samples_split=MIN_SAMPLES_SPLIT,
            balance_partition=True, use_bfs=False, use_3_trees=False,
            cat_idx=cat_idx, num_idx=num_idx,
            verbosity=-1)
        regressor = TransformedTargetRegressor(regressor=m,
            transformer=MinMaxScaler(feature_range=(-1, 1)))
        validator = model_selection.KFold(n_splits=NB_SPLITS, shuffle=False)
        scores = cross_val_score(
            regressor, X, y, cv=validator, scoring=rmse, n_jobs=1) # -1 for multithreading
        mean, std = scores.mean(), (scores.std() / 2)
        print(scores)
        logging.CloseVerificationLogger()

    DATASET = 'yearMSD'
    parser = Parser(dataset='year')
    SAMPLES = [300]


    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = DATASET + "_small" if num_samples == 300 else DATASET + "_full"
        print(dataset_name)
        logging.SetupVerificationLogger(dataset_name)
        X, y, cat_idx, num_idx, task = parser.Parse(n_rows=num_samples)
        rmse = make_scorer(metrics.mean_squared_error, squared=False)

        m = estimator.DPGBDT(  # type: ignore
            PRIVACY_BUDGET, NB_TREES, NB_TREES_PER_ENSEMBLE, MAX_DEPTH, LEARNING_RATE,
            n_classes=len(set(y)) if task == 'classification' else None,
            gradient_filtering=True,
            leaf_clipping=True,
            min_samples_split=MIN_SAMPLES_SPLIT,
            balance_partition=True, use_bfs=False, use_3_trees=False,
            cat_idx=cat_idx, num_idx=num_idx,
            verbosity=-1)
        regressor = TransformedTargetRegressor(regressor=m,
            transformer=MinMaxScaler(feature_range=(-1, 1)))
        validator = model_selection.KFold(n_splits=NB_SPLITS, shuffle=False)
        scores = cross_val_score(
            regressor, X, y, cv=validator, scoring=rmse, n_jobs=1) # -1 for multithreading
        mean, std = scores.mean(), (scores.std() / 2)
        print(scores)
        logging.CloseVerificationLogger()
