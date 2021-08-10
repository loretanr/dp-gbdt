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

from data.parser.parser import Parser
from evaluation import estimator
import time


PRIVACY_BUDGET = 0.1
NB_SPLITS = 5 # number of CV folds
NB_TREES_PER_ENSEMBLE = 50
NB_TREES = 50
MIN_SAMPLES_SPLIT = 2
LEARNING_RATE = 0.1
MAX_DEPTH = 6


if __name__ == '__main__':
    DATASET = 'abalone'
    parser = Parser(dataset=DATASET)
    SAMPLES = [5000]

    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = DATASET + "_small" if num_samples == 300 else DATASET + "_full"
        print(dataset_name)
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
        validator = model_selection.KFold(n_splits=NB_SPLITS)
        start = time.time()
        scores = cross_val_score(
            m, X, y, cv=validator, scoring=rmse, n_jobs=-1) # -1 for multithreading
        stop = time.time()
        print(str(scores) + "   ({:.1f}s)".format(stop-start))


    DATASET = 'yearMSD'
    parser = Parser(dataset=DATASET)
    SAMPLES = [10000]

    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = DATASET + "_small" if num_samples == 300 else DATASET + "_custom_size_" + str(num_samples)
        print(dataset_name)
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
        validator = model_selection.KFold(n_splits=NB_SPLITS)
        start = time.time()
        scores = cross_val_score(
            m, X, y, cv=validator, scoring=rmse, n_jobs=-1) # -1 for multithreading
        stop = time.time()
        print(str(scores) + "   ({:.1f}s)".format(stop-start))


    DATASET = 'adult'
    parser = Parser(dataset=DATASET)
    SAMPLES = [4000]

    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = DATASET + "_small" if num_samples == 300 else DATASET + "_custom_size_" + str(num_samples)
        print(dataset_name)
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
        validator = model_selection.KFold(n_splits=NB_SPLITS)
        start = time.time()
        scores = cross_val_score(
            m, X, y, validator=validator, scoring='accuracy', n_jobs=-1) # -1 for multithreading
        stop = time.time()
        print(str(scores) + "   ({:.1f}s)".format(stop-start))        
