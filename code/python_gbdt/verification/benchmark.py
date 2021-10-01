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


PRIVACY_BUDGET = 0.5
NB_SPLITS = 5 # number of CV folds
NB_TREES_PER_ENSEMBLE = 10
NB_TREES = 10
MIN_SAMPLES_SPLIT = 2
LEARNING_RATE = 0.1
MAX_DEPTH = 6
GRADIENT_FILTERING = False
LEAF_CLIPPING = True
BALANCE_PARTITION = True


if __name__ == '__main__':
    DATASET = 'abalone'
    parser = Parser(dataset=DATASET)
    SAMPLES = [300,1000,4177]
    SAMPLES = []

    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = DATASET + "_" + str(num_samples)
        print(dataset_name)
        X, y, cat_idx, num_idx, task = parser.Parse(n_rows=num_samples)
        rmse = make_scorer(metrics.mean_squared_error, squared=False)

        m = estimator.DPGBDT(  # type: ignore
            PRIVACY_BUDGET, NB_TREES, NB_TREES_PER_ENSEMBLE, MAX_DEPTH, LEARNING_RATE,
            n_classes=len(set(y)) if task == 'classification' else None,
            gradient_filtering=GRADIENT_FILTERING,
            leaf_clipping=LEAF_CLIPPING,
            min_samples_split=MIN_SAMPLES_SPLIT,
            balance_partition=BALANCE_PARTITION, use_bfs=False, use_3_trees=False,
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
    SAMPLES = [300,1000]
    SAMPLES = []

    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = DATASET + "_" + str(num_samples)
        print(dataset_name)
        X, y, cat_idx, num_idx, task = parser.Parse(n_rows=num_samples)
        rmse = make_scorer(metrics.mean_squared_error, squared=False)

        m = estimator.DPGBDT(  # type: ignore
            PRIVACY_BUDGET, NB_TREES, NB_TREES_PER_ENSEMBLE, MAX_DEPTH, LEARNING_RATE,
            n_classes=len(set(y)) if task == 'classification' else None,
            gradient_filtering=GRADIENT_FILTERING,
            leaf_clipping=LEAF_CLIPPING,
            min_samples_split=MIN_SAMPLES_SPLIT,
            balance_partition=BALANCE_PARTITION, use_bfs=False, use_3_trees=False,
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
    SAMPLES = [300,1000,5000]

    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = DATASET + "_" + str(num_samples)
        print(dataset_name)
        X, y, cat_idx, num_idx, task = parser.Parse(n_rows=num_samples)
        rmse = make_scorer(metrics.mean_squared_error, squared=False)

        m = estimator.DPGBDT(  # type: ignore
            PRIVACY_BUDGET, NB_TREES, NB_TREES_PER_ENSEMBLE, MAX_DEPTH, LEARNING_RATE,
            n_classes=len(set(y)) if task == 'classification' else None,
            gradient_filtering=GRADIENT_FILTERING,
            leaf_clipping=LEAF_CLIPPING,
            min_samples_split=MIN_SAMPLES_SPLIT,
            balance_partition=BALANCE_PARTITION, use_bfs=False, use_3_trees=False,
            cat_idx=cat_idx, num_idx=num_idx,
            verbosity=-1)
        validator = model_selection.KFold(n_splits=NB_SPLITS)
        start = time.time()
        scores = cross_val_score(
            m, X, y, cv=validator, scoring='accuracy', n_jobs=-1) # -1 for multithreading
        stop = time.time()
        print(str(scores) + "   ({:.1f}s)".format(stop-start))        
