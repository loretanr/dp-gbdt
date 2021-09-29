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

PRIVACY_BUDGET = 0.5
NB_SPLITS = 5 # number of CV folds
NB_TREES_PER_ENSEMBLE = 5
NB_TREES = 5
MIN_SAMPLES_SPLIT = 2
LEARNING_RATE = 0.1
MAX_DEPTH = 6
GRADIENT_FILTERING = False
LEAF_CLIPPING = True
BALANCE_PARTITION = True

def get_dataset_name(num_samples):
    return DATASET + "_size_" + str(num_samples)



if __name__ == '__main__':

    DPGBDT.model.VERIFICATION_MODE = True

    DATASET = 'abalone'
    parser = Parser(dataset=DATASET)
    SAMPLES = [300]
    SAMPLES = []

    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = get_dataset_name(num_samples)
        print(dataset_name)
        logging.SetupVerificationLogger(dataset_name)
        X, y, cat_idx, num_idx, task = parser.Parse(n_rows=num_samples)
        rmse = make_scorer(metrics.mean_squared_error, squared=False)

        m = estimator.DPGBDT(  # type: ignore
            PRIVACY_BUDGET, NB_TREES, NB_TREES_PER_ENSEMBLE, MAX_DEPTH, LEARNING_RATE,
            n_classes=len(set(y)) if task == 'classification' else None,
            gradient_filtering=GRADIENT_FILTERING,
            leaf_clipping=LEAF_CLIPPING,
            min_samples_split=MIN_SAMPLES_SPLIT,
            balance_partition=BALANCE_PARTITION, 
            use_bfs=False, 
            use_3_trees=False,
            cat_idx=cat_idx, num_idx=num_idx,
            verbosity=-1)
        regressor = TransformedTargetRegressor(regressor=m)
        validator = model_selection.KFold(n_splits=NB_SPLITS, shuffle=False)
        scores = cross_val_score(
            regressor, X, y, cv=validator, scoring=rmse, n_jobs=1) # -1 for multithreading
        mean, std = scores.mean(), (scores.std() / 2)
        print(scores)
        logging.CloseVerificationLogger()


    DATASET = 'yearMSD'
    parser = Parser(dataset=DATASET)
    SAMPLES = [150]
    SAMPLES = []

    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = get_dataset_name(num_samples)
        print(dataset_name)
        logging.SetupVerificationLogger(dataset_name)
        X, y, cat_idx, num_idx, task = parser.Parse(n_rows=num_samples)
        rmse = make_scorer(metrics.mean_squared_error, squared=False)

        m = estimator.DPGBDT(  # type: ignore
            PRIVACY_BUDGET, NB_TREES, NB_TREES_PER_ENSEMBLE, MAX_DEPTH, LEARNING_RATE,
            n_classes=len(set(y)) if task == 'classification' else None,
            gradient_filtering=GRADIENT_FILTERING,
            leaf_clipping=LEAF_CLIPPING,
            min_samples_split=MIN_SAMPLES_SPLIT,
            balance_partition=BALANCE_PARTITION,
            use_bfs=False,
            use_3_trees=False,
            cat_idx=cat_idx, num_idx=num_idx,
            verbosity=-1)
        validator = model_selection.KFold(n_splits=NB_SPLITS, shuffle=False)
        scores = cross_val_score(
            m, X, y, cv=validator, scoring=rmse, n_jobs=1) # -1 for multithreading
        mean, std = scores.mean(), (scores.std() / 2)
        print(scores)
        logging.CloseVerificationLogger()


    DATASET = 'adult'
    parser = Parser(dataset=DATASET)
    # SAMPLES = [300,1000]
    SAMPLES = [320]
    # SAMPLES = []

    for num_samples in SAMPLES:
        DPGBDT.model.cv_fold_counter = 0
        dataset_name = get_dataset_name(num_samples)
        print(dataset_name)
        logging.SetupVerificationLogger(dataset_name)
        X, y, cat_idx, num_idx, task = parser.Parse(n_rows=num_samples)
        rmse = make_scorer(metrics.mean_squared_error, squared=False)

        m = estimator.DPGBDT(  # type: ignore
            PRIVACY_BUDGET, NB_TREES, NB_TREES_PER_ENSEMBLE, MAX_DEPTH, LEARNING_RATE,
            n_classes=len(set(y)) if task == 'classification' else None,
            gradient_filtering=GRADIENT_FILTERING,
            leaf_clipping=LEAF_CLIPPING,
            min_samples_split=MIN_SAMPLES_SPLIT,
            balance_partition=BALANCE_PARTITION,
            use_bfs=False,
            use_3_trees=False,
            cat_idx=cat_idx, num_idx=num_idx,
            binary_classification=True,
            verbosity=-1)
        scores = cross_val_score(
            m, X, y, scoring='accuracy', n_jobs=1) # -1 for multithreading
        mean, std = 100 - (scores.mean() * 100), (scores.std() * 100 / 2)
        print(scores)
        logging.CloseVerificationLogger()
