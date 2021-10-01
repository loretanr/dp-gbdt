# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Evaluation of our DPGBDT implementation."""

import json
import numpy as np

from sklearn.model_selection import cross_val_score

from data.parser.parser import Parser
from evaluation import estimator

# The dataset to use for evaluation
DATASET = 'bcw'
# The privacy budget to use for evaluation
PRIVACY_BUDGETS = [10]
# The number of time to repeat the experiment to get an average accuracy
NB_SPLITS = 5
# Number of rows to use from the dataset
SAMPLES = [700]
# Nb trees for each ensemble
NB_TREES_PER_ENSEMBLE = 5

if __name__ == '__main__':

  for nb_samples in SAMPLES:
    # Read the data
    parser = Parser(dataset=DATASET)
    X, y, cat_idx, num_idx, task = parser.Parse(n_rows=nb_samples)
    print("bcw, zeroR 0.6501")


    # Own model
    models = [estimator.DPGBDT] # seems not to work with DPRef, great.

    for model in models:
      model_name = str(model).split('.')[-1][:-2]
      print('------------ Processing model: {0:s}'.format(model_name))
      for config in ['Vanilla','DFS']:
        for idx, budget in enumerate(PRIVACY_BUDGETS):
          if config == 'Vanilla' and idx != 0:
            continue
          if model_name == 'DPRef' and config != 'DFS':
            continue
          budget = np.around(np.float64(budget), decimals=2)
          if config == 'Vanilla':
            budget = 0.
          m = model(
              privacy_budget=10,
              nb_trees=30,
              nb_trees_per_ensemble=30,
              max_depth=6,
              learning_rate=0.1,
              n_classes=len(set(y)) if task == 'classification' else None,
              min_samples_split=2,
              balance_partition=True,
              gradient_filtering=False,
              leaf_clipping=True,
              use_bfs=False,
              use_3_trees=False,
              cat_idx=cat_idx,
              num_idx=num_idx,
              verbosity=-1)  # type: ignore
          scores = cross_val_score(
              m, X, y, scoring='accuracy', n_jobs=1)
          # mean, std = 100 - (scores.mean() * 100), (scores.std() * 100 / 2)
          print(config, scores, "-> mean {:.4f}".format(scores.mean()))
