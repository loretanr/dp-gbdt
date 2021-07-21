# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Evaluation of our DPGBDT implementation."""

import json
import numpy as np
from datetime import datetime


from sklearn.model_selection import cross_val_score

from data.parser.parser import Parser
from evaluation import estimator

# The dataset to use for evaluation
DATASET = 'adult'
# The privacy budget to use for evaluation
PRIVACY_BUDGETS = [0.5]
# The number of time to repeat the experiment to get an average accuracy
NB_SPLITS = 5
# Number of rows to use from the dataset
SAMPLES = [300]
# Nb trees for each ensemble
NB_TREES_PER_ENSEMBLE = 5

PATH = "./results/adult/"

# The optimal parameters found through the grid search for the baseline model
with open(PATH + 'model_params.json') as json_file:
  MODEL_PARAMS = json.load(json_file)

if __name__ == '__main__':

  for nb_samples in SAMPLES:
    # Read the data
    parser = Parser(dataset=DATASET)
    X, y, cat_idx, num_idx, task = parser.Parse(n_rows=nb_samples)

    # Own model
    models = [estimator.DPGBDT]

    for model in models:
      model_name = str(model).split('.')[-1][:-2]
      print('------------ Processing model: {0:s}'.format(model_name))
      for config in ['DFS']:
        for idx, budget in enumerate(PRIVACY_BUDGETS):
          if config == 'Vanilla' and idx != 0:
            continue
          if model_name == 'DPRef' and config != 'DFS':
            continue
          model_params = MODEL_PARAMS.get(model_name).get(config)
          budget = np.around(np.float(budget), decimals=2)
          if config == 'Vanilla':
            budget = 0.
          nb_trees = NB_TREES_PER_ENSEMBLE
          min_samples_split = model_params.get(
              'min_samples_split', 2) if nb_samples == max(SAMPLES) else int(
                  model_params.get('min_samples_split', 20) / 10)
          m = model(
              budget,
              nb_trees,
              NB_TREES_PER_ENSEMBLE,
              model_params.get('max_depth'),
              model_params.get('learning_rate'),
              n_classes=len(set(y)) if task == 'classification' else None,
              gradient_filtering=True,
              leaf_clipping=True,
              max_leaves=model_params.get('max_leaves'),
              min_samples_split=min_samples_split,
              balance_partition=model_params.get('balance_partition'),
              use_bfs=model_params.get('use_bfs', False),
              use_3_trees=model_params.get('use_3_trees', False),
              cat_idx=cat_idx,
              num_idx=num_idx,
              verbosity=1)  # type: ignore
          scores = cross_val_score(
              m, X, y, scoring='accuracy', n_jobs=1)
          print(scores)
          mean, std = 100 - (scores.mean() * 100), (scores.std() * 100 / 2)
          print("====== PB {} SCORE {} ======".format(budget, mean))

