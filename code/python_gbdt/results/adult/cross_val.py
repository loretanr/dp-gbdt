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
PRIVACY_BUDGETS = np.arange(0.1, 1.0, 0.1)
# The number of time to repeat the experiment to get an average accuracy
NB_SPLITS = 5
# Number of rows to use from the dataset
SAMPLES = [300, 5000]
# Nb trees for each ensemble
NB_TREES_PER_ENSEMBLE = 50

PATH = "./results/adult/"

# The optimal parameters found through the grid search for the baseline model
with open(PATH + 'model_params.json') as json_file:
  MODEL_PARAMS = json.load(json_file)

if __name__ == '__main__':
  now = datetime.now().strftime("%d-%m-%y_%H:%M")
  output = open(PATH + 'data_' + now + '.csv', 'a')
  output.write('dataset,nb_samples,privacy_budget,nb_tree,nb_tree_per_ensemble,'
               'max_depth,max_leaves,learning_rate,nb_of_runs,mean,std,'
               'model,config,balance_partition\n')

  for nb_samples in SAMPLES:
    # Read the data
    parser = Parser(dataset=DATASET)
    X, y, cat_idx, num_idx, task = parser.Parse(n_rows=nb_samples)

    # Own model
    models = [estimator.DPGBDT]

    for model in models:
      model_name = str(model).split('.')[-1][:-2]
      print('------------ Processing model: {0:s}'.format(model_name))
      for config in ['Vanilla', 'BFS', 'DFS', '3-trees']:
        for idx, budget in enumerate(PRIVACY_BUDGETS):
          if config == 'Vanilla' and idx != 0:
            continue
          if model_name == 'DPRef' and config != 'DFS':
            continue
          model_params = MODEL_PARAMS.get(model_name).get(config)
          budget = np.around(np.float(budget), decimals=2)
          if config == 'Vanilla':
            budget = 0.
          nb_trees = model_params.get(
              'nb_trees') if nb_samples == max(SAMPLES) else int(
                  model_params.get('nb_trees') / 10)
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
              verbosity=-1)  # type: ignore
          scores = cross_val_score(
              m, X, y, scoring='accuracy', n_jobs=-1)
          mean, std = 100 - (scores.mean() * 100), (scores.std() * 100 / 2)
          output.write(
              '{0:s},{1:d},{2:f},{3:d},{4:d},{5:d},{6:d},{7:f},'  # type: ignore
              '{8:d},{9:f},{10:f},{11:s},{12:s}\n'.format(
                  DATASET, nb_samples, budget, nb_trees,
                  NB_TREES_PER_ENSEMBLE,
                  model_params.get('max_depth'),
                  -1, model_params.get('learning_rate'), NB_SPLITS, mean, std,
                  model_name, config,
                  str(model_params.get('balance_partition'))))
  output.close()
