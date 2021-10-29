# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Evaluation of our DPGBDT implementation."""

import json
import numpy as np
from datetime import datetime

from sklearn import metrics, model_selection
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from data.parser.parser import Parser
from evaluation import estimator

# The dataset to use for evaluation
DATASET = 'abalone'
# The privacy budget to use for evaluation
PRIVACY_BUDGETS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,4,5,6,7,8,9,10]
# PRIVACY_BUDGETS = [0.1, 0.3, 0.6, 1, 1.5, 2, 3, 5, 7, 9]
# The number of time to repeat the experiment to get an average accuracy
NB_SPLITS = 5
# Number of rows to use from the dataset
SAMPLES = [5000]
# Nb trees for each ensemble
NB_TREES_PER_ENSEMBLE = 50

PATH = "./results/abalone/"

# The optimal parameters found through the grid search for the baseline model
with open(PATH + 'model_params.json') as json_file:
  MODEL_PARAMS = json.load(json_file)

if __name__ == '__main__':
  now = datetime.now().strftime("%d-%m-%y_%H:%M")
  output = open(PATH + 'results_AFTERMORITZ_VSGRAPH' + now + '.csv', 'a')
  output.write('dataset,nb_samples,privacy_budget,nb_tree,nb_tree_per_ensemble,'
               'max_depth,max_leaves,learning_rate,nb_of_runs,mean,std,'
               'model,config,balance_partition\n')

  for nb_samples in SAMPLES:
    # Read the data
    parser = Parser(dataset=DATASET)
    X, y, cat_idx, num_idx, task = parser.Parse(n_rows=nb_samples)

    # Own model
    # models = [estimator.DPGBDT, estimator.DPRef]
    models = [estimator.DPGBDT]

    rmse = make_scorer(
        metrics.mean_squared_error, squared=False)

    for model in models:
      model_name = str(model).split('.')[-1][:-2]
      print('------------ Processing model: {0:s}'.format(model_name))
      # for config in ['Vanilla', 'DFS']:
      for config in ['DFS']:
        for idx, budget in enumerate(PRIVACY_BUDGETS):
          if config == 'Vanilla' and idx != 0:
            continue
          if model_name == 'DPRef' and config != 'DFS':
            continue
          model_params = MODEL_PARAMS.get(model_name).get(config)
          if config == 'Vanilla':
            budget = 0.
          nb_trees = 50
          m = model(  # type: ignore
              budget,
              nb_trees,
              NB_TREES_PER_ENSEMBLE,
              model_params.get('max_depth'),
              model_params.get('learning_rate'),
              n_classes=len(set(y)) if task == 'classification' else None,
              gradient_filtering=False,
              leaf_clipping=model_params.get('leaf_clipping'),
              max_leaves=model_params.get('max_leaves'),
              min_samples_split=2,
              balance_partition=True,
              use_bfs=False,
              use_3_trees=False,
              cat_idx=cat_idx,
              num_idx=num_idx,
              verbosity=-1)  # type: ignore
          regressor = TransformedTargetRegressor(regressor=m)     # REMOVED MINMAXSCALER
          validator = model_selection.KFold(n_splits=NB_SPLITS, random_state=np.random.randint(0,100000) ,shuffle=True)               # somehow shuffle=True did not shuffle!!!!
          scores = cross_val_score(                                           # leaving it away now shuffles!
              regressor, X, y, cv=validator, scoring=rmse, n_jobs=-1)
          mean, std = scores.mean(), (scores.std() / 2)
          output.write(
              '{0:s},{1:d},{2:f},{3:d},{4:d},{5:d},{6:d},{7:f},'  # type: ignore
              '{8:d},{9:f},{10:f},{11:s},{12:s},{13:s}\n'.format(
                  DATASET, min(nb_samples, len(y)), budget, nb_trees,
                  NB_TREES_PER_ENSEMBLE,
                  model_params.get('max_depth'),
                  -1, model_params.get('learning_rate'), NB_SPLITS, mean, std,
                  model_name, config,
                  str(model_params.get('balance_partition'))))
  output.close()
