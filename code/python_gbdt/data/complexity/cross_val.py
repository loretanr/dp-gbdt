# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Compute problem difficulty (i.e. error) for the baseline on synthetic
datasets.
"""

# pylint: disable=redefined-outer-name

import json
import numpy as np

from sklearn import model_selection
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from data.parser.parser import Parser

# The datasets to use for evaluation
from evaluation import estimator

DATASETS = ['synthetic_A', 'synthetic_B', 'synthetic_C', 'synthetic_D']
# The number of time to repeat the experiment to get an average accuracy
NB_SPLITS = 3
# Number of rows to use from the dataset
SAMPLES = [300, 5000, 15000, 25000, 50000, 75000, 100000]
# Label to train for
LABELS = ['loss', 'cost']
# Model parameters
with open('./model_params.json') as json_file:
  MODEL_PARAMS = json.load(json_file)


def Mape(y: np.array, y_pred: np.array) -> float:
  """Mean absolute percentage error.

  Args:
    y (np.array): The true values.
    y_pred (np.array): The predicted values.

  Returns:
    float: The mean absolute percentage error.
  """
  y[y == 0.] = 10e-6
  mape = np.mean((np.abs((y - y_pred) / y))) * 100  # type: float
  return mape


if __name__ == '__main__':
  output = open('data.csv', 'w')
  output.write('dataset,nb_samples,privacy_budget,nb_tree,nb_tree_per_ensemble,'
               'max_depth,max_leaves,learning_rate,nb_of_runs,mean,std,'
               'min_y,max_y,model,config,balance_partition,label\n')

  for dataset in DATASETS:
    print('Processing dataset: {0:s}'.format(dataset))
    for label in LABELS:
      # print('Processing label: {0:s}'.format(label))
      mean_score = 0.
      mean_std = 0.
      for nb_samples in SAMPLES:
        # print('Processing {0:d} samples'.format(nb_samples))
        # Read the data
        parser = Parser(dataset=dataset, objective=label)
        X, y, cat_idx, num_idx, task = parser.Parse(n_rows=nb_samples)

        # Baseline models to compare to
        models = [estimator.DPGBDT]

        error_metric = make_scorer(Mape)
        validator = model_selection.KFold(n_splits=NB_SPLITS,
                                          shuffle=True,
                                          random_state=0)

        for model in models:
          model_name = str(model).split('.')[-1][:-2]
          model_params = MODEL_PARAMS.get(model_name).get('Vanilla')
          nb_trees = model_params.get('nb_trees')
          nb_trees_per_ensemble = model_params.get(
              'nb_trees_per_ensemble', nb_trees)
          m = model(
              0.,  # Run our model in non-DP mode
              nb_trees,
              nb_trees_per_ensemble,
              model_params.get('max_depth'),
              model_params.get('learning_rate'),
              n_classes=len(set(y)) if task == 'classification' else None,
              max_leaves=model_params.get('max_leaves'),
              min_samples_split=model_params.get('min_samples_split', 2),
              balance_partition=model_params.get('balance_partition', True),
              use_bfs=model_params.get('use_bfs', False),
              use_3_trees=model_params.get('use_3_trees', False),
              cat_idx=cat_idx,
              num_idx=num_idx,
              verbosity=-1)  # type: ignore
          regressor = TransformedTargetRegressor(
              regressor=m,
              transformer=MinMaxScaler(feature_range=(-1, 1)))
          scores = cross_val_score(
              regressor, X, y,
              cv=validator, scoring=error_metric, n_jobs=-1)
          mean, std = scores.mean(), (scores.std() / 2)
          if nb_samples != 300:
            mean_score += mean
            mean_std += std
          print('Score: {0:f} +- {1:f}'.format(mean, std))
          output.write('{0:s},{1:d},{2:d},{3:d},{4:d},{5:d},{6:d},{7:f},'
                       '{8:d},{9:f},{10:f},{11:f},{12:f},{13:s},{14:s},'
                       '{15:d},{16:s}\n'.format(
            dataset, nb_samples, -1, nb_trees, nb_trees_per_ensemble,
            model_params.get('max_depth'),
            model_params.get('max_leaf_nodes', -1),
            model_params.get('learning_rate', -1), NB_SPLITS, mean, std,
            min(y), max(y), model_name, 'Vanilla', -1, label))
      mean_score /= (len(SAMPLES) - 1)
      mean_std /= (len(SAMPLES) - 1)
      print('---------------------------------------------------------------')
      print('Dataset {0:s} - Label {1:s}: Mean score {2:f}% +- {3:f}%'.format(
          dataset, label, mean_score, mean_std))
      print('---------------------------------------------------------------')
  output.close()
