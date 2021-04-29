# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Implement baseline gradient boosted trees."""

import json
from typing import Any, Dict

import numpy as np

from sklearn import metrics, model_selection
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, \
  ExtraTreesRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from data.parser.parser import Parser

# The number of time to repeat the experiment to get an average accuracy
NB_SPLITS = 5
NB_TREES = [1] + list(range(10, 110, 10))
MAX_DEPTH = range(2, 7)
# Since LightGBT implementation has max leaves defaulting to 31, use this
# value for grid-search
# MAX_LEAVES = list(range(3, 32))  # Min and max nb of leaves for depth 2 and 6
# MAX_LEAVES.insert(0, None)  # type: ignore
LEARNING_RATE = np.arange(0.01, 1, 0.01)
# We focus on least square methods since this is the only one supported by
# the paper in LightGBM
LOSS = ['ls']
SAMPLES = 5000

# Different dataset
DATASETS = ['abalone', 'adult', 'bcw', 'questionnaires', 'year']

if __name__ == '__main__':
  models = [GradientBoostingRegressor,
            RandomForestRegressor,
            ExtraTreesRegressor]
  best_params = {}  # type: Dict[str, Any]
  output = open('baseline_grid_search.json', 'w')
  for DATASET in DATASETS:
    print('----- Working on dataset: {0:s}'.format(DATASET))
    best_params[DATASET] = {}
    parser = Parser(dataset=DATASET)
    X, y, _, _, _ = parser.Parse(n_rows=SAMPLES)

    # Normalizing data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(y.reshape(-1, 1))
    y = scaler.transform(y.reshape(-1, 1)).reshape(-1)

    for model in models:
      model_name = str(model().__class__).split('.')[-1][:-2]
      print('----------- Grid searching on model: {0:s}'.format(model_name))
      if model_name in ['RandomForestRegressor', 'ExtraTreesRegressor']:
        params = {'n_estimators': NB_TREES,
                  'max_depth': MAX_DEPTH}
      else:
        params = {'n_estimators': NB_TREES,
                  'max_depth': MAX_DEPTH,
                  'learning_rate': LEARNING_RATE,
                  'loss': LOSS}  # type: ignore
      rmse = make_scorer(metrics.mean_squared_error,
                         squared=False,
                         greater_is_better=False)
      validator = model_selection.KFold(n_splits=NB_SPLITS, shuffle=True)
      clf = GridSearchCV(
          model(), params, scoring=rmse, n_jobs=-1, cv=validator, verbose=1)
      clf.fit(X, y)
      best_params[DATASET][model_name] = clf.best_params_

  output.write(json.dumps(best_params, indent=2))
  output.close()
