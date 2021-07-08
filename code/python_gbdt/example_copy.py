# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Example test file. Copied here from the public github,
    to check whether model/estimator is the same.py
    To verify because BFS gave such bad results. 
    
    Results are identical to giovanna/ code.
    """

from typing import Optional, Any

import numpy as np
# pylint: disable=import-error
import pandas as pd
from sklearn.model_selection import train_test_split
# pylint: enable=import-error

from evaluation import estimator

def gett_abalone(n_rows: Optional[int] = None) -> Any:
  """Parse the abalone dataset.

  Args:
    n_rows (int): Numbers of rows to read.
  Returns:
    Any: X, y, cat_idx, num_idx
  """
  # pylint: disable=redefined-outer-name,invalid-name
  # Re-encode gender information
  data = pd.read_csv(
      './data/src/real/abalone.data',
      names=['sex', 'length', 'diameter', 'height', 'whole weight',
             'shucked weight', 'viscera weight', 'shell weight', 'rings'])
  data['sex'] = pd.get_dummies(data['sex'])
  if n_rows:
    data = data.head(n_rows)
  y = data.rings.values.astype(np.float64)
  del data['rings']
  X = data.values.astype(np.float64)
  cat_idx = [0]  # Sex
  num_idx = list(range(1, X.shape[1]))  # Other attributes
  return X, y, cat_idx, num_idx


if __name__ == '__main__':
  PRIVACY_BUDGET = 1 # was 0.1
  # pylint: disable=redefined-outer-name,invalid-name
  X, y, cat_idx, num_idx = gett_abalone()
  X_train, X_test, y_train, y_test = train_test_split(X, y)

  # A simple baseline: mean of the training set
  y_pred = np.mean(y_train).repeat(len(y_test))
  print('Mean'.ljust(25), 'RMSE: {0:.2f}'.format(
    np.sqrt(np.mean(np.square(y_pred - y_test)))))

  # Train the model using a depth-first approach
  model = estimator.DPGBDT(privacy_budget=PRIVACY_BUDGET,
                           nb_trees=50,
                           nb_trees_per_ensemble=50,
                           max_depth=6,
                           learning_rate=0.1,
                           cat_idx=cat_idx,
                           num_idx=num_idx,
                           verbosity=-1)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print('Depth first growth'.ljust(25), 'RMSE: {0:.2f}'.format(
      np.sqrt(np.mean(np.square(y_pred - y_test)))), 
      '   #trees used: {}/{}'.format(model.num_used_trees, model.nb_trees))

  # Train the model using a best-leaf first approach
  model = estimator.DPGBDT(privacy_budget=PRIVACY_BUDGET,
                           nb_trees=50,
                           nb_trees_per_ensemble=50,
                           max_depth=6,
                           max_leaves=24,
                           learning_rate=0.1,
                           use_bfs=True,
                           cat_idx=cat_idx,
                           num_idx=num_idx)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print('Best-leaf first growth'.ljust(25), 'RMSE: {0:.2f}'.format(
      np.sqrt(np.mean(np.square(y_pred - y_test)))), 
      '   #trees used: {}/{}'.format(model.num_used_trees, model.nb_trees))

  # Train the model using 3-nodes trees combination approach
  model = estimator.DPGBDT(privacy_budget=PRIVACY_BUDGET,
                           nb_trees=50,
                           nb_trees_per_ensemble=50,
                           max_depth=6,
                           learning_rate=0.1,
                           use_3_trees=True,
                           cat_idx=cat_idx,
                           num_idx=num_idx)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print('3-nodes trees growth'.ljust(25),'RMSE: {0:.2f}'.format(
      np.sqrt(np.mean(np.square(y_pred - y_test)))), 
      '   #trees used: {}/{}'.format(model.num_used_trees, model.nb_trees))
