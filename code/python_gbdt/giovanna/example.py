# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Example test file."""

from typing import Optional, Any
import pprint

import numpy as np
# pylint: disable=import-error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
# pylint: enable=import-error

import estimator


def get_abalone(n_rows: Optional[int] = None) -> Any:
  """Parse the abalone dataset.

  Args:
    n_rows (int): Numbers of rows to read.
  Returns:
    Any: X, y, cat_idx, num_idx
  """
  # pylint: disable=redefined-outer-name,invalid-name
  # Re-encode gender information
  data = pd.read_csv(
      './abalone.data',
      names=['sex', 'length', 'diameter', 'height', 'whole weight',
             'shucked weight', 'viscera weight', 'shell weight', 'rings'])
  data['sex'] = pd.get_dummies(data['sex']) # make a number out of gender M/F
  if n_rows:
    data = data.head(n_rows)
  y = data.rings.values.astype(np.float64) # want to predict age
  del data['rings']                        # rest is X
  X = data.values.astype(np.float64) 
  cat_idx = [0]  # Sex
  num_idx = list(range(1, X.shape[1]))  # Other attributes
  return X, y, cat_idx, num_idx


if __name__ == '__main__':

  # They are identical
  def rmse(y_pred, y_test):
    #return np.sqrt(np.mean(np.square(y_pred - y_test))) # theos rmse
    return metrics.mean_squared_error(y_test, y_pred, squared=False) # sklearn's rmse

  pp = pprint.PrettyPrinter(indent=2)
  #pp.pprint(model.get_params())

  NB_TREES = 50
  PRIVACY_BUDGET = [0.5] # np.arange(0.1, 4.6, 0.3) # was 0.1
  BALANCE = True # for DFS and 3-nodes
  VERBOSE = True
  SCALE = False

  NUM_RUNS = 2

  X, y, cat_idx, num_idx = get_abalone()

  # scale!!!
  if SCALE:
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler(feature_range=(-1, 1))
    X[:,1:] = mms.fit_transform(X[:,1:]) # not sure if X should be
    y = mms.fit_transform(y.reshape(-1, 1)) # this affects the RMSE
    y = np.array([elem[0] for elem in y]) # restore the datastructure

  for p_value in PRIVACY_BUDGET:
    results_rmse = []
    results_trees = []
    for i in range(NUM_RUNS):

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

      # A simple baseline: mean of the training set
      y_pred = np.mean(y_train).repeat(len(y_test))

      rmse0 = rmse(y_pred,y_test)
      if VERBOSE:
        print('Mean'.ljust(25), 'RMSE: {0:.2f}'.format(rmse0))

      # Train the model using a depth-first approach
      model = estimator.DPGBDT(privacy_budget=p_value,
                              nb_trees=NB_TREES,
                              nb_trees_per_ensemble=50,
                              max_depth=6,
                              learning_rate=0.1,
                              cat_idx=cat_idx,
                              num_idx=num_idx,
                              verbosity=1,
                              balance_partition=BALANCE)
      
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      rmse1 = rmse(y_pred,y_test)
      ntree1 = model.num_used_trees
      if VERBOSE:
        print('Depth first growth'.ljust(25), 'RMSE: {0:.2f}'.format(rmse1), 
            '   #trees used: {}/{}'.format(ntree1, model.nb_trees))

      # Train the model using a best-leaf first approach
      model = estimator.DPGBDT(privacy_budget=p_value,
                              nb_trees=NB_TREES,
                              nb_trees_per_ensemble=50,
                              max_depth=6,
                              max_leaves=24,
                              learning_rate=0.1,
                              use_bfs=True,
                              cat_idx=cat_idx,
                              num_idx=num_idx,
                              verbosity=-1,
                              balance_partition=True)
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      rmse2 = rmse(y_pred,y_test)
      ntree2 = model.num_used_trees
      if VERBOSE:
        print('Best-leaf first growth'.ljust(25), 'RMSE: {0:.2f}'.format(rmse2), 
            '   #trees used: {}/{}'.format(ntree2, model.nb_trees))

      # Train the model using 3-nodes trees combination approach
      model = estimator.DPGBDT(privacy_budget=p_value,
                              nb_trees=NB_TREES,
                              nb_trees_per_ensemble=50,
                              max_depth=6,
                              learning_rate=0.1,
                              use_3_trees=True,
                              cat_idx=cat_idx,
                              num_idx=num_idx,
                              verbosity=-1,
                              balance_partition=BALANCE)
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      rmse3 = rmse(y_pred,y_test)
      ntree3 = model.num_used_trees
      if VERBOSE:
        print('3-nodes trees growth'.ljust(25),'RMSE: {0:.2f}'.format(rmse3), 
            '   #trees used: {}/{}'.format(ntree3, model.nb_trees))
        print("--------({}/{})".format(i,NUM_RUNS-1))
      results_rmse.append((rmse0,rmse1,rmse2,rmse3))
      results_trees.append((ntree1,ntree2,ntree3))

    print("\n===== Average ===== (pb {}) \n".format(round(p_value,1)))
    res0 = [round(sum(ele) / len(results_rmse),2) for ele in zip(*results_rmse)]
    res1 = [int(sum(ele) / len(results_trees)) for ele in zip(*results_trees)]
    print("RMSEs: {}   #trees: {}/{}".format(res0, res1, NB_TREES))
    print("RMSEs scale adjusted: {}".format([str(round(val/(max(y)-min(y)),2))+"%" for val in res0]))


  
