# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Estimator wrapper around the implementation."""

from typing import Any, Dict, List, Optional

import numpy as np
# pylint: disable=import-error
from sklearn.base import BaseEstimator
# pylint: enable=import-error

from model import GradientBoostingEnsemble


class DPGBDT(BaseEstimator):  # type: ignore
  """Scikit wrapper around the model."""
  # pylint: disable=too-many-arguments, invalid-name

  def __init__(self,
               privacy_budget: float,
               nb_trees: int,
               nb_trees_per_ensemble: int,
               max_depth: int,
               learning_rate: float,
               early_stop: int = 5,
               n_classes: Optional[int] = None,
               max_leaves: Optional[int] = None,
               min_samples_split: int = 2,
               gradient_filtering: bool = False,
               leaf_clipping: bool = False,
               balance_partition: bool = True,
               use_bfs: bool = False,
               use_3_trees: bool = False,
               use_decay: bool = False,
               cat_idx: Optional[List[int]] = None,
               num_idx: Optional[List[int]] = None,
               verbosity: int = -1) -> None: # was -1
    """Initialize the wrapper.

    Args:
      privacy_budget (float): The privacy budget to use.
      nb_trees (int): The number of trees in the model.
      nb_trees_per_ensemble (int): The number of trees per ensemble.
      max_depth (int): The max depth for the trees.
      learning_rate (float): The learning rate.
      early_stop (int): Optional. If the rmse doesn't decrease for <int>
          consecutive rounds, abort training. Default is 5.
      n_classes (int): Number of classes. Triggers regression (None) vs
          classification.
      max_leaves (int): Optional. The max number of leaf nodes for the trees.
          Tree will grow in a best-leaf first fashion until it contains
          max_leaves or until it reaches maximum depth, whichever comes first.
      min_samples_split (int): Optional. The minimum number of samples required
          to split an internal node. Default is 2.
      gradient_filtering (bool): Optional. Whether or not to perform gradient
          based data filtering during training (only available on regression).
          Default is False.
      leaf_clipping (bool): Optional. Whether or not to clip the leaves
          after training (only available on regression). Default is False.
      balance_partition (bool): Optional. Balance data repartition for training
          the trees. The default is True, meaning all trees within an ensemble
          will receive an equal amount of training samples. If set to False,
          each tree will receive <x> samples where <x> is given in line 8 of
          the algorithm in the author's paper.
      use_bfs (bool): Optional. If max_leaves is specified, then this is
          automatically True. This will build the tree in a BFS fashion instead
          of DFS. Default is False.
      use_3_trees (bool): Optional. If True, only build trees that have 3
          nodes, and then assemble nb_trees based on these sub-trees, at random.
          Default is False.
      use_decay (bool): Optional. If True, internal node privacy budget has a
          decaying factor.
      cat_idx (List): Optional. List of indices for categorical features.
      num_idx (List): Optional. List of indices for numerical features.
      verbosity (int): Optional. Verbosity level for debug messages. Default
          is -1, meaning only warnings and above are displayed. 0 is info,
          1 is debug.
    """
    self.model = None
    self.privacy_budget = privacy_budget
    self.nb_trees = nb_trees
    self.nb_trees_per_ensemble = nb_trees_per_ensemble
    self.max_depth = max_depth
    self.max_leaves = max_leaves
    self.min_samples_split = min_samples_split
    self.gradient_filtering = gradient_filtering
    self.leaf_clipping = leaf_clipping
    self.learning_rate = learning_rate
    self.early_stop = early_stop
    self.n_classes_ = n_classes
    self.balance_partition = balance_partition
    self.use_bfs = use_bfs
    self.use_3_trees = use_3_trees
    self.use_decay = use_decay
    self.cat_idx = cat_idx
    self.num_idx = num_idx
    self.verbosity = verbosity
    self.model = GradientBoostingEnsemble(
        nb_trees=self.nb_trees,
        nb_trees_per_ensemble=self.nb_trees_per_ensemble,
        n_classes=self.n_classes_,
        max_depth=self.max_depth,
        privacy_budget=self.privacy_budget,
        learning_rate=self.learning_rate,
        early_stop=self.early_stop,
        max_leaves=self.max_leaves,
        min_samples_split=self.min_samples_split,
        gradient_filtering=self.gradient_filtering,
        leaf_clipping=self.leaf_clipping,
        balance_partition=self.balance_partition,
        use_bfs=self.use_bfs,
        use_3_trees=self.use_3_trees,
        use_decay=self.use_decay,
        cat_idx=self.cat_idx,
        num_idx=self.num_idx,
        verbosity=self.verbosity)
    self.n_features_ = None
    self.num_used_trees = None

  def fit(self, X: np.array, y: np.array) -> 'GradientBoostingEnsemble':
    """Fit the model to the dataset.

    Args:
      X (np.array): The features.
      y (np.array): The label.

    Returns:
      GradientBoostingEnsemble: A GradientBoostingEnsemble object.
    """
    assert self.model
    self.n_features_ = X.shape[1]
    ensemble = self.model.Train(X, y)
    self.num_used_trees = len(ensemble.trees)
    return ensemble

  def predict(self, X: np.array) -> np.array:
    """Predict the label for a given dataset.

    Args:
      X (np.array): The dataset for which to predict values.

    Returns:
      np.array: The predictions.
    """
    assert self.model
    # try classification output first,
    # o.w. fallback to the raw regression values
    try:
      return self.model.PredictLabels(X)
    except ValueError:
      return self.model.Predict(X).squeeze()

  def predict_proba(self, X: np.array) -> np.array:
    """Predict class probabilities for X.

    Args:
      X (np.array): The dataset for which to predict values.

    Returns:
      np.array: The class probabilities of the input samples.
    """
    assert self.model
    return self.model.PredictProba(X)

  def get_params(
      self,
      deep: bool = True) -> Dict[str, Any]:  # pylint: disable=unused-argument
    """Stub for sklearn cross validation"""
    return {
        'privacy_budget': self.privacy_budget,
        'nb_trees': self.nb_trees,
        'nb_trees_per_ensemble': self.nb_trees_per_ensemble,
        'max_depth': self.max_depth,
        'learning_rate': self.learning_rate,
        'early_stop': self.early_stop,
        'n_classes': self.n_classes_,
        'max_leaves': self.max_leaves,
        'min_samples_split': self.min_samples_split,
        'gradient_filtering': self.gradient_filtering,
        'leaf_clipping': self.leaf_clipping,
        'balance_partition': self.balance_partition,
        'use_bfs': self.use_bfs,
        'use_3_trees': self.use_3_trees,
        'use_decay': self.use_decay,
        'cat_idx': self.cat_idx,
        'num_idx': self.num_idx,
        'verbosity': self.verbosity
    }

  def set_params(self, **parameters: Dict[str, Any]) -> 'DPGBDT':
    """Stub for sklearn cross validation"""
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self
