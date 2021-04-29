# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Estimator wrapper around our implementation."""

from typing import Dict, Any, Optional, List

import numpy as np
import lightgbm as lgb

from sklearn.base import BaseEstimator

# pylint: disable=redefined-outer-name,invalid-name
from DPGBDT.model import GradientBoostingEnsemble


class DPGBDT(BaseEstimator):  # type: ignore
  """Wrapper around our implementation for model evaluation."""
  # pylint: disable=too-many-arguments

  def __init__(self,
               privacy_budget: float,
               nb_trees: int,
               nb_trees_per_ensemble: int,
               max_depth: int,
               learning_rate: float,
               early_stop: int = 5,
               n_classes: Optional[int] = None,
               binary_classification: Optional[bool] = False,
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
               test_size: float = 0.3,
               verbosity: int = -1) -> None:
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
      binary_classification (bool): Optional. Whether or not to use the
          regression model to perform a binary classification. Default is False.
      max_leaves (int): Optional. The max number of leaf nodes for the trees.
          Tree will grow in a best-leaf first fashion until it contains
          max_leaves or until it reaches maximum depth, whichever comes first.
      min_samples_split (int): Optional. The minimum number of samples required
          to split an internal node. Default is 2.
      gradient_filtering (bool): Optional. Whether or not to perform gradient
          based data filtering during training. Default is False.
      leaf_clipping (bool): Optional. Whether or not to clip the leaves
          after training. Default is False.
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
      test_size (float): Optional. Percentage of instances to use for
          validation. Default is 30%.
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
    self.binary_classification = binary_classification
    self.balance_partition = balance_partition
    self.use_bfs = use_bfs
    self.use_3_trees = use_3_trees
    self.use_decay = use_decay
    self.cat_idx = cat_idx
    self.num_idx = num_idx
    self.test_size = test_size
    self.verbosity = verbosity
    self.model = GradientBoostingEnsemble(
        self.nb_trees,
        self.nb_trees_per_ensemble,
        n_classes=self.n_classes_,
        binary_classification=self.binary_classification,
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
        test_size=self.test_size,
        verbosity=self.verbosity)
    self.n_features_ = None

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
    return self.model.Train(X, y)

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
      predictions = self.model.Predict(X).squeeze()
      if not self.binary_classification:
        return predictions
      encoded = np.where(predictions >= 0, 1, -1)
      return encoded

  def predict_proba(self, X: np.array) -> np.array:
    """Predict class probabilities for X.

    Args:
      X (np.array): The dataset for which to predict values.

    Returns:
      np.array: The class probabilities of the input samples.
    """
    assert self.model
    if not self.binary_classification:
      return self.model.PredictProba(X)
    predictions = self.model.Predict(X).squeeze()
    encoded = np.where(predictions >= 0, 1, -1)
    sigmoid = self.sigmoid(np.abs(predictions))
    probs = []
    for idx, pred in enumerate(sigmoid):
      true_value = encoded[idx]
      if true_value == -1:
        probs.append([sigmoid[idx], 1 - sigmoid[idx]])
      else:
        probs.append([1 - sigmoid[idx], sigmoid[idx]])
    return probs

  def decision_path(self, X: np.array, tree_index: int) -> np.array:
    label = self.predict(X)[0]
    trees = self.model.trees[tree_index]
    tree = trees[label] if len(trees) > 1 else trees[0]
    decision_path = tree.GetDecisionPathForRow(X)

    class Wrapper:
      def __init__(self, decision_path, label) -> None:
        self.indices = decision_path
        self.label = label
    return Wrapper(decision_path, label)

  def get_samples_at_node(self,
                          node_id: int,
                          label: int,
                          tree_index: int) -> int:
    trees = [k_three[label] if len(k_three) > 1 else k_three[0]
             for k_three in self.model.trees]
    nodes = trees[tree_index].nodes
    return len([n for n in nodes if n.node_id == node_id][0].X)

  @staticmethod
  def sigmoid(X: np.array):
    return 1 / (1 + np.exp(-X))

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
        'binary_classification': self.binary_classification,
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
        'test_size': self.test_size,
        'verbosity': self.verbosity
    }

  def set_params(self,
                 **parameters: Dict[str, Any]) -> 'DPGBDT':
    """Stub for sklearn cross validation"""
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self


class DPRef(BaseEstimator):  # type: ignore
  """Wrapper around reference implementation for model evaluation."""
  # pylint: disable=too-many-arguments

  def __init__(
      self,
      privacy_budget: float,
      nb_trees: int,
      nb_trees_per_ensemble: int,
      max_depth: int,
      learning_rate: float,
      n_classes: Optional[int] = None,  # pylint: disable=unused-argument
      binary_classification: Optional[bool] = False,
      early_stop: int = 5,  # pylint: disable=unused-argument
      max_leaves: Optional[int] = None,
      min_samples_split: int = 2,
      gradient_filtering: bool = False,  # pylint: disable=unused-argument
      leaf_clipping: bool = False,
      balance_partition: bool = True,
      use_bfs: bool = False,  # pylint: disable=unused-argument
      use_3_trees: bool = False,  # pylint: disable=unused-argument
      use_decay: bool = False,  # pylint: disable=unused-argument
      cat_idx: Optional[List[int]] = None,  # pylint: disable=unused-argument
      num_idx: Optional[List[int]] = None,  # pylint: disable=unused-argument,
      test_size: float = 0.3,  # pylint: disable=unused-argument,
      verbosity: int = -1,  # pylint: disable=unused-argument
      ) -> None:
    """Initialize the wrapper.

    Args:
      privacy_budget (float): The privacy budget to use.
      nb_trees (int): The number of trees in the model.
      nb_trees_per_ensemble (int): The number of trees per ensemble.
      max_depth (int): The max depth for the trees.
      learning_rate (float): The learning rate.
      n_classes (int): Unused. For CV only.
      binary_classification (bool): Optional. Whether or not to use the
          regression model to perform a binary classification. Default is False.
      early_stop (int): Unused. For CV only.
      max_leaves (int): Optional. The max number of leaf nodes for the trees.
          Tree will grow in a best-leaf first fashion until it contains
          max_leaves or until it reaches maximum depth, whichever comes first.
      min_samples_split (int): Optional. The minimum number of samples required
          to split an internal node. Default is 2.
      gradient_filtering (bool): Unused. For CV only.
      leaf_clipping (bool): Optional. Whether or not to clip the leaves
          after training. Default is False.
      balance_partition (bool): Optional. Balance data repartition for training
          the trees. The default is True, meaning all trees within an ensemble
          will receive an equal amount of training samples. If set to False,
          each tree will receive <x> samples where <x> is given in line 8 of
          the algorithm in the author's paper.
      use_bfs (bool): Unused. For CV only.
      use_3_trees (bool): Unused. For CV only.
      use_decay (bool): Unused. For CV only.
      cat_idx (List[int]): Unused. For CV only.
      num_idx (List[int]): Unused. For CV only.
      test_size (float): Optional. Percentage of instances to use for
          validation. Default is 30%.
      verbosity (int): Unused. For CV only.
    """
    self.model = None
    self.privacy_budget = privacy_budget
    self.nb_trees = nb_trees
    self.nb_trees_per_ensemble = nb_trees_per_ensemble
    self.binary_classification = binary_classification
    self.max_depth = max_depth
    self.learning_rate = learning_rate
    self.max_leaves = max_leaves
    self.min_samples_split = min_samples_split
    self.leaf_clipping = leaf_clipping
    self.balance_partition = balance_partition
    self.params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': self.max_leaves,
        'max_depth': self.max_depth,
        'learning_rate': self.learning_rate,
        'num_iterations': self.nb_trees,
        'my_n_trees': self.nb_trees,
        'lambda_l2': 0.1,
        'bagging_freq': 1,
        'bagging_fraction': 0.5,
        'max_bin': 255,
        'total_budget': self.privacy_budget,
        'boost_method': 'DPBoost_2level',
        'high_level_boost_round': 1,
        'inner_boost_round': self.nb_trees_per_ensemble,
        'balance_partition': 1 if self.balance_partition else 0,
        'geo_clip': 1 if self.leaf_clipping else 0,
        'verbose': -1,
    }

  def fit(self, X: np.array, y: np.array) -> 'lgb.Booster':
    """Stub for sklearn cross validation"""
    self.model = lgb.train(self.params,
                           lgb.Dataset(X, label=y),
                           num_boost_round=self.nb_trees)
    return self.model

  def predict(self, X: np.array) -> np.array:
    """Stub for sklearn cross validation"""
    assert self.model
    predictions = np.asarray(self.model.predict(X))
    if not self.binary_classification:
      return predictions
    encoded = np.where(predictions >= 0, 1, -1)
    return encoded

  def get_params(self,
                 deep: bool = True) -> Dict[str, Any]:  # pylint: disable=unused-argument
    """Stub for sklearn cross validation"""
    return {
        'privacy_budget': self.privacy_budget,
        'nb_trees': self.nb_trees,
        'nb_trees_per_ensemble': self.nb_trees_per_ensemble,
        'binary_classification': self.binary_classification,
        'max_depth': self.max_depth,
        'learning_rate': self.learning_rate,
        'max_leaves': self.max_leaves,
        'min_samples_split': self.min_samples_split,
        'leaf_clipping': self.leaf_clipping,
        'balance_partition': self.balance_partition
    }

  def set_params(self,
                 **parameters: Dict[str, Any]) -> 'DPRef':
    """Stub for sklearn cross validation"""
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self
