# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Implement a random forest classifier."""

import math
import random
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor

from typing import Optional, List, Any, Tuple, Dict, TYPE_CHECKING
from scipy.stats import stats

import numpy as np

from prototyping.utils import utils

if TYPE_CHECKING:
  import pandas as pd


class RandomForest:
  """Implement a random forest."""

  def __init__(self,
               x: 'pd.DataFrame',
               y: 'pd.DataFrame',
               test_size: float = 0.3,
               nb_trees: int = 100,
               max_depth: int = -1,
               random_features: bool = False,
               criterion: str = 'entropy') -> None:
    """Initialize the random forest class.

    Args:
      x (pd.Dataframe): A pandas dataframe containing the features.
      y (pd.Dataframe): A pandas dataframe containing the labels.
      test_size (float): The percentage to use for splitting the training and
          testing data. Default is 0.3, i.e. 70% training and 30% testing.
      nb_trees (int): Optional. The number of decision trees in the forest.
          Default is 100.
      max_depth (int): Optional. The maximum depth at which to expand the
          trees. If None, the the tree is expanded until all leaves are pure.
      random_features (bool): Optional. If false, all features of the dataset
          are used. Otherwise, a subset of them only (sqrt, selected at random).
      criterion (str): Optional. The criterion to use to build the tree.
          Default is entropy. Available: ['entropy', 'gini']
    """
    self.x = x
    self.y = y
    self.test_size = test_size
    self.nb_trees = nb_trees
    self.max_depth = max_depth
    self.random_features = random_features
    self.criterion = criterion
    self.x_train, self.x_test, self.y_train, self.y_test = utils.Split(
        self.x, self.y, self.test_size)
    self.classifier = RandomForestClassifier(
        nb_trees=self.nb_trees,
        random_features=self.random_features,
        max_depth=self.max_depth,
        criterion=self.criterion)

  def Train(self) -> 'RandomForest':
    """Train the classifier

    Returns:
      RandomForest: A random forest classifier object.
    """
    self.classifier.Fit(self.x_train, self.y_train)
    return self

  def Predict(self) -> Tuple['pd.DataFrame', 'pd.DataFrame']:
    """Predict values using the trained classifier.

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the true data and
          the predictions.
    """
    y_pred = self.classifier.Predict(self.x_test)  # type: pd.DataFrame
    return self.y_test, y_pred


class DecisionNode:
  """Implement a decision node."""

  def __init__(self,
               index: int = -1,
               value: Optional[Any] = None,
               predictions: Optional[Any] = None,
               left_child: Optional[Any] = None,
               right_child: Optional[Any] = None) -> None:
    """Initialize a decision node.

    Args:
      index (int): An index for the feature on which the node splits.
      value (Any): The corresponding value for that index.
      predictions (Any): If the node is a leaf node, the associated
          predictions.
      left_child (DecisionNode): The left child of the node, if any.
      right_child (DecisionNode): The right child of the node, if any.
    """
    self.index = index
    self.value = value
    self.predictions = predictions
    self.left_child = left_child
    self.right_child = right_child


class DecisionTreeClassifier:
  """Implement a decision tree classifier."""

  def __init__(self,
               max_depth: Optional[int] = None,
               random_features: bool = False,
               criterion: str = 'entropy') -> None:
    """Initialize the decision tree classifier.

    Args:
      max_depth (int): Maximum depth for the decision trees.
      random_features (bool): Optional. If false, all features of the dataset
          are used. Otherwise, a subset of them only (sqrt, selected at random).
      criterion (str): Optional. The criterion to use to build the tree.
          Default is entropy. Available: ['entropy', 'gini']
    """
    self.root_node = None  # type: Optional[DecisionNode]
    self.max_depth = max_depth or -1
    self.features_indexes = []  # type: List[int]
    self.random_features = random_features
    self.criterion = criterion

  def Fit(self, dataset: np.ndarray) -> None:
    """Create the decision tree based on the data.

    Args:
      dataset (np.ndarray): The dataset to fit.
    """

    if self.random_features:
      nb_features = dataset.shape[1] - 1
      self.features_indexes = random.sample(
          range(nb_features), int(math.sqrt(nb_features)))
      # Select the random features and keep the label column
      dataset = np.c_[dataset[:, self.features_indexes], dataset[:, -1]]
    self.root_node = self._MakeTree(dataset, self.max_depth)

  def Predict(self, dataset: np.ndarray) -> List[Any]:
    """Return predictions for a list of input data.

    Args:
      dataset: The input data used for prediction. This is a ndarray (list of
      list)

    Returns:
      List[Any]: A list containing the predictions.
    """
    predictions = []
    for row in dataset:
      if self.random_features:
        row = row[self.features_indexes]
      predictions.append(self._Predict(row, self.root_node)[0])  # type: ignore
    return predictions

  def _Predict(self, row: List[Any], node: DecisionNode) -> Any:
    """Walk through the decision tree to classify the row.

    Args:
      row (List): The row to classify.
      node (DecisionNode): The current decision node.

    Returns:
      List[Any]: A prediction for the row.
    """
    if node.predictions:
      return list(node.predictions.keys())[0]

    value = row[node.index]
    if isinstance(value, (float, int)):
      if value >= node.value:  # type: ignore
        child_node = node.right_child
      else:
        child_node = node.left_child
    else:
      if value == node.value:
        child_node = node.right_child
      else:
        child_node = node.left_child
    return self._Predict(row, child_node)  # type: ignore

  @staticmethod
  def _Split(index: int,
             value: Any,
             x: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Split a dataset based on the value of a given feature.

    Args:
      index (int): The index of the feature to use for the split.
      value (Any): The value of that feature.
      x (List[Any]): The dataset (the features).

    Returns:
      Tuple[List[Any], List[Any]]: A split based on the feature's value for the
          dataset.
    """
    lhs, rhs = [], []
    for row in x:
      if row[index] < value:
        lhs.append(row)
      else:
        rhs.append(row)
    return lhs, rhs

  @staticmethod
  def CountOccurrence(dataset: List[Any]) -> Dict[Any, int]:
    """Count occurrences of each label in all rows.

    Args:
      dataset (List[Any]): The dataset.

    Returns:
      Dict[Any, int]: A dictionary that maps labels to their occurrences
          across the dataset.
    """
    results = defaultdict(int)  # type: Dict[Any, int]
    for row in dataset:
      results[row[-1]] += 1
    return results

  def _Entropy(self, dataset: List[Any]) -> float:
    """Compute the entropy for a set of rows.

    Args:
      dataset (List[Any]): A set of rows.

    Returns:
      float: The entropy of the set of rows.
    """
    results = self.CountOccurrence(dataset)
    ent = 0.
    for r in results.keys():
      p = float(results[r]) / len(dataset)
      ent = ent - p * (math.log(p) / math.log(2))
    return ent

  def _MakeTree(self, dataset: List[Any], depth: int) -> DecisionNode:
    """Recursively constructs the decision tree.

    Args:
      dataset (List[Any]): The set of rows (i.e. the dataset).
      depth (int): The depth for the tree.

    Returns:
      DecisionNode: The root node for the tree.
    """
    if len(dataset) == 0:
      return DecisionNode()
    if depth == 0:
      return DecisionNode(predictions=self.CountOccurrence(dataset))

    current_score = self._Entropy(
        dataset) if self.criterion == 'entropy' else float(math.inf)
    best_gain = 0.
    best_criteria = (None, None)  # type: Tuple[Any, Any]
    best_groups = (None, None)  # type: Tuple[Any, Any]

    if self.criterion == 'gini':
      labels = list({row[-1] for row in dataset})

    for feature_index in range(len(dataset[0]) - 1):
      for value in {row[feature_index] for row in dataset}:
        lhs, rhs = self._Split(feature_index, value, dataset)
        if self.criterion == 'entropy':
          p = float(len(lhs)) / len(dataset)
          gain = current_score - p * self._Entropy(
              lhs) - (1 - p) * self._Entropy(rhs)
          if gain > best_gain and len(lhs) > 0 and len(rhs) > 0:
            best_gain = gain
            best_criteria = (feature_index, value)
            best_groups = (lhs, rhs)
        elif self.criterion == 'gini':
          gini = self._GetGiniIndex(lhs, rhs, labels)
          if gini < current_score and len(lhs) > 0 and len(rhs) > 0:
            current_score = gini
            best_criteria = (feature_index, value)
            best_groups = (lhs, rhs)

    if best_gain > 0:
      left_child = self._MakeTree(best_groups[0], depth - 1)
      right_child = self._MakeTree(best_groups[1], depth - 1)
      return DecisionNode(index=best_criteria[0],
                          value=best_criteria[1],
                          left_child=left_child, right_child=right_child)
    return DecisionNode(predictions=self.CountOccurrence(dataset))

  @staticmethod
  def _GetGiniIndex(lhs: List[Any],
                    rhs: List[Any],
                    labels: List[Any]) -> float:
    """Calculate the Gini index for a list of groups and known class values.

    TODO: Depending on the dataset, using Gini seems to be either giving
      excellent results, or completely off ones. Could be an error in the code
      below. Investigate.

    Args:
      lhs (List[Any]): One possible split.
      rhs (List[Any]): Another possible split.
      labels (List[Any]): The list of possible labels.

    Returns:
      float: The Gini Index score.
    """
    # Count all samples at split point
    n_instances = float(sum([len(lhs), len(rhs)]))
    # Weighted sum Gini index for each group
    gini = 0.
    for split in [lhs, rhs]:
      size = float(len(split))
      if size == 0:
        continue
      score = 0.
      # Score the group based on the score for each class
      for label in labels:
        p = [row[-1] for row in split].count(label) / size
        score += p * p
      # Weight the group score by its relative size
      gini += (1. - score) * (size / n_instances)
    return gini


class RandomForestClassifier:
  """Implement a random forest classifier."""

  def __init__(self,
               nb_trees: int = 100,
               random_features: bool = False,
               max_depth: Optional[int] = None,
               criterion: str = 'entropy') -> None:
    """Initialize the random forest classifier.

    Args:
      nb_trees (int): Optional. The number of decision trees in the forest.
          Default is 100.
      random_features (bool): Optional. If false, all features of the dataset
          are used. Otherwise, a subset of them only (sqrt, selected at random).
      max_depth (int): Optional. The maximum depth at which to expand the
          trees. If None, the the tree is expanded until all leaves are pure.
      criterion (str): Optional. The criterion to use to build the trees.
          Default is entropy. Available: ['entropy', 'gini']
    """
    self.nb_trees = nb_trees
    self.max_depth = max_depth
    self.criterion = criterion
    self.random_features = None or random_features
    self.trees = []  # type: List[DecisionTreeClassifier]

  def FitTree(self, x_tree: np.ndarray) -> DecisionTreeClassifier:
    """Fit a decision tree.

    Args:
      x_tree (Tuple[int, np.ndarray]): A tuple containing the index of the
          tree, and the dataset on which to fit it.

    Returns:
      DecisionTreeClassifier: A fitted tree.
    """
    tree = DecisionTreeClassifier(
        max_depth=self.max_depth,
        random_features=self.random_features,
        criterion=self.criterion)
    tree.Fit(x_tree[1])
    return tree

  def Fit(self, x: 'pd.DataFrame', y: 'pd.DataFrame') -> None:
    """Fit the trees.

    Args:
      x (pd.Dataframe): A pandas dataframe containing the samples.
      y (pd.Dataframe): A pandas dataframe containing the labels.
    """
    with ProcessPoolExecutor(max_workers=16) as executor:
      # Put x and y in one big np 2d array, and select a random subset of
      # rows, with replacement
      sample = map(lambda l: [l, np.c_[x, y][[random.randint(
          0, x.shape[0] - 1) for _ in range(
              x.shape[0])], :]], range(self.nb_trees))
      # Train in parallel all trees with their respective random subset
      self.trees = list(executor.map(self.FitTree, sample))

  def Predict(self, x: np.ndarray) -> Any:
    """Predict a value for x using majority voting for the trees.

    Args:
      x (pd.DataFrame): A list of values of features to use for predicting
          the label.

    Returns:
      np.ndarray: The predictions for the dataset
    """
    return stats.mode([tree.Predict(x) for tree in self.trees])[0][0]
