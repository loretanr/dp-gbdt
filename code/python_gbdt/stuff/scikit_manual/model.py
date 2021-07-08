# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Implement a random forest algorithm using scikit learn's
DecisionTreeClassifier objects."""

import random

from typing import TYPE_CHECKING, Optional, List, Any, Tuple

from scipy.stats import stats
from sklearn.tree import DecisionTreeClassifier

from prototyping.utils import utils

if TYPE_CHECKING:
  import pandas as pd


class RandomForest:
  """Implement a random forest using scikit-learn's DecisionTreeClassifier."""

  def __init__(self,
               x: 'pd.DataFrame',
               y: 'pd.DataFrame',
               test_size: float = 0.3,
               nb_trees: int = 100,
               max_depth: int = -1,
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
      criterion (str): Optional. The criterion to use to build the tree.
          Default is entropy. Available: ['entropy', 'gini']
    """
    self.x = x
    self.y = y
    self.test_size = test_size
    self.nb_trees = nb_trees
    self.max_depth = max_depth
    self.criterion = criterion
    self.x_train, self.x_test, self.y_train, self.y_test = utils.Split(
        self.x, self.y, self.test_size)
    self.classifier = RandomForestClassifier()

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


class RandomForestClassifier:
  """Implement a random forest classifier."""

  def __init__(self,
               nb_trees: int = 100,
               max_depth: Optional[int] = None,
               criterion: str = 'entropy') -> None:
    """Initialize the random forest classifier.

    Args:
      nb_trees (int): Optional. The number of decision trees in the forest.
          Default is 100.
      max_depth (int): Optional. The maximum depth at which to expand the
          trees. If None, the the tree is expanded until all leaves are pure.
      criterion (str): Optional. The criterion to use to build the tree.
          Default is entropy. Available: ['entropy', 'gini']
    """
    self.nb_trees = nb_trees
    self.max_depth = max_depth
    self.nb_trees = nb_trees
    self.max_depth = max_depth
    self.criterion = criterion
    self.trees = []  # type: List[Any]

  def GrowForest(self) -> List[DecisionTreeClassifier]:
    """Populate the trees in the random forest."""
    return [DecisionTreeClassifier(
        max_depth=self.max_depth,
        criterion=self.criterion) for _ in range(self.nb_trees)]

  def Fit(self, x: 'pd.DataFrame', y: 'pd.DataFrame') -> None:
    """Fit the trees.

    Args:
      x (pd.Dataframe): A pandas dataframe containing the samples.
      y (pd.Dataframe): A pandas dataframe containing the labels.
    """
    self.trees = self.GrowForest()

    # Select a random selection (with replacement) of rows
    rows = self._GetSampleOfRows(x)
    x_tree = x[rows, :]
    y_tree = y[rows]
    for tree in self.trees:
      tree.fit(x_tree, y_tree)

  def Predict(self, x: 'pd.DataFrame') -> Any:
    """Predict a value for x.

    Args:
      x (pd.DataFrame): A list of values of features to use for predicting
          the label.

    Returns:
      List: The predictions for the dataset.
    """
    predictions = []
    raw_predictions = [tree.predict(x) for tree in self.trees]
    # For every row to predict in x
    for i in range(0, x.shape[0]):
      candidates = []
      # For every tree
      for j in range(0, self.nb_trees):
        candidates.append(raw_predictions[j][i])
      predictions.append(stats.mode(candidates)[0][0])
    return predictions

  @staticmethod
  def _GetSampleOfRows(x: 'pd.DataFrame') -> List[int]:
    """Take a random sample from the dataset, with replacement.

    Args:
      x (pd.DataFrame): A pandas dataframe containing samples.

    Returns:
      List[int]: A list of random indexes for rows to select for the trees.
    """
    return [random.randint(0, x.shape[0] - 1) for _ in range(x.shape[0])]
