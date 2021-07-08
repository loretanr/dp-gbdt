# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Implement a random forest algorithm using scikit learn."""
from typing import Tuple, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from prototyping.utils import utils


class RandomForest:
  """Implement a random forest using scikit-learn."""

  def __init__(self,
               x: pd.DataFrame,
               y: pd.DataFrame,
               test_size: float = 0.3,
               nb_trees: int = 100,
               max_depth: Optional[int] = None,
               criterion: str = 'entropy') -> None:
    """Initialize the random forest class.

    Args:
      x (pd.Dataframe): A pandas dataframe containing the samples.
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
    self.classifier = RandomForestClassifier(
        n_estimators=self.nb_trees,
        max_depth=self.max_depth,
        criterion=self.criterion)

  def Train(self) -> 'RandomForest':
    """Train the classifier

    Returns:
      RandomForest: A random forest classifier object.
    """
    self.classifier.fit(self.x_train, self.y_train)
    return self

  def Predict(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Predict values using the trained classifier.

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the true data and
          the predictions.
    """
    y_pred = self.classifier.predict(self.x_test)  # type: pd.DataFrame
    return self.y_test, y_pred
