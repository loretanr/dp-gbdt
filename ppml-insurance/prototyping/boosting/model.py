# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Implement ensemble of gradient boosted trees."""

import math
from collections import defaultdict
from typing import Tuple, Dict, List, Any

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingEnsemble:
  """Implement gradient boosting on ensemble of trees.

  Attributes:
    X (np.array): The dataset
    y (np.array): The labels
    X_train (np.array): The training dataset.
    X_test (np.array): The testing dataset.
    y_train (np.array): The training labels.
    y_test (np.array): The testing labels.
    y_train_pred (np.array): The initial predictions.
    nb_trees (int): The total number of trees in the model.
    nb_trees_per_ensemble (int): The number of trees in each ensemble.
    test_size (float): Optional. The proportion of the dataset that should be
        used for testing vs. for training. Default is 25% testing,
        75% training.
    max_depth (int): Optional. The depth for the trees. Default is 6.
    privacy_budget (int): Optional. The privacy budget available for the
        model. Default is 1.
    learning_rate (float): Optional. The learning rate. Default is 0.1.
    shrinkage_rate (float): Optional. The shrinking rate for the trees in
        the ensemble. This controls how many rows will go into each tree's
        training set. Default is 0.1.
    threshold (int): Optional. Threshold for the loss function. For the
        square loss function (default), this is 1.
    ensembles (Dict[int, Any]): A dictionary mapping the ensemble IDs to the
        trees they contain.
  """
  # pylint: disable=invalid-name, too-many-arguments, unused-variable

  def __init__(self,
               X: np.array,
               y: np.array,
               nb_trees: int,
               nb_trees_per_ensemble: int,
               test_size: float = 0.25,
               max_depth: int = 6,
               privacy_budget: int = 1,
               learning_rate: float = 0.1,
               shrinkage_rate: float = 0.1,
               threshold: int = 1) -> None:
    """Initialize the GradientBoostingEnsemble class.

    Args:
      X (np.array): The dataset
      y (np.array): The labels
      nb_trees (int): The total number of trees in the model.
      nb_trees_per_ensemble (int): The number of trees in each ensemble.
      test_size (float): Optional. The proportion of the dataset that should be
          used for testing vs. for training. Default is 25% testing,
          75% training.
      max_depth (int): Optional. The depth for the trees. Default is 6.
      privacy_budget (int): Optional. The privacy budget available for the
          model. Default is 1.
      learning_rate (float): Optional. The learning rate. Default is 0.1.
      shrinkage_rate (float): Optional. The shrinking rate for the trees in
          the ensemble. This controls how many rows will go into each tree's
          training set. Default is 0.1.
      threshold (int): Optional. Threshold for the loss function. For the
          square loss function (default), this is 1.
    """
    self.X = X
    self.y = y
    self.nb_trees = nb_trees
    self.nb_trees_per_ensemble = nb_trees_per_ensemble

    self.test_size = test_size
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=self.test_size)
    # Initializing predictions, as the mean value of y_train
    self.y_train_pred = np.array([self.y_train.mean()] * len(self.y_train))
    self.max_depth = max_depth
    self.privacy_budget = privacy_budget
    self.learning_rate = learning_rate
    self.shrinkage_rate = shrinkage_rate
    self.threshold = threshold
    self.ensembles = defaultdict(lambda: [])  # type: Dict[int, List[Any]]

  def Train(self) -> 'GradientBoostingEnsemble':
    """Train the ensembles of gradient boosted trees.

    Returns:
      GradientBoostingEnsemble: A GradientBoostingEnsemble object.
    """

    # Initialize ensemble id, so that we know which ensemble trees belong to
    ensemble_id = 0

    # Train all trees
    for tree_index in range(self.nb_trees):
      current_tree_for_ensemble = tree_index % self.nb_trees_per_ensemble
      if current_tree_for_ensemble == 0:
        # Initialize the dataset and corresponding labels for the current
        # ensemble
        X = np.copy(self.X_train)
        y = np.copy(self.y_train)
        # Update the ensemble index
        ensemble_id += 1

      # Compute the number of rows that the current tree will use for training
      number_of_rows = int((len(self.X_train) * self.shrinkage_rate * math.pow(
        (1 - self.shrinkage_rate), current_tree_for_ensemble)) / (1 - math.pow(
        (1 - self.shrinkage_rate), self.nb_trees_per_ensemble)))
      if number_of_rows == 0:
        continue

      # Select <number_of_rows> rows at random from the ensemble dataset
      rows = np.random.randint(len(X), size=number_of_rows)
      X_tree = X[rows, :]
      y_tree = y[rows]
      y_pred_tree = self.y_train_pred[rows]

      # Remove the selected rows from the ensemble's dataset
      X = np.delete(X, rows, axis=0)
      y = np.delete(y, rows)

      # Compute the residuals between the predictions and true values
      residuals = -1 * self.ComputeGradientForLossFunction(y_tree, y_pred_tree)

      # Fit a decision tree on the residuals
      tree = DecisionTreeRegressor(max_depth=self.max_depth)
      tree.fit(X_tree, residuals)

      # Update the predictions
      self.y_train_pred += self.learning_rate * tree.predict(self.X_train)

      # Add the tree to its corresponding ensemble
      self.ensembles[ensemble_id].append(tree)

    return self

  def Predict(self) -> Tuple[np.array, np.array]:
    """Predict values from the ensemble of gradient boosted trees.

    Returns:
      Tuple[np.array, np.array]: A tuple containing the true values and the
          predictions.
    """

    # Compute the predictions for each ensemble
    ensembles_predictions = []
    for trees in self.ensembles.values():
      y_pred = np.array([self.y_train.mean()] * len(self.y_test))
      for tree in trees:
        y_pred += self.learning_rate * tree.predict(self.X_test)
      ensembles_predictions.append(y_pred)

    # Take the mean value of predictions across ensembles as final prediction
    # and return the true values and the predicted values
    return self.y_test, np.mean(ensembles_predictions, axis=0)

  @staticmethod
  def ComputeGradientForLossFunction(y: np.array, y_pred: np.array) -> np.array:
    """Compute the gradient of the loss function.

    Args:
      y (np.array): The true values.
      y_pred (np.array): The predictions.

    Returns:
      (np.array): The gradient of the loss function.
    """
    return -1 * (y - y_pred)
