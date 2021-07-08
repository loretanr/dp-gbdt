# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Utils methods used by models."""

from typing import Tuple, Any, TYPE_CHECKING

from sklearn import metrics
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
  import pandas as pd


def Split(x: 'pd.DataFrame',
          y: 'pd.DataFrame',
          test_size: float) -> Tuple[Any, ...]:
  """Split a dataset into training and testing.

  Args:
    x (pd.DataFrame): The dataframe containing the features.
    y (pd.DataFrame): The dataframe containing the labels.
    test_size (float): The percentage to use for splitting the training and
        testing data.

  Returns:
    Tuple[Any, ...]: A split for the training and testing data.
  """
  x_train, x_test, y_train, y_test = train_test_split(
      x, y, test_size=test_size)
  return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()  # pylint: disable=line-too-long


def Accuracy(y_test: 'pd.DataFrame', y_pred: 'pd.DataFrame') -> float:
  """Compute the accuracy between the predictions and the actual data.

  Args:
    y_test (pd.DataFrame): Real values for the data.
    y_pred (pd.DataFrame): Predictions for the data.

  Returns:
    float: An accuracy measure for the predictions.
  """
  accuracy = metrics.accuracy_score(y_test, y_pred)  # type: float
  return accuracy
