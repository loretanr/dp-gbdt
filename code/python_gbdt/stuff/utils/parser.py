# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Parse the HackerRank Developer Survey data set."""
import os
from typing import List, Union, Tuple

import pandas as pd


class Parser:
  """Parses a data set.

  This class parses a .csv file into a pandas dataframe and offers methods to
  filter out the resulting parsed df and perform other operations on the df
  to prepare data for use in ML methods.

  Attributes:
    path_to_data (str): A path to a dataset. It is assumed that the data are
        in a directory at the same level that utils/ is.
    parsed_data (pd.Dataframe): A pandas dataframe of the parsed data.
  """

  def __init__(self, path_to_data: str) -> None:
    """Initialize the parser class.

    Args:
      path_to_data (str): The path to the data.
    """
    self.path_to_data = path_to_data
    self.parsed_data = self.Parse()

  def Parse(self) -> pd.DataFrame:
    """Parse a .csv file and returns a pandas dataframe.

    Returns:
      pd.Dataframe: A pandas dataframe for the data to parse.

    Raises:
      FileNotFoundError: If the data file could not be found.
    """
    real_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), self.path_to_data)
    if not os.path.exists(real_path):
      raise FileNotFoundError('Please make sure that your data set is placed '
                              'at the same directory level as the "utils" '
                              'directory.')
    return pd.read_csv(real_path)

  def Filter(self,
             features: List[Tuple[str, Union[int, str]]]) -> 'Parser':
    """Filter rows from a dataframe which match a certain value for a
    certain feature.

    Args:
      features (List[Tuple[str, int|str]]): A list of tuples containing the
          feature (str) and the value (int|str) for that feature which we don't
          want in our data.

    Returns:
      Parser: A parser object.

    Raises:
      ValueError: If the requested features to be filtered out are not found in
          the dataframe.
    """
    data_features = self.parsed_data.columns
    for feature, value in features:
      if feature not in data_features:
        raise ValueError(
            'Requested feature to filter not found in the data set.')
      self.parsed_data = self.parsed_data[self.parsed_data[feature] != value]
    return self

  def Keep(self, features: List[str]) -> 'Parser':
    """Filter a dataframe so that it only contains the list of features
    passed in parameters.

    Args:
      features (List[str]): A list of features to keep (strings).

    Returns:
      Parser: A parser object.

    Raises:
      ValueError: If the requested feature is not found in the dataframe.
    """
    data_features = self.parsed_data.columns
    if not all(feature in data_features for feature in features):
      raise ValueError('Requested features not found in data set.')
    self.parsed_data = self.parsed_data[features]
    return self

  def FillNullValues(self) -> 'Parser':
    """Fill NULL values with the mode for each feature in the dataset.

    Returns:
      Parser: A parser object.
    """
    for column in self.parsed_data.columns:
      value_count = self.parsed_data[column].value_counts().index
      mode = value_count[0] if value_count[0] != '#NULL!' else value_count[1]
      self.parsed_data[column].replace('#NULL!', mode, inplace=True)
    return self

  def Data(self, fill_null_values: bool = False) -> pd.DataFrame:
    """Return the parsed data.

    Args:
      fill_null_values (bool): if True, replace Null values with a default
          value (here, the mode value for each feature). Default is False,
          i.e. do not fill Null values.
    Returns:
      pd.Dataframe: A pandas dataframe.
    """
    if fill_null_values:
      self.FillNullValues()
    return self.parsed_data
