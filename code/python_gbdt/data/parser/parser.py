# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Parser for the dataset."""

import os
import ssl
import warnings
from typing import Optional, Any, List
from urllib.request import urlretrieve

import pandas as pd
import numpy as np


class Parser:
  """Implement a parser class for multiple dataset.

  Attributes:
    dataset (str): The name of the dataset.
    paths (Dict[str, str]): A dictionary that maps the dataset to its source
        file path.
    columns (Dict[str, List[str]): A dictionary that maps the dataset to
        their features.
    data (pd.Dataframe): The parsed dataset.
  """
  # pylint: disable=invalid-name

  def __init__(self, dataset: Optional[str] = None,
               objective: str = 'loss',
               task: str = 'regression',
               bins: int = 5,
               binary_classification: Optional[bool] = False) -> None:
    """Initialize the parser class.

    Args:
      dataset (str): Optional. The dataset to use. Currently supported values
          are 'abalone' and 'questionnaires'.
      objective (str): One of ['loss', 'cost']. Only for synthetic datasets. If
          'loss', then the label will be a probability for a loss to occur
          for a given company. If 'cost', then the label will be a potential
          cost should a loss occur.
      task (str): One of ['regression', 'classification']. Only for synthetic
          datasets. This will change the target of the dataset.
      bins (int): How many labels to create for the synthetic datasets if the
          task is a classification task.
      binary_classification (bool): Optional. If True, binary labels are
          encoded -1 and 1 instead of 0 and 1.

    Raises:
      NotImplementedError: If the requested dataset has not been implemented
          yet.
    """

    # pylint: disable=protected-access
    ssl._create_default_https_context = ssl._create_unverified_context
    # pylint: enable=protected-access

    if dataset not in ['abalone', 'questionnaires', 'bcw', 'adult', 'yearMSD',
                       'synthetic_A', 'synthetic_B', 'synthetic_C',
                       'synthetic_D']:
      raise NotImplementedError('Parser not implemented for this dataset.')

    self.dataset = dataset
    self.objective = objective
    self.task = task
    self.bins = bins
    self.binary_classification = binary_classification
    self.paths = {
        'abalone': 'datasets/real/abalone.data',
        'questionnaires': 'datasets/real/questionnaires.csv',
        'bcw': 'datasets/real/breast-cancer-wisconsin.data',
        'synthetic_A': 'src/synthetic/synthetic_A.csv',
        'synthetic_B': 'src/synthetic/synthetic_B.csv',
        'synthetic_C': 'src/synthetic/synthetic_C.csv',
        'synthetic_D': 'src/synthetic/synthetic_D.csv'
    }
    self.columns = {
        'abalone': ['sex', 'length', 'diameter', 'height', 'whole weight',
                    'shucked weight', 'viscera weight', 'shell weight',
                    'rings'],
        # https://www.kaggle.com/lucasgreenwell/depression-anxiety-stress-scales-responses
        'questionnaires': [feature for question in
                           ['Q{0!s}A,Q{0!s}I,Q{0!s}E'.format(i).split(',') for i
                           in range(1, 43)] +
                          ['country,source,introelapse,testelapse,'
                           'surveyelapse'.split(',')] +
                          ['TIPI{0!s}'.format(i).split() for i in range(1, 11)]
                          + ['VCL{0!s}'.format(i).split() for i in range(1, 17)]
                          + ['education,urban,gender,engnat,age,screensize,'
                            'uniquenetworklocation,hand,religion,orientation,'
                            'race,voted,married,familysize,major'.split(',')]
                           for feature in question],
        'bcw': ['id', 'clump thickness', 'cell size', 'cell shape',
                'marginal adhesion', 'epithelial cell size', 'bare nuclei',
                'bland chromatin', 'normal nucleoli', 'mitoses', 'class']
    }
    real_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), self.paths.get(self.dataset, ''))
    if self.dataset in ['abalone', 'bcw']:
      self.data = pd.read_csv(real_path, names=self.columns.get(self.dataset))
    elif self.dataset == 'questionnaires':
      self.data = pd.read_csv(real_path, names=self.columns.get(self.dataset),
                              skiprows=[0], sep='\t')
    elif self.dataset.startswith('synthetic'):
      self.data = pd.read_csv(real_path)

  def Parse(self,
            n_rows: Optional[int] = None) -> Any:
    """Prepare the dataset for learning tasks.

    Args:
      n_rows (int): Optional. Number of rows to return from the dataset.
          Default returns all rows.

    Returns:
      Tuple[np.ndarray, np.ndarray]: Parsed X and y.

    Raises:
      RuntimeError: If the file with best features couldn't be parsed but is
          still requested by the user.
    """
    if self.dataset == 'abalone':
      return self.get_abalone(n_rows=n_rows)
    if self.dataset == 'questionnaires':
      return self.get_questionnaires(n_rows=n_rows)
    if self.dataset == 'bcw':
      return self.get_bcw(n_rows=n_rows)
    if self.dataset == 'adult':
      return self.get_adult(n_rows=n_rows)
    if self.dataset == 'yearMSD':
      return self.get_year(n_rows=n_rows)
    if self.dataset.startswith('synthetic'):
      return self.get_synthetic(n_rows=n_rows)
    return None

  @staticmethod
  def get_year(n_rows: Optional[int] = None) -> Any:
    """Return the YearPredictions dataset."""
    task = 'regression'
    get_year_url = ('https://archive.ics.uci.edu/ml/machine-learning'
                    '-databases/00203/YearPredictionMSD.txt.zip')
    filename = 'src/real/YearPredictionMSD.txt.zip'
    filename_real_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), filename)
    if not os.path.isfile(filename_real_path):
      urlretrieve(get_year_url, filename_real_path)
    year = pd.read_csv(filename_real_path, header=None, nrows=n_rows)
    X = year.iloc[:, 1:].values
    y = year.iloc[:, 0].values
    categorical_indices = []  # type: List[int]
    numerical_indices = list(range(0, X.shape[1]))  # All
    return X, y, categorical_indices, numerical_indices, task

  def get_bcw(self,
              n_rows: Optional[int] = None) -> Any:
    """Return the breast cancer dataset."""
    task = 'classification'
    # Drop rows with missing information
    self.data = self.data[~(self.data.astype(str) == '?').any(1)]
    # Drop ID
    del self.data['id']
    if n_rows:
      self.data = self.data.head(n_rows)
    y = self.data['class']
    # Re-encode label
    y = np.where(y[:] == 4, 1, 0)
    del self.data['class']
    X = self.data.values.astype(np.float)
    categorical_indices = []  # type: List[int]
    numerical_indices = list(range(0, X.shape[1]))  # All
    return X, y, categorical_indices, numerical_indices, task

  def get_abalone(
      self,
      n_rows: Optional[int] = None) -> Any:
    """Return the abalone dataset."""
    task = 'regression'
    # Re-encode gender information (if not already done)
    if isinstance(self.data['sex'].values[0], str):
      gender = []
      for i in range(len(self.data['sex'].values)):
        if self.data['sex'].values[i] == 'M':
          value = 1
        elif self.data['sex'].values[i] == 'F':
          value = 2
        else:
          value = 3
        gender.append(value)
      self.data['sex'] = gender
    current_data = self.data.head(n_rows)
    y = current_data.rings.values.astype(np.float)
    # del self.data['rings']
    X = (current_data.drop('rings', axis=1)).values.astype(np.float)
    categorical_indices = [0]  # Sex
    numerical_indices = list(range(1, X.shape[1]))  # Other attributes
    return X, y, categorical_indices, numerical_indices, task

  def get_questionnaires(
      self,
      n_rows: Optional[int] = None) -> Any:
    """Return the questionnaires dataset."""
    task = 'regression'
    _range = range(1, 43)
    features = ['Q{0!s}A'.format(i) for i in _range[1:]]  # type: ignore
    values = ['Q{0!s}A'.format(_range[0])]  # type: ignore
    X = self.data[features].values.astype(np.float)
    y = self.data[values].values.astype(np.float)
    if n_rows:
      X = X[:n_rows]
      y = y[:n_rows]
    categorical_indices = list(range(0, X.shape[1]))  # All
    numerical_indices = []  # type: List[int]
    return X, y, categorical_indices, numerical_indices, task


  def get_adult(self, n_rows: Optional[int] = None) -> Any:
    """Return the Adult dataset."""
    task = 'classification'
    warnings.filterwarnings("ignore")

    get_adult_train_url = ('https://archive.ics.uci.edu/ml/machine-learning'
                           '-databases/adult/adult.data')
    get_adult_test_url = ('https://archive.ics.uci.edu/ml/machine-learning'
                          '-databases/adult/adult.test')

    train_filename = 'src/real/adult.data'
    test_filename = 'src/real/adult.test'
    train_real_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), train_filename)
    test_real_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), test_filename)

    if not os.path.isfile(train_real_path):
      urlretrieve(get_adult_train_url, train_real_path)
    if not os.path.isfile(test_real_path):
      urlretrieve(get_adult_test_url, test_real_path)

    adult = pd.read_csv(train_real_path, header=None, sep=', ')     # diabled test set for now
    # adult = adult.append(
    #     pd.read_csv(test_real_path, header=None, skiprows=1, sep=', '))
    # Drop weight info
    adult.drop(columns=[2], axis=1, inplace=True)
    # Drop rows with missing information
    adult = adult[~(adult.astype(str) == '?').any(1)]
    if n_rows:
      adult = adult[:n_rows]
    if not self.binary_classification:
      y = np.where(adult.iloc[:, -1] == '>50K', 1, 0)
    else:
      y = np.where(adult.iloc[:, -1] == '>50K', 1, -1)
    adult = adult.iloc[:, :-1]
    for idx, row in adult.iterrows():
      if str(row.values[-1]).strip() != 'United-States':
        adult.at[idx, 13] = 'Other'
    adult.columns = range(adult.shape[1])
    categorical_indices = adult.select_dtypes(
        include=['object']).columns.tolist()
    numerical_indices = adult.select_dtypes(
        exclude=['object']).columns.tolist()
    for column in adult:
      if column in categorical_indices:
        # adult[column] = pd.get_dummies(adult[column])
        adult[column] = adult[column].astype('category') # fixed
        adult[column] = adult[column].cat.codes
    X = adult.values
    return X, y, categorical_indices, numerical_indices, task


  def get_synthetic(self,
                      n_rows: Optional[int] = None) -> Any:
    """Return synthetic dataset."""
    # self.data['company_size'] = pd.get_dummies(self.data['company_size'])
    # self.data['company_sector'] = pd.get_dummies(self.data['company_sector'])
    self.data['company_size'] = self.data['company_size'].astype(
      'category').cat.codes
    self.data['company_sector'] = self.data['company_sector'].astype(
      'category').cat.codes
    if n_rows and n_rows != len(self.data):
      self.data = self.data.head(n=n_rows)
    X = self.data[self.data.columns[:-2]].values
    categorical_indices = list(range(0, X.shape[1]))  # All
    numerical_indices = []  # type: List[int]
    if self.objective == 'loss':
      y = self.data[self.data.columns[-2]].values
    else:
      y = self.data[self.data.columns[-1]].values
    if self.task == 'classification':
      y_sorted = np.sort(y)
      bins = [y_sorted[i*(int(np.floor(len(
          y_sorted)/self.bins)))] for i in range(self.bins)]
      y = np.digitize(y, bins).astype(np.int) - 1
      if self.binary_classification:
        y[y == 0] = -1
      # y = np.digitize(y, np.linspace(
      #   0., max(y), self.bins + 1), right=True).astype(np.int) - 1
    return X, y, categorical_indices, numerical_indices, self.task
