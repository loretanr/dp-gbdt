# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Main file for regression tasks."""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from prototyping.boosting.model import GradientBoostingEnsemble

PATH_TO_DATA = 'data/src/abalone.data'

# The number of time to repeat the experiment to get an average accuracy
MAX_REPEAT = 30

# The criterion on which to split trees
# Possible values: 'entropy', 'gini'
CRITERION = 'gini'

if __name__ == '__main__':
  # Read the data and isolate the features we're interested in
  data = pd.read_csv(PATH_TO_DATA, names=[
      'sex', 'length', 'diameter', 'height', 'whole weight',
      'shucked weight', 'viscera weight', 'shell weight', 'rings'])

  total_accuracy = 0.

  for label in 'MFI':
    data[label] = data['sex'] == label
  del data['sex']
  y = data.rings.values
  del data['rings']
  X = data.values.astype(np.float)

  # Normalizing data
  scaler = MinMaxScaler(feature_range=(-1, 1))
  scaler.fit(y.reshape(-1, 1))
  y = scaler.transform(y.reshape(-1, 1)).reshape(-1)

  # To compute error on the actual data range
  multiplier = ((scaler.data_max_ - scaler.data_min_) / 2)[0]

  for i in range(MAX_REPEAT):
    # Gradient boosting tree
    model = GradientBoostingEnsemble(X, y, 40, 5)
    y_true, y_pred = model.Train().Predict()

    # Root mean square error
    accuracy = metrics.mean_squared_error(
        y_true, y_pred, squared=False) * multiplier
    total_accuracy += accuracy
    print('Accuracy for run {0:d}: {1:f} (boosting)'.format(i, accuracy))
    print('-----------------------------------------------------------')

  average_accuracy = total_accuracy / MAX_REPEAT
  print('Average accuracy on {0:d} runs: {1:f} (boosting)'.format(
      MAX_REPEAT, average_accuracy))
