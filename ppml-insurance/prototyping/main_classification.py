# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Main file for classification tasks."""

from typing import List, Tuple, Union

from prototyping.utils import parser, utils
from prototyping.scikit import model as scikit_model
from prototyping.scikit_manual import model as scikit_manual_model
from prototyping.manual import model as manual_model

PATH_TO_DATA = 'data/src/HackerRank-Developer-Survey-2018-Numeric.csv'

# Problem statement: given the age at which one started coding, the age they
# are now, whether or not they learned to code in uni or by themselves and
# the current professional language they use in their day to day job,
# are they non-binary, female or male.

# Students are excluded from the dataset.
# NULL values for other features will be assigned a default value.
FILTER = [('q8Student', 1)]  # type: List[Tuple[str, Union[str, int]]]


FEATURES = ['q1AgeBeginCoding', 'q2Age', 'q3Gender', 'q6LearnCodeUni',
            'q6LearnCodeSelfTaught', 'q22LangProfC', 'q22LangProfCPlusPlus',
            'q22LangProfJava', 'q22LangProfPython', 'q22LangProfRuby',
            'q22LangProfJavascript', 'q22LangProfCSharp', 'q22LangProfGo',
            'q22LangProfScala', 'q22LangProfPerl', 'q22LangProfSwift',
            'q22LangProfPascal', 'q22LangProfClojure', 'q22LangProfPHP',
            'q22LangProfHaskell', 'q22LangProfLua', 'q22LangProfR',
            'q22LangProfOther']

# The target label we are interested in
# 1: Male
# 2: Female
# 3: Non-binary
LABEL = 'q3Gender'

# Limit the amount of data to use for the training and classification tasks.
# For this exercise we limit the amount of surveys that we use to 200. The
# 200 participants will be chosen at random in the dataset.
LIMIT_PARTICIPANTS = 200

# The number of time to repeat the experiment to get an average accuracy
MAX_REPEAT = 30

# The criterion on which to split trees
# Possible values: 'entropy', 'gini'
CRITERION = 'gini'

if __name__ == '__main__':
  # Read the data and isolate the features we're interested in
  data = parser.Parser(PATH_TO_DATA).Filter(FILTER).Keep(FEATURES).Data(
      fill_null_values=True)

  total_accuracy_scikit = 0.
  total_accuracy_scikit_manual = 0.
  total_accuracy_manual = 0.

  for i in range(MAX_REPEAT):
    # Only select a few participants, at random
    x = data.sample(n=LIMIT_PARTICIPANTS).reset_index(drop=True)

    # Get the label column from the data (i.e. gender in our case), and remove
    # it from the features
    y = x[LABEL]
    del x[LABEL]

    # Scikit implementation evaluation
    model = scikit_model.RandomForest(
        x, y, criterion=CRITERION)
    real_data, predictions = model.Train().Predict()
    accuracy = utils.Accuracy(real_data, predictions)
    total_accuracy_scikit += accuracy
    print('Accuracy for run {0:d}: {1:f} (scikit impl.)'.format(i, accuracy))

    # Own implementation evaluation, with scikit's Decision Trees
    model = scikit_manual_model.RandomForest(  # type: ignore
        x, y, criterion=CRITERION)
    real_data, predictions = model.Train().Predict()
    accuracy = utils.Accuracy(real_data, predictions)
    total_accuracy_scikit_manual += accuracy
    print('Accuracy for run {0:d}: {1:f} (scikit manual impl.)'.format(
        i, accuracy))

    # Own implementation
    model = manual_model.RandomForest(  # type: ignore
        x, y, random_features=True, criterion=CRITERION)
    real_data, predictions = model.Train().Predict()
    accuracy = utils.Accuracy(real_data, predictions)
    total_accuracy_manual += accuracy
    print('Accuracy for run {0:d}: {1:f} (manual impl.)'.format(
        i, accuracy))

    print('-----------------------------------------------------------')

  average_accuracy = total_accuracy_scikit / MAX_REPEAT
  print('Average accuracy on {0:d} runs: {1:f} (scikit impl.)'.format(
      MAX_REPEAT, average_accuracy))
  average_accuracy = total_accuracy_scikit_manual / MAX_REPEAT
  print('Average accuracy on {0:d} runs: {1:f} (scikit manual impl.)'.format(
      MAX_REPEAT, average_accuracy))
  average_accuracy = total_accuracy_manual / MAX_REPEAT
  print('Average accuracy on {0:d} runs: {1:f} (manual impl.)'.format(
      MAX_REPEAT, average_accuracy))
