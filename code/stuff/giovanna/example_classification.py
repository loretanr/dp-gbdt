import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from estimator import DPGBDT

if __name__ == '__main__':
  N = 500
  # Random normal distributed data (500 rows, 3 cols, 2 coordinates values each)
  X = np.random.normal(
      loc=[(-1, 1), (2, 5), (4, -4)],
      scale=[(1, 1), (0.5, 0.5), (1.5, 1.5)],
      size=(N, 3, 2),
  )
  # X: "reshape into 2 columns and whatever many rows"
  X = X.reshape((-1, 2))
  # y = col vector like [0,1,2,0,1,2,...]'
  y = np.arange(3)[np.newaxis, :].repeat(N, axis=0).reshape(-1)
  # basically we have data from 3 different normal distributions alternated in the rows of X and y
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

  model = DPGBDT(
      privacy_budget=0.1,
      n_classes=len(set(y_train)),
      nb_trees=50,
      nb_trees_per_ensemble=50,
      max_depth=3,
      use_3_trees=False,
      learning_rate=0.1,
  )
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print(classification_report(y_test, y_pred))
  print("Score %.4f" % np.mean(y_test == y_pred))

  y_pred = model.predict(X_train)
  print("Score %.4f (train)" % np.mean(y_train == y_pred))


"""
              precision    recall  f1-score   support

           0       0.64      0.86      0.73       218
           1       0.70      0.22      0.33       178
           2       0.76      0.92      0.83       204

    accuracy                           0.69       600
   macro avg       0.70      0.67      0.63       600
weighted avg       0.69      0.69      0.65       600

Score 0.6917
Score 0.6700 (train)
"""

# Precision = TP / (TP+FP)
# Recall (sensitivity) = TP / (TP+FN)
# f1-score = harmonic mean of precision and recall
# support = #samples of the true response that lie in that class
# Accuracy = TP+TN / (TP+FP+FN+TN)