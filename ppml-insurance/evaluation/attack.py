# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Membership inference attack."""

from typing import Any, Optional

# pylint: disable=line-too-long
from art.attacks.inference.attribute_inference import \
  AttributeInferenceWhiteBoxLifestyleDecisionTree, \
  AttributeInferenceWhiteBoxDecisionTree
from art.attacks.inference.membership_inference import \
  MembershipInferenceBlackBoxRuleBased, MembershipInferenceBlackBox
from art.estimators.classification.scikitlearn import \
  ScikitlearnGradientBoostingClassifier, ScikitlearnDecisionTreeClassifier

from sklearn.model_selection import train_test_split
# pylint: enable=line-too-long

import numpy as np

from data.parser.parser import Parser
from evaluation.estimator import DPGBDT


class AttackClassifier:
  """Wrapper for membership inference attacks."""
  # pylint: disable=invalid-name, redefined-outer-name
  def __init__(self,
               classifier: Any,
               X_train: np.ndarray,
               X_test: np.ndarray,
               y_train: np.array,
               y_test: np.array,
               attack: str = 'membership_inference') -> None:
    """Initialize the attack classifier.

    Args:
      classifier (Any): A classifier.
      X_train (np.ndarray): The training dataset.
      X_test (np.ndarray): The testing dataset.
      y_train (np.array): The training labels.
      y_test (np.array): The testing labels.
      attack (str): The attack to perform on the classifier. Default is
          membership inference attack.
    """
    self.target_classifier = classifier
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test
    self.attack = attack

  def Attack(self, attack_feature: Optional[int] = None, lifestyle: bool = True) -> Any:
    """Runs the attack and returns its accuracy."""
    bla = self.target_classifier.fit(self.X_train, self.y_train)
    if self.attack == 'membership_inference_label':
      classifier = ScikitlearnGradientBoostingClassifier(self.target_classifier)
      attack_classifier = MembershipInferenceBlackBoxRuleBased(classifier)
      # infer attacked feature
      inferred_train = attack_classifier.infer(self.X_train, self.y_train)
      inferred_test = attack_classifier.infer(self.X_test, self.y_test)
      # check accuracy
      train_acc = np.sum(inferred_train) / len(inferred_train)
      test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
      attack_accuracy = (train_acc * len(inferred_train) + test_acc * len(
          inferred_test)) / (len(inferred_train) + len(inferred_test))
      return train_acc, test_acc, attack_accuracy
    elif self.attack == 'membership_inference':
      classifier = ScikitlearnGradientBoostingClassifier(self.target_classifier)
      attack_classifier = MembershipInferenceBlackBox(classifier, attack_model_type='rf')

      # train attack model
      attack_classifier.fit(self.X_train, self.y_train,
                            self.X_test, self.y_test)
      # get inferred values
      inferred_train_bb = attack_classifier.infer(self.X_train, self.y_train)
      inferred_test_bb = attack_classifier.infer(self.X_test, self.y_test)

      # check accuracy
      train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
      test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
      attack_accuracy = (train_acc * len(inferred_train_bb) + test_acc * len(
          inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
      return train_acc, test_acc, attack_accuracy
    elif self.attack == 'attribute_inference':
      if not isinstance(self.target_classifier, DPGBDT):
        raise TypeError('Attribute inference attack only implemented for '
                        'DPGBDT')
      X_train_feature = self.X_train[:, attack_feature].copy().reshape(-1, 1)
      X_train_for_attack = np.delete(self.X_train, attack_feature, 1)
      attack_classifier = ScikitlearnDecisionTreeClassifier(
          self.target_classifier)
      if lifestyle:
        wb_attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(
          attack_classifier, attack_feature=attack_feature)
      else:
        wb_attack = AttributeInferenceWhiteBoxDecisionTree(
            attack_classifier, attack_feature=attack_feature)
      # get original model's predictions
      x_train_predictions = np.array(
        [np.argmax(arr) for arr in attack_classifier.predict(
            self.X_train)]).reshape(-1, 1)
      values = list(set(self.X_train[:, attack_feature]))
      priors = []
      for v in values:
        priors.append(len(
            self.X_train[self.X_train[:, attack_feature] == v]) / len(
                self.X_train))
      # get inferred values
      inferred_train_wb1 = wb_attack.infer(
        X_train_for_attack,
        x_train_predictions,
        values=values, priors=priors)
      # check accuracy
      train_acc = np.sum(
        inferred_train_wb1 == np.around(
          X_train_feature, decimals=8).reshape(1, -1)) / len(inferred_train_wb1)
      return train_acc, None, None
    return None

  def MultiProcessAttack(self, data: str, n_repeat: int = 3) -> Any:
    """For multi-processing.

    Args:
      n_repeat (int): Number of times to repeat the experiment.
      data (str): Configuration of the model being attacked. Format string
          ready for scores and stdev.
    """
    attack_scores, train_scores, test_scores = [], [], []
    for _ in range(n_repeat):
      train_acc, test_acc, attack_accuracy = self.Attack()
      attack_scores.append(attack_accuracy * 100)
      train_scores.append(train_acc * 100)
      test_scores.append(test_acc * 100)
    gap = np.asarray(train_acc) - np.asarray(test_acc)
    return data.format(
        np.mean(train_scores), np.std(train_scores),
        np.mean(test_scores), np.std(test_scores),
        np.mean(gap), np.std(gap),
        np.mean(attack_scores), np.std(attack_scores))


if __name__ == '__main__':
  parser = Parser(dataset='synthetic_A',
                  task='classification',
                  bins=3)
  x, y, cat_idx, num_idx, task = parser.Parse(n_rows=5000)
  X_train, X_test, y_train, y_test = train_test_split(
      x, y, random_state=0, test_size=0.7)
  train_acc, test_acc, attack_accuracy = AttackClassifier(
      DPGBDT(privacy_budget=0.,
             nb_trees=5,
             nb_trees_per_ensemble=5,
             max_depth=15,
             learning_rate=0.5,
             n_classes=len(set(y_train)),
             gradient_filtering=True,
             leaf_clipping=True,
             num_idx=num_idx,
             cat_idx=cat_idx,
             verbosity=0),  # type: ignore
      X_train,
      X_test,
      y_train,
      y_test,
      attack='membership_inference_label').Attack()
  if attack_accuracy is not None:
    print('Attack accuracy: ', attack_accuracy)
    print('Prediction accuracy (train): ', train_acc * 100)
    print('Prediction accuracy (test): ', test_acc * 100)
  else:
    print('Attack accuracy: ', train_acc)
