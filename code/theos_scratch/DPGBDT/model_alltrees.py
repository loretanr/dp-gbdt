# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Implement ensemble of differentially private gradient boosted trees.

From: https://arxiv.org/pdf/1911.04209.pdf

no 2nd-split aka alltrees
noifbinary, leafclippipng
"""

import math
import operator
from collections import defaultdict
from queue import Queue
from typing import List, Any, Optional, Dict

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
# pylint: disable=line-too-long
from sklearn.ensemble._gb_losses import LeastSquaresError, \
  MultinomialDeviance, LossFunction, BinomialDeviance
# pylint: enable=line-too-long

from DPGBDT import logging

logging.SetUpLogger(__name__)
logger = logging.GetLogger(__name__)


class GradientBoostingEnsemble:
  """Implement gradient boosting ensemble of trees.

  Attributes:
    nb_trees (int): The total number of trees in the model.
    nb_trees_per_ensemble (int): The number of trees in each ensemble.
    max_depth (int): The depth for the trees.
    privacy_budget (float): The privacy budget available for the model.
    learning_rate (float): The learning rate.
    l2_threshold (int): Threshold for the loss function. For the square loss
        function (default), this is 1.
    l2_lambda (float): Regularization parameter for l2 loss function.
        For the square loss function (default), this is 0.1.
    trees (List[List[DifferentiallyPrivateTree]]): A list of k-classes DP trees.
  """
  # pylint: disable=invalid-name, too-many-arguments, unused-variable

  def __init__(self,
               nb_trees: int,
               nb_trees_per_ensemble: int,
               n_classes: Optional[int] = None,
               binary_classification: Optional[bool] = False,
               max_depth: int = 6,
               privacy_budget: float = 1.0,
               learning_rate: float = 0.1,
               early_stop: int = 5,
               max_leaves: Optional[int] = None,
               min_samples_split: int = 2,
               gradient_filtering: bool = False,
               leaf_clipping: bool = False,
               balance_partition: bool = True,
               use_bfs: bool = False,
               use_3_trees: bool = False,
               use_decay: bool = False,
               cat_idx: Optional[List[int]] = None,
               num_idx: Optional[List[int]] = None,
               test_size: float = 0.3,
               verbosity: int = -1) -> None:
    """Initialize the GradientBoostingEnsemble class.

    Args:
      nb_trees (int): The total number of trees in the model.
      nb_trees_per_ensemble (int): The number of trees in each ensemble.
      n_classes (int): Number of classes. Triggers regression (None) vs
          classification.
      binary_classification (bool): Optional. Whether or not to use the
          regression model to perform a binary classification. Default is False.
      max_depth (int): Optional. The depth for the trees. Default is 6.
      privacy_budget (float): Optional. The privacy budget available for the
          model. Default is 1.0.
      learning_rate (float): Optional. The learning rate. Default is 0.1.
      early_stop (int): Optional. If the rmse doesn't decrease for <int>
          consecutive rounds, abort training. Default is 5.
      max_leaves (int): Optional. The max number of leaf nodes for the trees.
          Tree will grow in a best-leaf first fashion until it contains
          max_leaves or until it reaches maximum depth, whichever comes first.
      min_samples_split (int): Optional. The minimum number of samples required
          to split an internal node. Default is 2.
      gradient_filtering (bool): Optional. Whether or not to perform gradient
          based data filtering during training. Default is False.
      leaf_clipping (bool): Optional. Whether or not to clip the leaves
          after training. Default is False.
      balance_partition (bool): Optional. Balance data repartition for training
          the trees. The default is True, meaning all trees within an ensemble
          will receive an equal amount of training samples. If set to False,
          each tree will receive <x> samples where <x> is given in line 8 of
          the algorithm in the author's paper.
      use_bfs (bool): Optional. If max_leaves is specified, then this is
          automatically True. This will build the tree in a BFS fashion instead
          of DFS. Default is False.
      use_3_trees (bool): Optional. If True, only build trees that have 3
          nodes, and then assemble nb_trees based on these sub-trees, at random.
          Default is False.
      use_decay (bool): Optional. If True, internal node privacy budget has a
          decaying factor.
      cat_idx (List): Optional. List of indices for categorical features.
      num_idx (List): Optional. List of indices for numerical features.
      test_size (float): Optional. Percentage of instances to use for
          validation. Default is 30%.
      verbosity (int): Optional. Verbosity level for debug messages. Default
          is -1, meaning only warnings and above are displayed. 0 is info,
          1 is debug.
      """
    self.nb_trees = nb_trees
    self.nb_trees_per_ensemble = nb_trees_per_ensemble
    self.max_depth = max_depth
    # Too high a privacy budget generates a bunch of overflows in the
    # exponential mechanism. Better performances with lower orders of
    # magnitudes. Note that if the aim is to disable DP, then the budget
    # should be set to 0.
    self.privacy_budget = privacy_budget if privacy_budget <= 1000 else 1000
    if self.privacy_budget > 100:
      logger.warning('High privacy budget detected. If the aim is to '
                     'deactivate differential privacy, then please set '
                     'privacy_budget to 0. The budget has an upper of 1000 in '
                     'all other cases.')
    self.learning_rate = learning_rate
    self.early_stop = early_stop
    self.max_leaves = max_leaves
    self.min_samples_split = min_samples_split
    self.gradient_filtering = gradient_filtering
    self.leaf_clipping = leaf_clipping
    self.balance_partition = balance_partition
    self.use_bfs = use_bfs
    self.use_3_trees = use_3_trees
    self.use_decay = use_decay
    self.cat_idx = cat_idx
    self.num_idx = num_idx
    self.verbosity = verbosity
    self.trees = []  # type: List[List[DifferentiallyPrivateTree]]

    self.binary_classification = binary_classification

    # classification vs regression
    if not n_classes:
      self.loss_ = LeastSquaresError()  # type: LossFunction
    else:
      if n_classes == 2:
        if self.binary_classification:  # Through regression model
          self.loss_ = LeastSquaresError()
        else:  # Through classification model
          self.loss_ = BinomialDeviance(n_classes)
      else:
        self.loss_ = MultinomialDeviance(n_classes)

    self.init_ = self.loss_.init_estimator()

    # Loss parameters
    self.l2_threshold = 1.0
    self.l2_lambda = 0.1

    # Initial score
    self.init_score = None

    self.use_dp = privacy_budget > 0.  # Use differential privacy or not
    if not self.use_dp:
      logger.warning('! ! ! Differential privacy disabled ! ! !')

    if self.use_3_trees and self.use_bfs:
      # Since we're building 3-node trees it's the same anyways.
      self.use_bfs = False

    if self.verbosity <= -1:
      logger.setLevel(logging.WARNING)
    elif self.verbosity == 0:
      logger.setLevel(logging.INFO)
    else:
      logger.setLevel(logging.DEBUG)

    # This handles attribute comparison depending on the attribute's nature
    self.feature_to_op = defaultdict(
        lambda: (operator.lt, operator.ge))  # type: Dict[int, Any]
    if self.cat_idx:
      for feature_index in self.cat_idx:
        self.feature_to_op[feature_index] = (operator.eq, operator.ne)

    self.test_size = test_size

  def Train(self,
            X: np.array,
            y: np.array) -> 'GradientBoostingEnsemble':
    """Train the ensembles of gradient boosted trees.

    Args:
      X (np.array): The features.
      y (np.array): The label.

    Returns:
      GradientBoostingEnsemble: A GradientBoostingEnsemble object.
    """

    # Init gradients
    self.init_.fit(X, y)
    self.init_score = self.loss_.get_init_raw_predictions(
        X, self.init_)  # (n_samples, K)
    logger.debug('Training initialized with score: {}'.format(self.init_score))
    update_gradients = True

    X_train, X_test, y_train, y_test = train_test_split(      # not using these sets
    X, y, test_size=self.test_size, random_state=0)       # resp. not rejecting on line 383

    # Number of ensembles in the model
    nb_ensembles = int(np.ceil(self.nb_trees / self.nb_trees_per_ensemble))
    logger.info('Model will have {0:d} ensembles'.format(nb_ensembles))

    # Privacy budget allocated to all trees in each ensemble
    tree_privacy_budget = np.divide(self.privacy_budget, nb_ensembles)
    # K trees
    if self.loss_.is_multi_class:
      tree_privacy_budget /= self.loss_.K

    prev_score = np.inf

    # Train all trees
    for tree_index in range(self.nb_trees):
      # Compute sensitivity
      delta_g = 3 * np.square(self.l2_threshold)
      delta_v = min(self.l2_threshold / (1 + self.l2_lambda),
                    2 * self.l2_threshold * math.pow(
                      (1 - self.learning_rate), tree_index))

      current_tree_for_ensemble = tree_index % self.nb_trees_per_ensemble
      if current_tree_for_ensemble == 0:
        # Initialize the dataset and the gradients
        X_ensemble = np.copy(X)       # take X and y here for alltrees
        y_ensemble = np.copy(y)
        prev_score = np.inf
        update_gradients = True
        # gradient initialization will happen later in the per-class-loop

      if self.use_dp:
        # Compute the number of rows that the current tree will use for training
        if self.balance_partition:
          # All trees will receive same amount of samples
          if self.nb_trees % self.nb_trees_per_ensemble == 0:
            # Perfect split
            number_of_rows = int(len(X) / self.nb_trees_per_ensemble)
          else:
            # Partitioning data across ensembles
            if np.ceil(tree_index / self.nb_trees_per_ensemble) == np.ceil(
                self.nb_trees / self.nb_trees_per_ensemble):
              number_of_rows = int(len(X) / (
                  self.nb_trees % self.nb_trees_per_ensemble))
            else:
              number_of_rows = int(len(X) / self.nb_trees_per_ensemble) + int(
                  len(X) / (self.nb_trees % self.nb_trees_per_ensemble))
        else:
          # Line 8 of Algorithm 2 from the paper
          number_of_rows = int((len(X) * self.learning_rate * math.pow(
            (1 - self.learning_rate), current_tree_for_ensemble)) / (
                1 - math.pow((
                    1 - self.learning_rate), self.nb_trees_per_ensemble)))

        # If using the formula from the algorithm, some trees may not get
        # samples. In that case we skip the tree and issue a warning. This
        # should hint the user to change its parameters (likely the ensembles
        # are too unbalanced)
        if number_of_rows == 0:  # pylint: disable=no-else-continue
          logger.warning('The choice of trees per ensemble vs. the total number'
                         ' of trees is not balanced properly; some trees will '
                         'not get any training samples. Try using '
                         'balance_partition=True or change your parameters.')
          continue
        elif number_of_rows > len(X_ensemble):
          number_of_rows = len(X_ensemble)

        # Select <number_of_rows> rows at random from the ensemble dataset
        rows = np.random.choice(range(len(X_ensemble)),
                                size=number_of_rows,
                                replace=False)
        X_tree = X_ensemble[rows, :]
        y_tree = y_ensemble[rows]

        # train for each class a separate tree on the same rows.
        # In regression or binary classification, K has been set to one.
        k_trees = []  # type: List[DifferentiallyPrivateTree]
        for kth_tree in range(self.loss_.K):
          if tree_index == 0:
            # First tree, start with initial scores (mean of labels)
            assert self.init_score is not None
            gradients = self.ComputeGradientForLossFunction(
                y, self.init_score[:len(y)], kth_tree)
          else:
            # Update gradients of all training instances on loss l
            if update_gradients:
              gradients = self.ComputeGradientForLossFunction(
                  y_ensemble, self.Predict(
                      X_ensemble), kth_tree)  # type: ignore

          assert gradients is not None
          gradients_tree = gradients[rows]

          # Gradient based data filtering
          if self.gradient_filtering:
            gradients_tree[
                gradients_tree > self.l2_threshold] = self.l2_threshold
            gradients_tree[
                gradients_tree < -self.l2_threshold] = -self.l2_threshold

          logger.debug('Tree {0:d} will receive a budget of epsilon={1:f} and '
                       'train on {2:d} instances.'.format(
              tree_index, tree_privacy_budget, len(X_ensemble)))
          # Fit a differentially private decision tree
          tree = DifferentiallyPrivateTree(
              tree_index,
              self.learning_rate,
              self.l2_threshold,
              self.l2_lambda,
              tree_privacy_budget,
              delta_g,
              delta_v,
              self.loss_,
              leaf_clipping=self.leaf_clipping,   # added leaf_clipping, which was missing
              max_depth=self.max_depth,
              max_leaves=self.max_leaves,
              min_samples_split=self.min_samples_split,
              use_bfs=self.use_bfs,
              use_3_trees=self.use_3_trees,
              use_decay=self.use_decay,
              cat_idx=self.cat_idx,
              num_idx=self.num_idx)
          # in multi-class classification, the target has to be binary
          # as each tree is a per-class regressor
          y_target = ((y_tree == kth_tree).astype(np.float64)
                      if self.loss_.is_multi_class
                      else y_tree)
          tree.Fit(X_tree, y_target, gradients_tree)

          # Add the tree to its corresponding ensemble
          k_trees.append(tree)
      else:
        # Fit a normal decision tree
        k_trees = []
        for kth_tree in range(self.loss_.K):
          if tree_index == 0:
            # First tree, start with initial scores (mean of labels)
            assert self.init_score is not None
            gradients = self.ComputeGradientForLossFunction(
              y, self.init_score[:len(y)], kth_tree)
          else:
            # Update gradients of all training instances on loss l
            if update_gradients:
              gradients = self.ComputeGradientForLossFunction(
                y, self.Predict(X), kth_tree)  # type: ignore
          tree = DifferentiallyPrivateTree(
              tree_index,
              self.learning_rate,
              self.l2_threshold,
              self.l2_lambda,
              privacy_budget=0.,
              delta_g=0.,
              delta_v=0.,
              loss=self.loss_,
              max_depth=self.max_depth,
              max_leaves=self.max_leaves,
              min_samples_split=self.min_samples_split,
              use_bfs=self.use_bfs,
              use_3_trees=self.use_3_trees,
              use_decay=self.use_decay,
              cat_idx=self.cat_idx,
              num_idx=self.num_idx)
          tree.Fit(X, (y == kth_tree).astype(
              np.float64) if self.loss_.is_multi_class else y, gradients)
          # Add the tree to its corresponding ensemble
          k_trees.append(tree)
      self.trees.append(k_trees)

      score = self.loss_(y_test, self.Predict(X_test))  # i.e. mse or deviance
      logger.info('Decision tree {0:d} fit. Current score: {1:f} - Best '
                  'score so far: {2:f}'.format(tree_index, score, prev_score))

      if score >= prev_score:
        # This tree doesn't improve overall prediction quality, removing from model
        # not reusing gradients in multi-class as they are class-dependent
        update_gradients = self.loss_.is_multi_class
        # self.trees.pop()          # Uncommenting this is responsible for not rejecting trees
        if not self.use_dp:
          self.early_stop -= 1
          if self.early_stop == 0:
            logger.info('Early stop kicked in. No improvement, stopping.')
            break
      else:
        update_gradients = True
        prev_score = score
        # Remove the selected rows from the ensemble's dataset
        # The instances that were filtered out by GBF can still be used for the
        # training of the next trees
        if self.use_dp:
          logger.debug(
              'Success fitting tree {0:d} on {1:d} instances. Instances left '
              'for the ensemble: {2:d}'.format(
                  tree_index, len(rows), len(X_ensemble) - len(rows)))
          X_ensemble = np.delete(X_ensemble, rows, axis=0)
          y_ensemble = np.delete(y_ensemble, rows)

    return self

  def Predict(self, X: np.array) -> np.array:
    """Predict values from the ensemble of gradient boosted trees.

    See https://github.com/microsoft/LightGBM/issues/1778.

    Args:
      X (np.array): The dataset for which to predict values.

    Returns:
      np.array of shape (n_samples, K): The predictions.
    """
    # sum across the ensemble per class
    predictions = np.sum([[self.learning_rate * tree.Predict(X)
                           for tree in k_trees] for k_trees in self.trees],
                         axis=0).T
    if len(X) <= len(self.init_score):
      assert self.init_score is not None
      init_score = self.init_score[:len(predictions)]
      return np.add(init_score, predictions)
    assert self.init_ is not None
    assert self.loss_ is not None
    init_score = self.loss_.get_init_raw_predictions(X, self.init_)
    return np.add(init_score, predictions)

  def PredictLabels(self, X: np.ndarray) -> np.ndarray:
    """Predict labels out of the raw prediction values of `Predict`.
       Only defined for classification tasks.

    Args:
      X (np.ndarray): The dataset for which to predict labels.

    Returns:
      np.ndarray: The label predictions.

    Raises:
      ValueError: If the loss function doesn't match the prediction task.
    """
    if not isinstance(self.loss_, (MultinomialDeviance, BinomialDeviance)):
      raise ValueError("Labels are not defined for regression tasks.")

    raw_predictions = self.Predict(X)
    # pylint: disable=no-member,protected-access
    encoded_labels = self.loss_._raw_prediction_to_decision(raw_predictions)
    # pylint: enable=no-member,protected-access
    return encoded_labels

  def PredictProba(self, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities for X.

    Args:
      X (np.ndarray): The dataset for which to predict labels.

    Returns:
      np.ndarray: The class probabilities of the input samples.

    Raises:
      ValueError: If the loss function doesn't match the prediction task.
    """
    if not isinstance(self.loss_, (MultinomialDeviance, BinomialDeviance)):
      raise ValueError("Labels are not defined for regression tasks.")

    raw_predictions = self.Predict(X)
    # pylint: disable=no-member,protected-access
    probas = self.loss_._raw_prediction_to_proba(raw_predictions)
    # pylint: enable=no-member,protected-access
    return probas

  def ComputeGradientForLossFunction(self,
                                     y: np.array,
                                     y_pred: np.array,
                                     k: int) -> np.array:
    """Compute the gradient of the loss function.

    Args:
      y (np.array): The true values.
      y_pred (np.array): The predictions.
      k (int): the class.

    Returns:
      (np.array): The gradient of the loss function.
    """
    if self.loss_.is_multi_class:
      y = (y == k).astype(np.float64)
    # sklearn's impl is using the negative gradient (i.e. y - F).
    # Here the positive gradient is used though
    return -self.loss_.negative_gradient(y, y_pred, k=k)


class DecisionNode:
  """Implement a decision node.

  Attributes:
    X (np.array): The dataset.
    y (np.ndarray): The dataset labels.
    gradients (np.array): The gradients for the dataset instances.
    index (int): An index for the feature on which the node splits.
    value (Any): The corresponding value for that index.
    depth (int): The depth of the node.
    left_child (DecisionNode): The left child of the node, if any.
    right_child (DecisionNode): The right child of the node, if any.
    prediction (float): For a leaf node, holds the predicted value.
    processed (bool): If a node has been processed during BFS tree construction.
  """
  # pylint: disable=too-many-arguments

  def __init__(self,
               X: Optional[np.array] = None,
               y: Optional[np.array] = None,
               gradients: Optional[np.array] = None,
               index: Optional[int] = None,
               value: Optional[Any] = None,
               gain: Optional[Any] = None,
               depth: Optional[int] = None,
               left_child: Optional['DecisionNode'] = None,
               right_child: Optional['DecisionNode'] = None,
               prediction: Optional[float] = None) -> None:
    """Initialize a decision node.

    Args:
      X (np.array): Optional. The dataset associated to the node. Only for
          BFS tree and 3-tree construction.
      y (np.ndarray): Optional. The dataset labels associated to the node and
          used for the leaf predictions. Only for BFS tree and 3-tree
          construction.
      gradients (np.array): The gradients for the dataset instances.
      index (int): Optional. An index for the feature on which the node splits.
          Default is None.
      value (Any): Optional. The corresponding value for that index. Default
          is None.
      gain (Any): Optional. The gain that splitting on this value and this
          feature generates. Default is None.
      depth (int): Optional. The depth for the node. Only for BFS tree
          construction.
      left_child (DecisionNode): Optional. The left child of the node, if any.
          Default is None.
      right_child (DecisionNode): Optional. The right child of the node, if any.
          Default is None.
      prediction (float): Optional. For a leaf node, holds the predicted value.
          Default is None.
    """
    # pylint: disable=invalid-name

    self.X = X
    self.y = y
    self.gradients = gradients
    self.index = index
    self.value = value
    self.gain = gain
    self.depth = depth
    self.left_child = left_child
    self.right_child = right_child
    self.prediction = prediction
    self.processed = False

    # To export the tree and plot it, we have to conform to scikit's attributes
    self.n_outputs = 1
    self.node_id = None  # type: Optional[int]


class TreeExporter:
  """Class to export and plot decision trees using Scikit's library."""
  def __init__(self, nodes: List[DecisionNode], l2_lambda: float) -> None:
    self.l2_lambda = l2_lambda
    nodes = sorted(nodes, key=lambda x: x.node_id)
    self.n_outputs = 1
    self.value = []
    self.children_left = []
    self.children_right = []
    self.threshold = []
    self.impurity = []
    for node in nodes:
      if node.prediction is not None:
        self.value.append(np.asarray([[node.prediction]]))
      else:
        # Not a leaf node, so we return that value it'd have been if this was
        # a leaf node
        if node.node_id != 0:  # skipping root node
          assert node.gradients is not None
          intermediate_pred = (-1 * np.sum(node.gradients) / (len(
              node.gradients) + self.l2_lambda))  # type: float
          self.value.append(np.asarray([[intermediate_pred]]))
        else:
          self.value.append(np.asarray([[0.]]))

      if node.value is not None:
        self.threshold.append(node.value)
      else:
        self.threshold.append(-1)

      if not node.left_child:
        self.children_left.append(-1)
      else:
        self.children_left.append(node.left_child.node_id)  # type: ignore
      if not node.right_child:
        self.children_right.append(-1)
      else:
        self.children_right.append(node.right_child.node_id)  # type: ignore

      if node.gain is not None:
        self.impurity.append(node.gain)
      else:
        self.impurity.append(-1)

    self.feature = [node.index for node in nodes]
    self.n_node_samples = [len(node.X) for node in nodes]  # type: ignore
    self.n_classes = [1]
    self.weighted_n_node_samples = np.full(fill_value=1, shape=len(nodes))


class DifferentiallyPrivateTree(BaseEstimator):  # type: ignore
  """Implement a differentially private decision tree.

  Attributes:
    root_node (DecisionNode): The root node of the decision tree.
    nodes_bfs (List[DecisionNode]): All nodes in the tree.
    tree_index (int): The index of the tree being trained.
    learning_rate (float): The learning rate.
    l2_threshold (float): Threshold for leaf clipping.
    l2_lambda (float): Regularization parameter for l2 loss function.
    privacy_budget (float): The tree's privacy budget.
    delta_g (float): The utility function's sensitivity.
    delta_v (float): The sensitivity for leaf clipping.
    loss (LossFunction): An sklearn loss wrapper
        suitable for regression and classification.
    max_depth (int): Max. depth for the tree.
  """
  # pylint: disable=invalid-name,too-many-arguments

  def __init__(self,
               tree_index: int,
               learning_rate: float,
               l2_threshold: float,
               l2_lambda: float,
               privacy_budget: float,
               delta_g: float,
               delta_v: float,
               loss: LossFunction,
               max_depth: int = 6,
               max_leaves: Optional[int] = None,
               min_samples_split: int = 2,
               leaf_clipping: bool = False,
               use_bfs: bool = False,
               use_3_trees: bool = False,
               use_decay: bool = False,
               cat_idx: Optional[List[int]] = None,
               num_idx: Optional[List[int]] = None) -> None:
    """Initialize the decision tree.

    Args:
      tree_index (int): The index of the tree being trained.
      learning_rate (float): The learning rate.
      l2_threshold (float): Threshold for leaf clipping.
      l2_lambda (float): Regularization parameter for l2 loss function.
      privacy_budget (float): The tree's privacy budget.
      delta_g (float): The utility function's sensitivity.
      delta_v (float): The sensitivity for leaf clipping.
      loss (LossFunction): An sklearn loss wrapper
          suitable for regression and classification.
          Valid options are: `LeastSquaresError` or `MultinomialDeviance`.
      max_depth (int): Optional. Max. depth for the tree. Default is 6.
      max_leaves (int): Optional. The max number of leaf nodes for the trees.
          Tree will grow in a best-leaf first fashion until it contains
          max_leaves or until it reaches maximum depth, whichever comes first.
      min_samples_split (int): Optional. The minimum number of samples required
          to split an internal node. Default is 2.
      leaf_clipping (bool): Optional. Whether or not to clip the leaves
          after training. Default is False.
      use_bfs (bool): Optional. If max_leaves is specified, then this is
          automatically True. This will build the tree in a BFS fashion instead
          of DFS. Default is False.
      use_3_trees (bool): Optional. If True, only build trees that have 3
          nodes, and then assemble nb_trees based on these sub-trees, at random.
          Default is False.
      use_decay (bool): Optional. If True, internal node privacy budget has a
          decaying factor.
      cat_idx (List): Optional. List of indices for categorical features.
      num_idx (List): Optional. List of indices for numerical features.
    """
    assert type(loss) in [LeastSquaresError,
                          MultinomialDeviance,
                          BinomialDeviance]

    self.root_node = None  # type: Optional[DecisionNode]
    self.nodes_bfs = Queue()  # type: Queue[DecisionNode]
    self.nodes = []  # type: List[DecisionNode]
    self.tree_index = tree_index
    self.learning_rate = learning_rate
    self.l2_threshold = l2_threshold
    self.l2_lambda = l2_lambda
    self.privacy_budget = privacy_budget
    self.delta_g = delta_g
    self.delta_v = delta_v
    self.loss = loss
    self.max_depth = max_depth
    self.max_leaves = max_leaves
    self.min_samples_split = min_samples_split
    self.leaf_clipping = leaf_clipping
    self.use_bfs = use_bfs
    self.use_3_trees = use_3_trees
    self.use_decay = use_decay
    self.cat_idx = cat_idx
    self.num_idx = num_idx

    self.use_dp = privacy_budget > 0.  # Use differential privacy or not

    # This handles attribute comparison depending on the attribute's nature
    self.feature_to_op = defaultdict(
      lambda: (operator.lt, operator.ge))  # type: Dict[int, Any]
    if self.cat_idx:
      for feature_index in self.cat_idx:
        self.feature_to_op[feature_index] = (operator.eq, operator.ne)

    if self.max_leaves and not use_bfs:
      # If max_leaves is specified, we grow the tree in a best-leaf first
      # approach
      self.use_bfs = True

    # To keep track of total number of leaves in the tree
    self.current_number_of_leaves = 0
    self.max_leaves_reached = False

    # To export the tree and plot it, we have to conform to scikit's attributes
    self.criterion = 'gain'
    self.tree_ = None  # type: Optional[TreeExporter]

    self.decision_path = []

  def Fit(self, X: np.array, y: np.ndarray, gradients: np.array) -> None:
    """Fit the tree to the data.

    Args:
      X (np.array): The dataset.
      y (np.ndarray): The dataset labels.
      gradients (np.array): The gradients for the dataset instances.
    """

    # Construct the tree recursively
    if self.use_bfs:
      self.root_node = self.MakeTreeBFS(X, y, gradients)
    else:
      current_depth = 0
      self.root_node = self.MakeTreeDFS(
          X, y, gradients, current_depth, self.max_depth)

    leaves = [node for node in self.nodes if node.prediction]

    if self.use_dp:
      if self.leaf_clipping:
        # Clip the leaf nodes
        logger.debug('Performing geometric leaf clipping')
        ClipLeaves(
            leaves, self.l2_threshold, self.learning_rate, self.tree_index)

      # Add noise to the predictions
      privacy_budget_for_leaf_node = self.privacy_budget / 2
      laplace_scale = self.delta_v / privacy_budget_for_leaf_node
      logger.debug('Adding Laplace noise with scale: {0:f}'.format(
          laplace_scale))
      AddLaplacianNoise(leaves, laplace_scale)

    # Make the tree exportable if we want to print it
    # Assign unique IDs to nodes
    node_ids = Queue()  # type: Queue[int]
    for node_id in range(0, len(self.nodes)):
      node_ids.put(node_id)
    if not self.use_bfs:
      self.AssignNodeIDs(self.root_node, node_ids)
    else:
      root_node = max(self.nodes, key=lambda x: len(x.X))  # type: ignore
      self.AssignNodeIDs(root_node, node_ids)
    self.tree_ = TreeExporter(self.nodes, self.l2_lambda)

  def MakeTreeDFS(self,
                  X: np.array,
                  y: np.ndarray,
                  gradients: np.array,
                  current_depth: int,
                  max_depth: int,
                  X_sibling: Optional[np.array] = None,
                  gradients_sibling: Optional[np.array] = None) -> DecisionNode:
    """Build a tree recursively, in DFS fashion.

    Args:
      X (np.array): The dataset.
      y (np.ndarray): The dataset labels.
      gradients (np.array): The gradients for the dataset instances.
      current_depth (int): Current depth for the tree.
      max_depth (int): Max depth for the tree.
      X_sibling (np.array): Optional. The subset of data in the sibling node.
          For 3_trees only.
      gradients_sibling (np.array): Optional. The gradients in the sibling
          node. For 3_trees only.

    Returns:
      DecisionNode: A decision node.
    """

    def MakeLeafNode() -> DecisionNode:
      node = DecisionNode(X=X,
                          y=y,
                          gradients=gradients,
                          prediction=self.GetLeafPrediction(gradients, y),
                          depth=current_depth)
      self.nodes.append(node)
      return node

    if current_depth == max_depth or len(X) < self.min_samples_split:
      # Max depth reached or not enough samples to split node, node is a leaf
      # node
      return MakeLeafNode()

    if not self.use_3_trees:
      best_split = self.FindBestSplit(X, gradients, current_depth)
    else:
      if current_depth != 0:
        best_split = self.FindBestSplit(
            X, gradients, current_depth, X_sibling=X_sibling,
            gradients_sibling=gradients_sibling)
      else:
        best_split = self.FindBestSplit(X, gradients, current_depth)
    if best_split:
      logger.debug('Tree DFS: best split found at index {0:d}, value {1:f} '
                   'with gain {2:f}. Current depth is {3:d}'.format(
          best_split['index'], best_split['value'],
          best_split['gain'], current_depth))
      lhs_op, rhs_op = self.feature_to_op[best_split['index']]
      lhs = np.where(lhs_op(X[:, best_split['index']], best_split['value']))[0]
      rhs = np.where(rhs_op(X[:, best_split['index']], best_split['value']))[0]
      if not self.use_3_trees:
        left_child = self.MakeTreeDFS(
            X[lhs],  y[lhs], gradients[lhs], current_depth + 1, max_depth)
        right_child = self.MakeTreeDFS(
            X[rhs],  y[rhs], gradients[rhs], current_depth + 1, max_depth)
      else:
        left_child = self.MakeTreeDFS(
            X[lhs], y[lhs], gradients[lhs], current_depth + 1, max_depth,
            X_sibling=X[rhs], gradients_sibling=gradients[rhs])
        right_child = self.MakeTreeDFS(
            X[rhs], y[rhs], gradients[rhs], current_depth + 1, max_depth,
            X_sibling=X[lhs], gradients_sibling=gradients[lhs])
      node = DecisionNode(X=X,
                          gradients=gradients,
                          index=best_split['index'],
                          value=best_split['value'],
                          gain=best_split['gain'],
                          left_child=left_child,
                          right_child=right_child,
                          depth=current_depth)
      self.nodes.append(node)
      return node

    return MakeLeafNode()

  def MakeTreeBFS(self,
                  X: np.array,
                  y: np.ndarray,
                  gradients: np.array) -> DecisionNode:
    """Build a tree in a best-leaf first fashion.

    Args:
      X (np.array): The dataset.
      y (np.ndarray): The dataset labels.
      gradients (np.array): The gradients for the dataset instances.

    Returns:
      DecisionNode: A decision node.
    """

    best_split = self.FindBestSplit(X, gradients, current_depth=0)
    if not best_split:
      node = DecisionNode(X=X,
                          gradients=gradients,
                          prediction=self.GetLeafPrediction(gradients, y))
      self.nodes.append(node)
      return node

    logger.debug('Tree BFS: best split found at index {0:d}, value {1:f} with '
                'gain {2:f}.'.format(
        best_split['index'], best_split['value'], best_split['gain']))

    # Root node
    node = DecisionNode(X=X,
                        y=y,
                        gradients=gradients,
                        index=best_split['index'],
                        value=best_split['value'],
                        gain=best_split['gain'],
                        depth=0)
    self.nodes_bfs.put(node)
    self._ExpandTreeBFS()
    for node in self.nodes:
      # Assigning predictions to remaining leaf nodes if we had to stop
      # constructing the tree early because we reached max number of leaf nodes
      if not node.prediction and not node.left_child and not node.right_child:
        node.prediction = self.GetLeafPrediction(node.gradients, node.y)
    return node

  def _ExpandTreeBFS(self) -> None:
    """Expand a tree in a best-leaf first fashion.

    Implement https://researchcommons.waikato.ac.nz/bitstream/handle/10289/2317
    /thesis.pdf?sequence=1&isAllowed=y
    """

    # Node queue is empty or too many leaves, stopping
    if self.nodes_bfs.empty() or self.max_leaves_reached:
      return None

    current_node = self.nodes_bfs.get()

    # If there are not enough samples to split in that node, make it a leaf
    # node and process next node
    assert current_node.gradients is not None
    if len(current_node.gradients) < self.min_samples_split:
      self._MakeLeaf(current_node)
      if not self._IsMaxLeafReached():
        return self._ExpandTreeBFS()
      return None

    # If we reached max depth
    if current_node.depth == self.max_depth:
      self._MakeLeaf(current_node)
      if not self._IsMaxLeafReached():
        if self.max_leaves:
          return self._ExpandTreeBFS()
        while not self.nodes_bfs.empty():
          node = self.nodes_bfs.get()
          self._MakeLeaf(node)
      return None

    # Do the split
    assert current_node.X is not None
    assert current_node.y is not None
    assert current_node.gradients is not None
    assert current_node.depth is not None
    lhs_op, rhs_op = self.feature_to_op[current_node.index]  # type: ignore
    lhs = np.where(
        lhs_op(current_node.X[:, current_node.index], current_node.value))[0]
    rhs = np.where(
        rhs_op(current_node.X[:, current_node.index], current_node.value))[0]
    lhs_X, rhs_X = current_node.X[lhs], current_node.X[rhs]
    lhs_grad, rhs_grad = current_node.gradients[lhs], current_node.gradients[
        rhs]

    lhs_y, rhs_y = current_node.y[lhs], current_node.y[rhs]
    lhs_best_split = self.FindBestSplit(
        lhs_X, lhs_grad, current_depth=current_node.depth + 1)
    rhs_best_split = self.FindBestSplit(
        rhs_X, rhs_grad, current_depth=current_node.depth + 1)

    # Can't split the node, so this becomes a leaf node.
    if not lhs_best_split or not rhs_best_split:
      self._MakeLeaf(current_node)
      if not self._IsMaxLeafReached():
        return self._ExpandTreeBFS()
      return None

    logger.debug('Tree BFS: best split found at index {0:d}, value {1:f} with '
                 'gain {2:f}.'.format(
      lhs_best_split['index'], lhs_best_split['value'], lhs_best_split['gain']))
    logger.debug('Tree BFS: best split found at index {0:d}, value {1:f} with '
                 'gain {2:f}.'.format(
      rhs_best_split['index'], rhs_best_split['value'], rhs_best_split['gain']))

    # Splitting the node is possible, creating the children
    assert current_node.depth is not None
    left_child = DecisionNode(X=lhs_X,
                              y=lhs_y,
                              gradients=lhs_grad,
                              index=lhs_best_split['index'],
                              value=lhs_best_split['value'],
                              gain=lhs_best_split['gain'],
                              depth=current_node.depth + 1)
    right_child = DecisionNode(X=rhs_X,
                               y=rhs_y,
                               gradients=rhs_grad,
                               index=rhs_best_split['index'],
                               value=rhs_best_split['value'],
                               gain=rhs_best_split['gain'],
                               depth=current_node.depth + 1)

    current_node.left_child = left_child
    current_node.right_child = right_child
    self.nodes.append(current_node)

    # Adding them to the list of nodes for further expansion in best-gain order
    if lhs_best_split['gain'] >= rhs_best_split['gain']:
      self.nodes_bfs.put(left_child)
      self.nodes_bfs.put(right_child)
    else:
      self.nodes_bfs.put(right_child)
      self.nodes_bfs.put(left_child)
    return self._ExpandTreeBFS()

  def _MakeLeaf(self, node: DecisionNode) -> None:
    """Make a node a leaf node.

    Args:
      node (DecisionNode): The node to make a leaf from.
    """
    node.prediction = self.GetLeafPrediction(node.gradients, node.y)
    self.current_number_of_leaves += 1
    self.nodes.append(node)

  def _IsMaxLeafReached(self) -> bool:
    """Check if we reached maximum number of leaf nodes.

    Returns:
      bool: True if we reached the maximum number of leaf nodes,
          False otherwise.
    """
    leaf_candidates = 0
    for node in list(self.nodes_bfs.queue):
      if not node.left_child and not node.right_child:
        leaf_candidates += 1
    if self.max_leaves:
      if self.current_number_of_leaves + leaf_candidates >= self.max_leaves:
        self.max_leaves_reached = True
    return self.max_leaves_reached

  def FindBestSplit(self,
                    X: np.array,
                    gradients: np.array,
                    current_depth: Optional[int] = None,
                    X_sibling: Optional[np.array] = None,
                    gradients_sibling: Optional[np.array] = None,
                    ) -> Optional[Dict[str, Any]]:
    """Find best split of data using the exponential mechanism.

    Args:
      X (np.array): The dataset.
      gradients (np.array): The gradients for the dataset instances.
      current_depth (int): Optional. The current depth of the tree. If
          specified, the privacy budget decays with the depth growing.
      X_sibling (np.array): Optional. The subset of data in the sibling node.
          For 3_trees only.
      gradients_sibling (np.array): Optional. The gradients in the sibling
          node. For 3_trees only.

    Returns:
      Optional[Dict[str, Any]]: A dictionary containing the split
          information, or none if no split could be done.
    """

    if self.use_dp:
      if current_depth and self.use_decay:
        privacy_budget_for_node = np.around(
            np.divide(self.privacy_budget/2, np.power(
                2, current_depth)), decimals=7)
      else:
        privacy_budget_for_node = np.around(
            np.divide(self.privacy_budget/2, self.max_depth), decimals=7)

      if self.use_3_trees and current_depth != 0:
        # If not for the root node splitting, budget is divided by the 3-nodes
        privacy_budget_for_node /= 2

      logger.debug('Using {0:f} budget for internal leaf nodes.'.format(
          privacy_budget_for_node))

    probabilities = []
    max_gain = -np.inf
    # Iterate over features
    for feature_index in range(X.shape[1]):
      binary_split = len(np.unique(X[:, feature_index])) == 2
      # Iterate over unique value for this feature
      for idx, value in enumerate(np.unique(X[:, feature_index])):
        # Find gain for that split
        # if binary_split and idx == 1:
        #   # If the attribute only has 2 values then we don't need to care for
        #   # both gains as they're equal
        #   prob = {
        #     'index': feature_index,
        #     'value': value,
        #     'gain': 0.
        #   }
        # else:
        gain = self.ComputeGain(
            feature_index, value, X, gradients, X_sibling=X_sibling,
            gradients_sibling=gradients_sibling)
        if gain == -1:
          # Feature's value cannot be chosen, skipping
          continue
        # Compute probability for exponential mechanism
        if self.use_dp:
          gain = (privacy_budget_for_node * gain) / (2. * self.delta_g)
        if gain > max_gain:
          max_gain = gain
        prob = {
            'index': feature_index,
            'value': value,
            'gain': gain
        }
        probabilities.append(prob)
    if self.use_dp:
      return ExponentialMechanism(probabilities, max_gain)
    return max(
        probabilities, key=lambda x: x['gain']) if probabilities else None

  def GetLeafPrediction(self, gradients: np.array, y: np.ndarray) -> float:
    """Compute the leaf prediction.

    Args:
      gradients (np.array): The gradients for the dataset instances.
      y (np.ndarray): The dataset labels.

    Returns:
      float: The prediction for the leaf node
    """
    return ComputePredictions(gradients, y, self.loss, self.l2_lambda)

  def Predict(self, X: np.array) -> np.array:
    """Return predictions for a list of input data.

    Args:
      X: The input data used for prediction.

    Returns:
      np.array: An array with the predictions.
    """
    predictions = []
    for row in X:
      predictions.append(self._Predict(row, self.root_node))  # type: ignore
    return np.asarray(predictions)

  def _Predict(self, row: np.array, node: DecisionNode) -> float:
    """Walk through the decision tree to output a prediction for the row.

    Args:
      row (np.array): The row to classify.
      node (DecisionNode): The current decision node.

    Returns:
      float: A prediction for the row.
    """
    self.decision_path.append(node.node_id)
    if node.prediction is not None:
      return node.prediction
    assert node.index is not None
    value = row[node.index]
    _, rhs_op = self.feature_to_op[node.index]
    if rhs_op(value, node.value):
      child_node = node.right_child
    else:
      child_node = node.left_child
    return self._Predict(row, child_node)  # type: ignore

  def ComputeGain(self,
                  index: int,
                  value: Any,
                  X: np.array,
                  gradients: np.array,
                  X_sibling: Optional[np.array] = None,
                  gradients_sibling: Optional[np.array] = None) -> float:
    """Compute the gain for a given split.

    See https://dl.acm.org/doi/pdf/10.1145/2939672.2939785

    Args:
      index (int): The index for the feature to split on.
      value (Any): The feature's value to split on.
      X (np.array): The dataset.
      gradients (np.array): The gradients for the dataset instances.
      X_sibling (np.array): Optional. The subset of data in the sibling node.
          For 3_trees only.
      gradients_sibling (np.array): Optional. The gradients in the sibling
          node. For 3_trees only.

    Returns:
      float: The gain for the split.
    """
    lhs_op, rhs_op = self.feature_to_op[index]
    lhs = np.where(lhs_op(X[:, index], value))[0]
    rhs = np.where(rhs_op(X[:, index], value))[0]
    if len(lhs) == 0 or len(rhs) == 0:
      # Can't split on this feature as all instances share the same value
      return -1

    if self.use_3_trees and X_sibling is not None:
      X = np.concatenate((X, X_sibling), axis=0)
      gradients = np.concatenate(
          (gradients, gradients_sibling), axis=0)
      lhs = np.where(lhs_op(X[:, index], value))[0]
      rhs = np.where(rhs_op(X[:, index], value))[0]

    lhs_grad, rhs_grad = gradients[lhs], gradients[rhs]
    lhs_gain = np.square(np.sum(lhs_grad)) / (
        len(lhs) + self.l2_lambda)  # type: float
    rhs_gain = np.square(np.sum(rhs_grad)) / (
        len(rhs) + self.l2_lambda)  # type: float
    total_gain = lhs_gain + rhs_gain
    return total_gain if total_gain >= 0. else 0.

  def AssignNodeIDs(self,
                    node: DecisionNode,
                    node_ids: Queue  # type: ignore
                    ) -> None:
    """Walk through the tree and assign a unique ID to the decision nodes.

    Args:
      node (DecisionNode): The node of the tree to assign an ID to.
      node_ids (Queue): Queue that contains all available node ids.
    """
    node.node_id = node_ids.get()
    if node.left_child:
      self.AssignNodeIDs(node.left_child, node_ids)
    if node.right_child:
      self.AssignNodeIDs(node.right_child, node_ids)

  @staticmethod
  def fit() -> None:
    """Stub for BaseEstimator."""
    return

  @staticmethod
  def predict() -> None:
    """Stub for BaseEstimator"""
    return

  def GetDecisionPathForRow(self, row: np.array) -> np.array:
    self.decision_path = []
    self.Predict(row)
    return self.decision_path


def ClipLeaves(leaves: List[DecisionNode],
               l2_threshold: float,
               learning_rate: float,
               tree_index: int) -> None:
  """Clip leaf nodes.

  If the prediction is higher than the threshold, set the prediction to
  that threshold.

  Args:
    leaves (List[DecisionNode]): The leaf nodes.
    l2_threshold (float): Threshold of the l2 loss function.
    learning_rate (float): The learning rate.
    tree_index (int): The index for the current tree.
  """
  threshold = l2_threshold * math.pow((1 - learning_rate), tree_index)
  for leaf in leaves:
    assert leaf.prediction is not None
    if np.abs(leaf.prediction) > threshold:
      if leaf.prediction > 0:
        leaf.prediction = threshold
      else:
        leaf.prediction = -1 * threshold


def AddLaplacianNoise(leaves: List[DecisionNode],
                      scale: float) -> None:
  """Add laplacian noise to the leaf nodes.

  Args:
    leaves (List[DecisionNode]): The list of leaves.
    scale (float): The scale to use for the laplacian distribution.
  """

  for leaf in leaves:
    noise = np.random.laplace(0, scale)
    logger.debug('Leaf value before noise: {0:f}'.format(
        np.float(leaf.prediction)))
    leaf.prediction += noise
    logger.debug('Leaf value after noise: {0:f}'.format(
        np.float(leaf.prediction)))


def ComputePredictions(gradients: np.ndarray,
                       y: np.ndarray,
                       loss: LossFunction,
                       l2_lambda: float) -> float:
  """Computes the predictions of a leaf.

  Used in the `DifferentiallyPrivateTree` as well as in `SplitNode`
  for the 3-tree version.

  Ref:
    Friedman 01. "Greedy function approximation: A gradient boosting machine."
      (https://projecteuclid.org/euclid.aos/1013203451)

  Args:
    gradients (np.ndarray): The positive gradients y˜ for the dataset instances.
    y (np.ndarray): The dataset labels y.
    loss (LossFunction): An sklearn loss wrapper
        suitable for regression and classification.
    l2_lambda (float): Regularization parameter for l2 loss function.

  Returns:
    Prediction γ of a leaf
  """
  if len(gradients) == 0:
    prediction = 0.  # type: ignore
  elif loss.is_multi_class:
    # sum of neg. gradients divided by sum of 2nd derivatives
    # aka one Newton-Raphson step
    # for details ref. (eq 33+34) in Friedman 01.
    prediction = -1 * np.sum(gradients) * (loss.K - 1) / loss.K
    denom = np.sum((y + gradients) * (1 - y - gradients))
    prediction = 0 if abs(denom) < 1e-150 else prediction / (
        denom + l2_lambda)
  else:
    prediction = (-1 * np.sum(gradients) / (len(
        gradients) + l2_lambda))
  return prediction


def ExponentialMechanism(
    probabilities: List[Dict[str, Any]],
    max_gain: float,
    reverse: bool = False) -> Optional[Dict[str, Any]]:
  """Apply the exponential mechanism.

  Args:
    probabilities (List[Dict]): List of probabilities to choose from.
    max_gain (float): The maximum gain amongst all probabilities in the list.
    reverse (bool): Optional. If True, sort probabilities in reverse order (
        i.e. lower gains are better).

  Returns:
    Dict: a candidate (i.e. probability) from the list.
  """

  if (np.asarray([prob['gain'] for prob in probabilities]) <= 0.0).all():
    # No split offers a positive gain, node should be a leaf node
    return None

  with np.errstate(all='raise'):
    try:
      gains = np.asarray([prob['gain'] for prob in probabilities if prob[
        'gain'] != 0.], dtype=np.float128)
      for prob in probabilities:
        # e^0 is 1, so checking for that
        if prob['gain'] <= 0.:
          prob['probability'] = 0.
        else:
          prob['probability'] = np.exp(prob['gain'] - logsumexp(gains))
    # Happens when np.exp() overflows because of a gain that's too high
    except FloatingPointError:
      for prob in probabilities:
        gain = prob['gain']
        if gain > 0.:
          # Check if the gain of each candidate is too small compared to
          # the max gain seen up until now. If so, set the probability for
          # this split to 0.
          try:
            _ = np.exp(max_gain - gain)
          except FloatingPointError:
            prob['probability'] = 0.
          # If it's not too small, we need to compute a new sum that
          # doesn't overflow. For that we only take into account 'large'
          # gains with respect to the current candidate. If again the
          # difference is so small that it would still overflow, we set the
          # probability for this split to 0.
          sub_sum_exp = 0.
          try:
            sub_sum_exp = logsumexp(np.asarray(gains - gain, dtype=np.float128))
          except FloatingPointError:
            prob['probability'] = 0.

          # Other candidates compare similarly, so we can compute a
          # probability. If it underflows, set it to 0 as well.
          if sub_sum_exp > 0.:
            try:
              prob['probability'] = np.exp(0. - sub_sum_exp)  # E.q. to 1/e^sub
            except FloatingPointError:
              prob['probability'] = 0.
        else:
          prob['probability'] = 0.

  # Apply the exponential mechanism
  previous_prob = 0.
  random_prob = np.random.uniform()
  for prob in probabilities:
    if prob['probability'] != 0.:
      prob['probability'] += previous_prob
      previous_prob = prob['probability']

  op = operator.ge if not reverse else operator.le
  # Try and find a candidate at least 10 times before giving up and making
  # the node a leaf node
  for _ in range(10):
    for prob in probabilities:
      if op(prob['probability'], random_prob):
        return prob
    random_prob = np.random.uniform()
  return None
