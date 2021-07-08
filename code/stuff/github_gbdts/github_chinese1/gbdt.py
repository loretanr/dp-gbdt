'''
Created on Nov 30st, 2019
Tree-Based Regression Methods
@author: Shihao Liu
'''

import numpy as np
import logging 
import argparse

from decision_tree import DecisionTree, Data

parser = argparse.ArgumentParser()
parser.add_argument('-maximum_height', default=3, type=int)
parser.add_argument('-iterations', default=20, type=int)
parser.add_argument('-lr', default=0.5, type=float)

args = parser.parse_args()

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GBDT():
    def __init__(self, lr=0.3, iterations=80, maximum_height=10):
        self.forest = []
        self.learning_rate = lr
        self.iterations = iterations
        self.lr_list = []
        self.maximum_height = maximum_height

    def fit(self, dataset):
        base_constant_prediction = np.mean(dataset[:, -1])
        base_tree = DecisionTree(is_leaf=True, split_value=base_constant_prediction, maximum_height=self.maximum_height)
        residuals = dataset[:, -1] - base_constant_prediction
        self.lr_list.append(1)
        self.forest.append(base_tree)
        features = dataset[:, :-1]
        for i in range(self.iterations):
            dataset = np.concatenate((features, residuals[:, None]), axis=1)
            tree = DecisionTree(maximum_height=self.maximum_height)
            tree.build_tree(tree, dataset)
            for i, example in enumerate(dataset):
                residuals[i] -= self.learning_rate * tree.predict(tree, example)
            self.forest.append(tree)
            self.lr_list.append(self.learning_rate)

    def predict(self, dataset):
        loss = 0
        for example in dataset:
            prediction = 0
            for (lr, tree) in zip(self.lr_list, self.forest):
            # for tree in self.forest:
                prediction += lr * tree.predict(tree, example)
            loss += (example[-1] - prediction) ** 2
            logger.info("真实输出为{:.2f}，预测值为{:.2f}".format(example[-1], prediction))
        logger.info("********** 模型的loss为{:.2f} **********".format(loss))

if __name__ == '__main__':
    # 训练
    data = Data('bikeSpeedVsIq_train.txt')
    training_set = data.load_file()
    gbdt = GBDT(lr=args.lr, iterations=args.iterations, maximum_height=args.maximum_height)
    gbdt.fit(training_set)

    # 预测
    data = Data('./bikeSpeedVsIq_test.txt')
    test_set = data.load_file()
    gbdt.predict(test_set)

    # sklearn自带的波士顿房价数据，及其gbdt训练预测。
    # from sklearn.datasets import load_boston
    # from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import GradientBoostingRegressor
    # data = load_boston()
    # X_train, X_test, y_train, y_test = train_test_split(data["data"],data["target"],test_size=0.3, random_state=0)
    # gbmodel = GradientBoostingRegressor()
    # gbmodel.fit(X_train, y_train)
    # gbdt.predict(test_set)
