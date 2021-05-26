#!/usr/local/bin/python2
# -*- coding: utf-8 -*-
"""
@Time: 2018/11/11 15:20
@Author: zhaoxingfeng
@Function：调自己写的xgboost C++接口实现训练、预测和序列化模型及加载等功能
@Version: V1.1
"""
from ctypes import *
import pandas as pd
import numpy as np
import ctypes
from sklearn import metrics


class Config(Structure):
    _fields_ = [("n_estimators", c_int),
                ("max_depth", c_int),
                ("learning_rate", c_float),
                ("min_samples_split", c_int),
                ("min_data_in_leaf", c_int),
                ("min_child_weight", c_float),
                ("colsample_bytree", c_float),
                ("reg_gamma", c_float),
                ("reg_lambda", c_float),
                ("max_bin", c_int)]

class XGBClassifier(object):
    def __init__(self, n_estimators=100, max_depth=10, learning_rate=0.1, min_samples_split=2,
                 min_data_in_leaf=1, colsample_bytree=1.0, min_child_weight=1.0, reg_gamma=0.0,
                 reg_lambda=0.0, max_bin=100):
        self.config = Config()
        self.config.max_depth = c_int(int(max_depth))
        self.config.n_estimators = c_int(int(n_estimators))
        self.config.learning_rate = c_float(learning_rate)
        self.config.min_samples_split = c_int(int(min_samples_split))
        self.config.min_data_in_leaf = c_int(int(min_data_in_leaf))
        self.config.min_child_weight = c_float(min_child_weight)
        self.config.colsample_bytree = c_float(colsample_bytree)
        self.config.reg_gamma = c_float(reg_gamma)
        self.config.reg_lambda = c_float(reg_lambda)
        self.config.max_bin = c_int(max_bin)
        self._LIB = cdll.LoadLibrary(r"x64/Debug/xgboost-cpp.dll")
        self.handle = ctypes.c_void_p()

    def fit(self, dataset, labels):
        dataset_feature = np.array(dataset, dtype=np.float32)
        dataset_labels = np.array(labels, dtype=np.int32)

        self._LIB.BoosterTrain(byref(self.config), dataset_feature.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               dataset_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                               c_int(dataset_feature.shape[0]), c_int(dataset_feature.shape[1]),
                               ctypes.byref(self.handle))
        del dataset_feature, dataset_labels

    def predict_proba(self, dataset):
        dataset_feature = np.array(dataset, dtype=np.float32)
        preds = np.zeros(dataset_feature.shape[0], dtype=np.float32)
        self._LIB.BoosterPredict(dataset_feature.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                 c_int(dataset_feature.shape[0]), c_int(dataset_feature.shape[1]),
                                 ctypes.byref(self.handle), preds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        return np.vstack((1.0 - preds, preds)).transpose()

    # Save Booster to string.
    def model_to_string(self):
        string_buffer = ctypes.create_string_buffer(1 << 20)
        ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
        self._LIB.BoosterSaveModelToString(ctypes.byref(self.handle), ptr_string_buffer)
        return string_buffer.value.decode()

    def __getstate__(self):
        this = self.__dict__.copy()
        this.pop('_LIB', None)
        handle = this['handle']
        if handle is not None:
            this["handle"] = self.model_to_string()
        return this

    def __setstate__(self, state):
        self.config = state.get('config', None)
        self._LIB = cdll.LoadLibrary(r"x64/Debug/xgboost-cpp.dll")
        model_str = state.get('handle', None)
        if model_str is not None:
            handle = ctypes.c_void_p()
            self._LIB.BoosterLoadModelFromString(ctypes.c_char_p(model_str.encode('utf-8')), ctypes.byref(handle))
            state['handle'] = handle
        self.__dict__.update(state)

def scipy_ks_score(target, proba):
    from scipy.stats import ks_2samp
    return ks_2samp(proba[target == 1], proba[target != 1]).statistic


if __name__ == '__main__':
    import time
    start = time.time()
    from sklearn.externals import joblib

    df = pd.read_csv(r"source/pima indians.csv", header=None)
    # df = pd.read_csv(r"source/credit_card.csv", header=None, nrows=1000)
    df = df.ix[:, 1:].reset_index()
    xgb = XGBClassifier(n_estimators=10,
                        max_depth=6,
                        learning_rate=0.4,
                        min_samples_split=50,
                        min_data_in_leaf=20,
                        colsample_bytree=1.0,
                        min_child_weight=5,
                        reg_gamma=0.3,
                        reg_lambda=0.3,
                        max_bin=100)
    train_count = int(0.7 * len(df))
    xgb.fit(df.iloc[:train_count, :-1], df.iloc[:train_count, -1])
    joblib.dump(xgb, "model.pkl")
    xgb = joblib.load("model.pkl")
    print("Train auc=%s" % metrics.roc_auc_score(df.iloc[:train_count, -1], xgb.predict_proba(df.iloc[:train_count, :-1])[:, 1]))
    print("Test auc=%s" % metrics.roc_auc_score(df.iloc[train_count:, -1], xgb.predict_proba(df.iloc[train_count:, :-1])[:, 1]))
    print("Train ks=%s" % scipy_ks_score(df.iloc[:train_count, -1], xgb.predict_proba(df.iloc[:train_count, :-1])[:, 1]))
    print("Test ks=%s" % scipy_ks_score(df.iloc[train_count:, -1], xgb.predict_proba(df.iloc[train_count:, :-1])[:, 1]))
    print("Running time=%s s" % (time.time() - start))
