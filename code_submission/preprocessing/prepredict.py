#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/5/14 20:41
# @Author:  Mecthew
import time

import numpy as np
import pandas as pd
import scipy
from sklearn.svm import LinearSVC
from sklearn.linear_model import logistic
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
from utils.logger import get_logger
logger = get_logger("INFO")


class SVM:
    def __init__(self, **kwargs):
        self.name = "SVM"
        self._model = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=500, class_weight=None, random_state=666))

    def fit(self, x_train, y_train):
        self._model.fit(x_train, y_train)

    def predict(self, x_test):
        return self._model.predict_proba(x_test)


class LR:
    def __init__(self, **kwargs):
        self.name = "LR"
        self._model = logistic.LogisticRegression(C=1.0, solver="liblinear", multi_class="auto",
                                                  class_weight=None, max_iter=100, random_state=666)

    def fit(self, x_train, y_train):
        self._model.fit(x_train, y_train)

    def predict(self, x_test):
        return self._model.predict_proba(x_test)


def prepredict(graph_df, train_indices, use_valid, use_ohe=False):
    t1 = time.time()
    fea_table = graph_df['fea_table'].set_index(keys="node_index")
    train_indices = train_indices
    if use_valid:
        valid_indices = list(set(graph_df['train_indices']) - set(train_indices))
        test_indices = graph_df['test_indices'] + valid_indices
    else:
        test_indices = graph_df['test_indices']
    train_label = graph_df['train_label'].set_index('node_index').loc[train_indices][['label']]

    x_train, y_train = fea_table.loc[train_indices].to_numpy(), train_label.to_numpy()
    x_test = fea_table.loc[test_indices].to_numpy()
    lr = LR()
    lr.fit(x_train, y_train)

    if use_ohe:
        ohe = OneHotEncoder(handle_unknown="ignore").fit(y_train.reshape(-1, 1))
        x_train_feat, x_test_feat = ohe.transform(np.argmax(lr.predict(x_train), axis=1).reshape(-1, 1)).toarray(), \
                                    ohe.transform(np.argmax(lr.predict(x_test), axis=1).reshape(-1, 1)).toarray()
    else:
        x_train_feat, x_test_feat = lr.predict(x_train), \
                                    lr.predict(x_test)
    pre_feat = np.concatenate([x_train_feat, x_test_feat], axis=0)
    total_indices = np.concatenate([train_indices, test_indices], axis=0)

    train_predict = np.argmax(x_train_feat, axis=1)
    train_acc = accuracy_score(y_true=y_train, y_pred=train_predict)
    t2 = time.time()
    logger.info("Time cost for training {}: {}s, train acc {}".format(lr.name, t2-t1, train_acc))

    return pd.DataFrame(data=pre_feat, index=total_indices)


def lpa_predict(graph_df, n_class, train_indices, use_valid, max_iter=100, tol=1e-3, use_ohe=False):
    t1 = time.time()
    train_indices = train_indices
    if use_valid:
        valid_indices = list(set(graph_df['train_indices']) - set(train_indices))
        test_indices = graph_df['test_indices'] + valid_indices
    else:
        test_indices = graph_df['test_indices']
    train_label = graph_df['train_label'].set_index('node_index').loc[train_indices][['label']].to_numpy()
    print("Train label shape {}".format(train_label.shape))
    train_label = train_label.reshape(-1)
    edges = graph_df['edge_file'][['src_idx', 'dst_idx', 'edge_weight']].to_numpy()
    edge_index = edges[:, :2].astype(np.int).transpose()    # transpose to (2, num_edges)
    edge_weight = edges[:, 2].astype(np.float)
    num_nodes = len(train_indices) + len(test_indices)

    t2 = time.time()
    total_indices = np.concatenate([train_indices, test_indices], axis=0)
    adj = sp.coo_matrix((edge_weight, edge_index), shape=(num_nodes, num_nodes)).tocsr()
    adj = adj[total_indices]       # reorder
    adj = adj[:, total_indices]

    t3 = time.time()
    logger.debug("Time cost for transform adj {}s".format(t3 - t2))
    row_sum = np.array(adj.sum(axis=1), dtype=np.float)
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    normal_adj = sp.diags(d_inv).dot(adj).tocsr().transpose()

    Pll = normal_adj[:len(train_indices), :len(train_indices)].copy()
    Plu = normal_adj[:len(train_indices), len(train_indices):].copy()
    Pul = normal_adj[len(train_indices):, :len(train_indices)].copy()
    Puu = normal_adj[len(train_indices):, len(train_indices):].copy()
    label_mat = np.eye(n_class)[train_label]
    label_mat_prob = label_mat.copy()
    print("Pul shape {}, label_mat shape {}".format(Pul.shape, label_mat_prob.shape))

    Pul_dot_lable_mat = Pul.dot(label_mat)
    unlabel_mat = np.zeros(shape=(len(test_indices), n_class))
    iter, changed = 0, np.inf
    t4 = time.time()
    logger.debug("Time cost for prepare matrix {}s".format(t4-t3))
    while iter < max_iter and changed > tol:
        if iter % 10 == 0:
            logger.debug("---> Iteration %d/%d, changed: %f" % (iter, max_iter, changed))

        iter += 1
        pre_unlabel_mat = unlabel_mat
        unlabel_mat = Puu.dot(unlabel_mat) + Pul_dot_lable_mat
        label_mat_prob = Pll.dot(label_mat_prob) + Plu.dot(pre_unlabel_mat)
        changed = np.abs(pre_unlabel_mat - unlabel_mat).sum()
    logger.debug("Time cost for training lpa {}".format(time.time() - t4))
    # preds = np.argmax(np.array(unlabel_mat), axis=1)
    # unlabel_mat = np.eye(n_class)[preds]
    train_acc = accuracy_score(y_true=train_label, y_pred=np.argmax(label_mat_prob, axis=1))
    logger.info("LPA training acc {}".format(train_acc))
    logger.info("Time cost for LPA {}s".format(time.time() - t1))
    total_indices = np.concatenate([train_indices, test_indices], axis=0)
    if use_ohe:
        ohe = OneHotEncoder(handle_unknown="ignore").fit(train_label.reshape(-1, 1))
        label_mat_ohe = ohe.transform(np.argmax(label_mat_prob, axis=1).reshape(-1, 1)).toarray()
        unlabel_mat_ohe = ohe.transform(np.argmax(unlabel_mat, axis=1).reshape(-1, 1)).toarray()
        lu_mat_ohe = np.concatenate([label_mat_ohe, unlabel_mat_ohe], axis=0)
        return pd.DataFrame(data=lu_mat_ohe, index=total_indices), train_acc
    else:
        unlabel_mat_prob = unlabel_mat
        lu_mat_prob = np.concatenate([label_mat_prob, unlabel_mat_prob], axis=0)
        return pd.DataFrame(data=lu_mat_prob, index=total_indices), train_acc


def is_nonnegative_integer(x_feats):
    is_nonnegative = (x_feats >= 0).all()
    is_integer = True
    for feat in x_feats:
        feat_int_sum = np.array(feat, dtype=np.int).sum()
        feat_sum = np.array(feat, dtype=np.float).sum()
        is_integer = (feat_int_sum == feat_sum)
        if is_integer is False:
            break
    return is_nonnegative and is_integer
