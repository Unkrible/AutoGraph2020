import math
import time

import pandas as pd
import numpy as np
from torch_geometric.nn.models import Node2Vec
import torch
from torch.optim import Adam


def drop_n_unique(x, n=1):
    drop_cols = []
    for col in x:
        if x[col].nunique() == n:
            drop_cols.append(col)
    print(f"Drop {drop_cols} by condition (nunique={n})")
    all_zero = len(drop_cols) == len(x.columns)
    x.drop(columns=drop_cols, inplace=True, axis=1)
    print(f"Remain cols {x.columns}")
    return all_zero


def count_nonzero(x):
    non_zero = (x != 0).sum(axis=1)
    if non_zero.nunique() != 1:
        non_zero /= non_zero.max()
        x['non_zero'] = non_zero


def feat_engineering(x, edges=None, num_nodes=None):
    # TODO: out of memory
    all_zero = drop_n_unique(x)
    if all_zero:
        # print(f"Translate all zero to one hot encode")
        # x = pd.get_dummies(x.index)
        # return x.to_numpy()
        print("Use normalized weight as feature")
        edge_weights = np.zeros((num_nodes, num_nodes), dtype=np.float)
        edge_weights[edges['src_idx'], edges['dst_idx']] = edges['edge_weight']
        # for i in range(num_nodes):
        #     max_weight = np.max(edge_weights[:, i])
        #     min_weight = np.min(edge_weights[:, i])
        #     range_weight = max_weight - min_weight
        #     if math.isclose(range_weight, 0, abs_tol=1e-4):
        #         continue
        #     edge_weights[:, i] = (edge_weights[:, i] - min_weight) / range_weight
        return edge_weights
    count_nonzero(x)
    return x.to_numpy()


def feat_row_sum_inv_normalize(x):
    """
    :param x: np.ndarray, raw features.
    :return:  np.ndarray, normalized features
    """
    x_feat = x.astype(dtype=np.float64)
    inv_x_rowsum = np.power(x_feat.sum(axis=1), -1).flatten()
    inv_x_rowsum[np.isinf(inv_x_rowsum)] = 0.
    x_diag_mat = np.diag(inv_x_rowsum)
    normalized_x = x_diag_mat.dot(x_feat)
    return normalized_x


def get_node2vec_embedding(data, num_nodes, edge_index, embedding_dim=300):
    """
    :param data: pd.DataFrame.
    :param num_nodes: int, number of nodes.
    :param edge_index: np.ndarray, shape = (2, edge_num)
    :return: np.ndarray, shape = (num_nodes, embedding_dim)
    """
    t1 = time.time()
    edge_index = torch.tensor(edge_index)
    train_indices = data['train_indices']
    test_indices = data['test_indices']
    total_indices = sorted(train_indices + test_indices, reverse=False)
    train_label = data['train_label']['label'].values

    node2vec = Node2Vec(num_nodes=num_nodes, embedding_dim=embedding_dim, walk_length=10,
                        context_size=5, walks_per_node=1)
    optimizer = Adam(node2vec.parameters(), lr=1e-1, weight_decay=1e-4)
    for i in range(10):
        optimizer.zero_grad()
        node2vec.forward(subset=torch.tensor(total_indices))
        loss = node2vec.loss(edge_index=edge_index)
        loss.backward()
        optimizer.step()
        print("loss at epoch{}: {}".format(i, loss.item()))

    x_feats = node2vec.embedding(torch.tensor(total_indices)).detach().numpy()
    print("Time cost for node2vec {}s".format(time.time() - t1))
    return x_feats
