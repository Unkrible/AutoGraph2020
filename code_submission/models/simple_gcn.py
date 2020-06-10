#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/5/6 9:39
# @Author:  Mecthew
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SplineConv, GCNConv


class SimpleGCN(torch.nn.Module):

    # TODO: 网络太弱
    def __init__(
            self, num_layers=3, hidden=32, features_num=16, num_class=2, dropout=0.5,
            drop_edge_controller=None,
            **kwargs
    ):
        super(SimpleGCN, self).__init__()
        hidden = max(hidden, num_class * 2)
        # self.conv1 = SplineConv(features_num, hidden, dim=1, kernel_size=2)
        # self.conv2 = SplineConv(hidden, num_class, dim=1, kernel_size=2)
        self.conv1 = GCNConv(features_num, hidden * 2)
        self.conv2 = GCNConv(hidden * 2, num_class)
        self.dropout = dropout
        self.drop_edge_controller = drop_edge_controller

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # edge_index, edge_weight = self.drop_edge_controller.drop_edges(edge_index, edge_weight, 0.2)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__
