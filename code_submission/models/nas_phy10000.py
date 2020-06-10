#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-05-22

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import ARMAConv


class NasPhy10000(nn.Module):
    def __init__(self, features_num, num_class, num_layers=2, dropout=0.5, hidden=64, edge_num=1000, **kwargs):
        super(NasPhy10000, self).__init__()
        print(f"edge num {edge_num}")
        hidden_dim = max(hidden, num_class * 2)
        cur_dim, hidden_dim, output_dim = features_num, hidden_dim, hidden_dim
        multi_head = edge_num < 1400000
        self.cells = nn.ModuleList()
        for _ in range(num_layers):
            cell = NasPhy10000Cell(cur_dim, hidden_dim, output_dim, multi_head)
            self.cells.append(cell)
            cur_dim = cell.output_dim
        self.classifier = nn.Linear(cur_dim, num_class)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p=self.dropout, training=self.training)
        for cell in self.cells:
            x = cell(x, edge_index, edge_weight)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)


class NasPhy10000Cell(nn.Module):
    # best structure:{'action': [0, 'linear', 2, 'arma', 'tanh', 'add'], 'hyper_param': [0.001, 0.5, 0.0001, 128]}
    def __init__(self, cur_dim, hidden_dim, output_dim, multi_head):
        super(NasPhy10000Cell, self).__init__()
        self._cur_dim = cur_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        self.preprocessor = nn.Linear(cur_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.arma = ARMAConv(output_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.preprocessor(x)
        h1 = F.leaky_relu(self.linear(h))
        h2 = F.leaky_relu(self.arma(h1, edge_index, edge_weight=edge_weight))
        out = F.tanh(torch.add(h1, h2))
        return out

    @property
    def output_dim(self):
        return self._output_dim
