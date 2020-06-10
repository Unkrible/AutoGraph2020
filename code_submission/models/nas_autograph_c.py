#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-05-14

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, ARMAConv, SGConv, ChebConv, SAGEConv


class NasAutoGraphC(nn.Module):
    def __init__(self, features_num, num_class, num_layers=2, dropout=0.5, hidden=64, edge_num=1000, **kwargs):
        super(NasAutoGraphC, self).__init__()
        hidden_dim = max(hidden, num_class * 2)
        his_dim, cur_dim, hidden_dim, output_dim = features_num, features_num, hidden_dim, hidden_dim
        multi_head = edge_num < 1400000
        self.cells = nn.ModuleList()
        for _ in range(num_layers):
            cell = NasAutoGraphCCell(his_dim, cur_dim, hidden_dim, output_dim, multi_head)
            self.cells.append(cell)
            his_dim, cur_dim = cur_dim, cell.output_dim
        self.classifier = nn.Linear(cur_dim, num_class)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = x
        for cell in self.cells:
            h, x = cell(h, x, edge_index, edge_weight)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)


class NasAutoGraphCCell(nn.Module):
    # best structure: {'action': [1, 'sage', 1, 'cheb', 'linear', 'add'], 'hyper_param': [0.0005, 0.8, 1e-05, 256]}
    def __init__(self, his_dim, cur_dim, hidden_dim, output_dim, multi_head):
        super(NasAutoGraphCCell, self).__init__()
        self._cur_dim = cur_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        self.preprocessor_x = nn.Linear(cur_dim, hidden_dim)
        self.headers = 2 if multi_head else 1
        self.cheb = ChebConv(hidden_dim, output_dim, K=2, bias=True)
        self.sage = SAGEConv(hidden_dim, output_dim)
        self.linear = nn.Linear(self._output_dim, self._output_dim)

    def forward(self, h, x, edge_index, edge_weight):
        his = x
        x = self.preprocessor_x(x)
        o1 = F.leaky_relu(self.cheb(x, edge_index, edge_weight))
        o2 = F.leaky_relu(self.sage(x, edge_index, edge_weight))
        o3 = self.linear(torch.add(o1, o2))
        return his, o3

    @property
    def output_dim(self):
        return self._output_dim


