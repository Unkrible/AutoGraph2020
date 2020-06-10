#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/5/12 23:20
# @Author:  Mecthew
import numpy as np
import torch_geometric.utils as gtils


class DropEdgeEachStep:
    def __init__(self, adj, unique_edges):
        self.adj = adj
        self.unique_edges = unique_edges
        self.counter = 0

    def drop_edges(self, edge_index, edge_weight, drop_rate=0.2):
        num_edges = len(self.unique_edges)
        num_preserved_edges = int(num_edges * (1-drop_rate))
        preserved_edges_idx = self.unique_edges[np.random.permutation(num_edges)[:num_preserved_edges]]
        perm = self._make_undirected(preserved_edges_idx)
        if self.counter == 0:
            print(f"Is undirected after drop edges: {gtils.is_undirected(edge_index[:, perm])}")
        self.counter += 1
        return edge_index[:, perm], edge_weight[perm]

    def _make_undirected(self, edge_index):
        symmetry = self.adj[edge_index]
        undirected = np.union1d(edge_index, symmetry)
        return undirected
