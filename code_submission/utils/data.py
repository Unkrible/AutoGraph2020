import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from collections import defaultdict

from utils import get_logger

logger = get_logger("DEBUG")


class GraphDataset(Dataset):

    def __init__(self, data):
        super(GraphDataset, self).__init__()
        self.data = data
        self.length = data.train_mask.sum()
        logger.info(f"Graph dataset: length({self.length})")
        self.indices = (data.train_mask == 1).nonzero()

    def __getitem__(self, index: int):
        index = self.indices[index]
        return index

    def __len__(self) -> int:
        return self.length

    def resample(self):
        return self


class GraphSampleDataset(Dataset):

    def __init__(
            self, data, n_class, y_train,
    ):
        super(GraphSampleDataset, self).__init__()
        self.data = data
        self.class_info = self._init_info(n_class, y_train)
        self.train_indices = (data.train_mask == 1).nonzero()
        self.indices, self.length = None, None
        self.n_class = n_class
        self.min_num, self.max_num, self.sample_nums = self._init_sample_num(
            len(y_train) / n_class
        )

    def _init_sample_num(self, n_mean):
        num = [self.class_info[i]['num'] for i in range(self.n_class)]
        n_min = min(num)
        n_max = max(num)
        n_max = int(min(n_max * 0.8, n_mean * 1.2))
        n_min = int(max(n_min * 1.5, n_mean * 0.5))
        sample_nums = [max(n_min, min(n_max, ele)) for ele in num]
        for i in range(self.n_class):
            print(f"Sample {sample_nums[i]} / {num[i]} from class {i}")
        return n_min, n_max, sample_nums

    def resample(self):
        self.indices = self._init_indices(self.n_class)
        self.length = len(self.indices)
        return self

    def _init_indices(self, n_class):
        all_indices = []
        for i in range(n_class):
            num = self.class_info[i]['num']
            indices = self.class_info[i]['indices']
            sampled_indices = []
            sampled_num = self.sample_nums[i]
            while sampled_num > 0:
                cur = min(sampled_num, num)
                sampled_indices.append(np.random.permutation(indices)[:cur])
                sampled_num -= cur
            all_indices += sampled_indices
        return np.concatenate(all_indices)

    def _init_info(self, n_class, y_train):
        class_info = defaultdict(dict)
        for i in range(n_class):
            indices = np.where(y_train[:] == i)[0]
            num = len(indices)
            class_info[i]['num'] = num
            class_info[i]['indices'] = indices
        return class_info

    def __getitem__(self, index: int):
        index = self.train_indices[self.indices[index]]
        return index

    def __len__(self):
        return self.length


class Sampler:
    def __init__(self, data, num_edges, device):
        self.data = data
        self.device = device
        self._origin_num_edges = num_edges
        self.adj, self.unique_edges, self.num_edges = None, None, None

    def _construct_adj(self):
        self.adj, self.unique_edges = Sampler.__construct_adj(self._origin_num_edges, self.data.edge_index)
        self.num_edges = len(self.unique_edges)
        print(f"num edge {self._origin_num_edges}, unique edge {self.num_edges}")

    @staticmethod
    def __construct_adj(num_edges, edges_tensor):
        unique = np.zeros(num_edges, dtype=np.bool)
        adj = np.zeros(num_edges, dtype=np.int)
        edges = edges_tensor.cpu().numpy()
        edges_dict = defaultdict(lambda: 0)
        for i in range(num_edges):
            if not (edges[1, i], edges[0, i]) in edges_dict:
                unique[i] = True
            edges_dict[(edges[0, i], edges[1, i])] = i
        for i in range(num_edges):
            adj[i] = edges_dict[(edges[1, i], edges[0, i])]
        del edges_dict
        return adj, np.argwhere(unique)

    def stub_sampler(self):
        return self.data.to(self.device)

    def _make_undirected(self, edge_index):
        symmetry = self.adj[edge_index]
        undirected = np.union1d(edge_index, symmetry)
        print(f"Before undirected {len(edge_index)}, after undirected {len(undirected)}")
        return undirected

    def random_edge_sampler(self, percent=1.0):
        """
        Randomly drop edge
        Args:
            percent: preserve edges' percent

        Returns: data

        """

        if percent >= 1.0:
            return self.stub_sampler()

        if self.adj is None:
            self._construct_adj()

        data = self.data
        num_preserved_edges = int(percent * self.num_edges)
        perm = self.unique_edges[np.random.permutation(self.num_edges)[:num_preserved_edges]]
        perm = self._make_undirected(perm)
        random_data = Data(
            x=data.x, y=data.y,
            train_indices=data.train_indices, train_mask=data.train_mask,
            test_indices=data.test_indices, test_mask=data.test_mask,
            edge_index=data.edge_index[:, perm], edge_weight=data.edge_weight[perm]
        )
        if hasattr(data, "valid_indices"):
            random_data.valid_indices = data.valid_indices
            random_data.valid_mask = data.valid_mask
        return random_data.to(self.device)
