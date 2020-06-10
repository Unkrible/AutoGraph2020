import torch
from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


# 这是graph classification级别的网络
class SAGE(torch.nn.Module):
    def __init__(self, data, device, embed_dim=128, features_num=16, num_class=2, **kwargs):
        super(SAGE, self).__init__()

        self.conv1 = SAGEConv(features_num, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_class)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.data = data
        self.item_embedding = data.x
        self.device = device

    def forward(self, indices):
        data = self.data
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        print(f"batch {len(x)}")
        batch = torch.tensor([len(x)], dtype=torch.long).to(self.device)

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))

        # x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        print(f"gmp {gmp(x, batch).shape}, gap {gap(x, batch).shape}")
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))

        # x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))

        # x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        print(f"x1 {x1.shape}, x2 {x2.shape}, x3 {x3.shape}")

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        print(indices.shape)
        print(x.shape)
        x = torch.sigmoid(self.lin3(x))[indices, :]
        print(x.shape)

        return F.log_softmax(x, dim=-1)
