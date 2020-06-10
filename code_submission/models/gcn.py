import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):

    # TODO: 网络太弱
    def __init__(self, num_layers=3, hidden=32, features_num=16, num_class=2, **kwargs):
        super(GCN, self).__init__()
        hidden = max(hidden, num_class * 2)
        self.conv1 = GCNConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x1 = F.elu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.elu(self.first_lin(x))
        # TODO: dropout rate
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_weight=edge_weight))
        x = x1 + x
        x = F.dropout(x, p=0.5, training=self.training)
        # x = torch.cat([x1, x], dim=1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
