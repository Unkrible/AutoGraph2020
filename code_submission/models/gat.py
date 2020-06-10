import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, features_num, num_class, num_layers=3, hidden=32, **kwargs):
        super(GAT, self).__init__()
        hidden = max(hidden, num_class * 2)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden, hidden))
        self.input_lin = nn.Linear(features_num, hidden)
        self.output_lin = nn.Linear(hidden, num_class)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.leaky_relu(self.input_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.leaky_relu(conv(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.output_lin(x)
        return F.log_softmax(x, dim=-1)

