import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.nn import GCNConv


class EmbGCN(torch.nn.Module):

    def __init__(self, num_layers=2, hidden=32, emb_dim=64, num_class=2, num_nodes=None, **kwargs):
        super(EmbGCN, self).__init__()
        hidden = max(hidden, num_class * 2)
        self.conv1 = GCNConv(emb_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.emb = Embedding(num_nodes, emb_dim)
        self.first_lin = Linear(emb_dim, hidden)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.emb.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight, node_index = data.x, data.edge_index, data.edge_weight, data.node_index
        x = self.emb(node_index)
        x1 = F.elu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.elu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_weight=edge_weight))
        x = x1 + x
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
