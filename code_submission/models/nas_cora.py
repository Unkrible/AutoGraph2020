import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, ARMAConv


class NasCora(nn.Module):
    def __init__(self, features_num, num_class, num_layers=2, dropout=0.5, hidden=64, edge_num=1000, **kwargs):
        super(NasCora, self).__init__()
        hidden_dim = max(hidden, num_class * 2)
        multi_head = edge_num < 1400000
        cur_dim, hidden_dim, output_dim = features_num, hidden_dim, hidden_dim
        self.cells = nn.ModuleList()
        for _ in range(num_layers):
            cell = NasCoraCell(cur_dim, hidden_dim, output_dim, multi_head)
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


class NasCoraCell(nn.Module):
    def __init__(self, cur_dim, hidden_dim, output_dim, multi_head):
        super(NasCoraCell, self).__init__()
        self._cur_dim = cur_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self.headers = 6 if multi_head else 1

        self.preprocessor = nn.Linear(cur_dim, hidden_dim)
        self.gat6 = GATConv(hidden_dim, output_dim, heads=self.headers)
        self.gcn0 = GCNConv(hidden_dim, output_dim)
        self.gcn1 = GCNConv(hidden_dim, output_dim)
        self.arma = ARMAConv(output_dim * self.headers, output_dim)
    
    def forward(self, x, edge_index, edge_weight):
        h = self.preprocessor(x)
        h1 = F.leaky_relu(self.gat6(h, edge_index))
        h2 = F.leaky_relu(self.gcn0(h, edge_index, edge_weight=edge_weight))
        h3 = F.leaky_relu(self.gcn1(h, edge_index, edge_weight=edge_weight))
        h4 = F.leaky_relu(self.arma(h1, edge_index, edge_weight))
        out = torch.cat([h1, h2, h3, h4], dim=1)
        return F.tanh(out)

    @property
    def output_dim(self):
        return self._output_dim * (self.headers + 3)

