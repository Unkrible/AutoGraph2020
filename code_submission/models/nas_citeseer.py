import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class NasCiteseer(nn.Module):
    def __init__(self, features_num, num_class, num_layers=2, dropout=0.5, hidden=64, edge_num=1000, **kwargs):
        super(NasCiteseer, self).__init__()
        print(f"edge num {edge_num}")
        hidden_dim = max(hidden, num_class * 2)
        cur_dim, hidden_dim, output_dim = features_num, hidden_dim, hidden_dim
        multi_head = edge_num < 1400000
        self.cells = nn.ModuleList()
        for _ in range(num_layers):
            cell = NasCiteseerCell(cur_dim, hidden_dim, output_dim, multi_head)
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


class NasCiteseerCell(nn.Module):
    def __init__(self, cur_dim, hidden_dim, output_dim, multi_head):
        super(NasCiteseerCell, self).__init__()
        self._cur_dim = cur_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        self.preprocessor = nn.Linear(cur_dim, hidden_dim)
        self.heads = 6 if multi_head else 1
        self.gat6 = GATConv(hidden_dim, output_dim, heads=self.heads)
        self.linear = nn.Linear(self._output_dim * (self.heads + 1), self._output_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.preprocessor(x)
        h1 = F.leaky_relu(self.gat6(h, edge_index))
        out = self.linear(torch.cat([h, h1], dim=1))
        return out

    @property
    def output_dim(self):
        return self._output_dim
