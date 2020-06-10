import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, ARMAConv


class NasPubmed(nn.Module):
    def __init__(self, features_num, num_class, num_layers=2, dropout=0.5, hidden=64, edge_num=1000, **kwargs):
        super(NasPubmed, self).__init__()
        hidden_dim = max(hidden, num_class * 2)
        his_dim, cur_dim, hidden_dim, output_dim = features_num, features_num, hidden_dim, hidden_dim
        multi_head = edge_num < 1400000
        self.cells = nn.ModuleList()
        for _ in range(num_layers):
            cell = NasPubmedCell(his_dim, cur_dim, hidden_dim, output_dim, multi_head)
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


class NasPubmedCell(nn.Module):
    def __init__(self, his_dim, cur_dim, hidden_dim, output_dim, multi_head):
        super(NasPubmedCell, self).__init__()
        self._cur_dim = cur_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        self.preprocessor_h = nn.Linear(his_dim, hidden_dim)
        self.preprocessor_x = nn.Linear(cur_dim, hidden_dim)
        self.headers = 8 if multi_head else 1
        self.gat8 = GATConv(hidden_dim, output_dim, heads=self.headers)
        self.arma = ARMAConv(hidden_dim, output_dim)

    def forward(self, h, x, edge_index, edge_weight):
        his = x
        x = self.preprocessor_x(x)
        h = self.preprocessor_h(h)
        o3 = F.leaky_relu(self.arma(h, edge_index, edge_weight))
        o4 = F.leaky_relu(self.gat8(x, edge_index))
        o5 = F.tanh(torch.cat([o3, o4], dim=1))
        return his, o5

    @property
    def output_dim(self):
        return self._output_dim * (1 + self.headers)

