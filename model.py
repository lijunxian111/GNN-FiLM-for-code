# -*- coding: utf-8 -*-
from torch_geometric.nn.conv import FiLMConv, HeteroConv
import torch.nn
import torch.nn.functional as F
from collections import OrderedDict
from read_data import construct_graph

#GNN-FiLM模型
class GNNFilm(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, num_relations,
                 dropout=0.5):
        super(GNNFilm, self).__init__()
        self.embedding = torch.nn.Embedding(in_channels, hidden_channels)
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(FiLMConv(hidden_channels, hidden_channels, num_relations))
        for _ in range(n_layers - 1):
            self.convs.append(FiLMConv(hidden_channels, hidden_channels, num_relations))
        self.norms = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin_l = torch.nn.Sequential(OrderedDict([
            ('lin1', torch.nn.Linear(hidden_channels, int(hidden_channels // 4), bias=True)),
            ('lrelu', torch.nn.LeakyReLU(0.2)),
            ('lin2', torch.nn.Linear(int(hidden_channels // 4), out_channels, bias=True))]))

    def forward(self, x, edge_index, edge_type):
        x = self.embedding(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_l(x)
        return x

if __name__ == "__main__":
    model = GNNFilm(in_channels=7, hidden_channels=64, out_channels=2, n_layers=2, num_relations=5)
    g = construct_graph('data/chart.pkl')
    #print(g.edge_types)
    g = g.to_homogeneous()
    #print(torch.unique(g.edge_type))
    #edge_type = g.edge_type
    res = model(g.x, g.edge_index, g.edge_type)
    print(res.shape)
