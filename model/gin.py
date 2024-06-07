import sys
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Callable, Optional, Union
from torch_geometric.typing import Adj, Tensor, SparseTensor, OptPairTensor, OptTensor, Size
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_geometric.nn.conv import MessagePassing, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import add_remaining_self_loops


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, nconvs=3, dropout=0, pooling='mean', **kwargs):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(nconvs):

            if i:
                nn = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            else:
                nn = Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        self.dropout = dropout
        self.pooling = pooling

    def forward(self, x, edge_index, batch, edge_weight=None):
        xs = []
        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.bns[i](x)
            xs.append(x)
        x = self.convs[-1](x, edge_index, edge_weight)
        xs.append(x)

        if self.pooling == 'mean':
            gs = [global_mean_pool(x, batch=batch) for x in xs]
        elif self.pooling == 'sum':
            gs = [global_add_pool(x, batch=batch) for x in xs]
        x = torch.cat(gs, dim=1)
        return x

    def get_embeddings(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1))
        x = self.forward(x, edge_index, batch)
        return x


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nconvs=5, training=True):
        super(Net, self).__init__()
        self.encoder = GIN(input_dim, hidden_dim, nconvs)
        self.embedding_dim = hidden_dim * nconvs
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        if training:
            self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0])
        y = self.encoder(x, edge_index, batch)
        y = self.proj_head(y)
        return y

