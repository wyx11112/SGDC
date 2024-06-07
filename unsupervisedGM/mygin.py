import sys
import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Callable, Optional, Union
from torch_geometric.typing import Adj, Tensor, SparseTensor, OptPairTensor, OptTensor, Size
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_geometric.nn.conv import MessagePassing, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import add_remaining_self_loops


class G_GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nconvs=3, dropout=0, pooling='mean', **kwargs):
        super().__init__()

        self.convs = torch.nn.ModuleList([])
        self.convs.append(GINConv(torch.nn.Linear(input_dim, hidden_dim), train_eps=True))

        for _ in range(nconvs - 1):
            self.convs.append(GINConv(torch.nn.Linear(hidden_dim, hidden_dim), train_eps=True))
        # self.project = Linear(hidden_dim, output_dim)
        self.norms = torch.nn.ModuleList([])
        for _ in range(nconvs):
            if nconvs == 1:
                norm = torch.nn.Identity()
            else:
                norm = torch.nn.BatchNorm1d(hidden_dim)
            self.norms.append(norm)

        self.dropout = dropout
        self.pooling = pooling

    def forward(self, edge_index, x, batch, edge_weight=None):

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.norms[i](x)
            x = F.relu(x)
        # x = self.convs[-1](x, edge_index, edge_weight)

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch=batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch=batch)

        return x

    def get_emb(self, loader, device):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if isinstance(data, list):
                    data = data[0].to(device)
                data = data.to(device)
                batch, x, edge_index = data.batch, data.x, data.edge_index

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)

                for i in range(len(self.convs) - 1):
                    x = self.convs[i](x, edge_index)
                    x = self.norms[i](x)
                    x = F.relu(x)
                x = self.convs[-1](x, edge_index)

                if self.pooling == 'mean':
                    x = global_mean_pool(x, batch=batch)
                elif self.pooling == 'sum':
                    x = global_add_pool(x, batch=batch)
                # x = self.forward(edge_index, x, batch, None)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    # only for SSL!!!
    def loss_cal(self, x, x_aug):

        T = 1
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (torch.einsum('i,j->ij', x_abs, x_aug_abs) + 1e-7)
        # is_nan = torch.isnan(sim_matrix)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss
