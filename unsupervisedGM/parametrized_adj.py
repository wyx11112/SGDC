import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from itertools import product
import numpy as np

class PGE(nn.Module):

    def __init__(self, n_syn, nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
        super(PGE, self).__init__()
        if args.dataset in ['PROTEINS', 'NCI1', 'DD', 'NCI109']:
           nhid = 128
        if args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace', 'ogbg-molpcba', 'ogbg-code2']:
           nhid = 128

        self.n_syn = n_syn
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        edge_index_list = []
        edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        for i in range(n_syn):
            edge_index_list.append(np.expand_dims(edge_index.T, 0))
        edge_index = np.concatenate(edge_index_list, 0)
        self.edge_index = edge_index
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.cnt = 0
        self.args = args
        self.nnodes = nnodes

    def forward(self, x):
        # if self.args.dataset == 'reddit' and self.args.reduction_rate >= 0.01:
        #     edge_index = self.edge_index
        #     n_part = 5
        #     splits = np.array_split(np.arange(edge_index.shape[1]), n_part)
        #     edge_embed = []
        #     for idx in splits:
        #         tmp_edge_embed = torch.cat([x[edge_index[0][idx]],
        #                 x[edge_index[1][idx]]], axis=1)
        #         for ix, layer in enumerate(self.layers):
        #             tmp_edge_embed = layer(tmp_edge_embed)
        #             if ix != len(self.layers) - 1:
        #                 tmp_edge_embed = self.bns[ix](tmp_edge_embed)
        #                 tmp_edge_embed = F.relu(tmp_edge_embed)
        #         edge_embed.append(tmp_edge_embed)
        #     edge_embed = torch.cat(edge_embed)
        # else:
        edge_index = self.edge_index
        # edge_embed = torch.cat([x[edge_index[:,0]],[edge_index[:,1]]], axis=-1)
        edge_emb_list = []
        for i in range(x.shape[0]):
            x_ = x[i].squeeze(0)
            edge_ = edge_index[i]
            edge_embed = torch.cat([x_[edge_[0]], x_[edge_[1]]], dim=1)
            for ix, layer in enumerate(self.layers):
                edge_embed = layer(edge_embed)
                if ix != len(self.layers) - 1:
                    edge_embed = self.bns[ix](edge_embed)
                    edge_embed = F.relu(edge_embed)
            edge_emb_list.append(edge_embed.unsqueeze(0))
        edge_embed = torch.cat(edge_emb_list, 0) #(n_syn, e, 2*d)

        adj = edge_embed.reshape(self.n_syn, self.nnodes, self.nnodes)

        adj = (adj + adj.transpose(1, 2))/2
        adj = torch.sigmoid(adj)
        adj_list = []
        for i in range(adj.shape[0]):
            adj_ = adj[0]
            adj_ = adj_ - torch.diag(torch.diag(adj_, 0))
            adj_list.append(adj_.unsqueeze(0))
        return torch.cat(adj_list, 0)

    @torch.no_grad()
    def inference(self, x):
        # self.eval()
        adj_syn = self.forward(x)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)

