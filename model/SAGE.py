import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.inits import uniform


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, bias=False, batch_norm=False):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.convs = torch.nn.ModuleList([])

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, hidden_channels, bias=bias))
        else:
            if batch_norm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(in_channels, hidden_channels, bias=bias))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, bias=bias))
                if batch_norm:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, bias=bias))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x) if self.batch_norm else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    @torch.no_grad()
    def inference(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            # x = self.bns[i](x) if self.batch_norm else x
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z = self.encoder(x, edge_index)
        z1 = self.encoder(x1, edge_index1)
        z2 = self.encoder(x2, edge_index2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)