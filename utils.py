import torch
from torch.utils.data import Dataset
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == 'ours':

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def initialize_edge_weight(data):
    data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float)
    data.x = torch.as_tensor(data.x, dtype=torch.float32)
    return data


class RemoveEdgeAttr(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)

        data.x = data.x.float()
        return data

def tensor2data(x_list, adj_list):
    assert x_list.shape[0] == adj_list.shape[0]
    sampled = np.ndarray((x_list.size(0),), dtype=object)
    for i in range(x_list.shape[0]):
       x = x_list[i].squeeze(0)
       g = adj_list[i].squeeze(0).nonzero().T
       sampled[i] = (Data(x=x, edge_index=g))
    return sampled


def load_split(split_file):
    with open(split_file, 'r', encoding='utf-8') as fin:
        for i in fin:
            training_idx, val_idx, test_idx = i.strip().split('\t')
            training_idx = [int(x) for x in training_idx.split(' ')]
            val_idx = [int(x) for x in val_idx.split(' ')]
            test_idx = [int(x) for x in test_idx.split(' ')]
            split = [training_idx, val_idx, test_idx]
    return split


def avg_num_node(name):
    dataset2avg_num_node = {'MNIST':71, 'CIFAR10':118, 'DD':285, 'MUTAG':18,
                        'NCI1':30, 'NCI109':30,'PROTEINS':39,
                        'ogbg-molhiv':26, 'ogbg-molbbbp':24, 'ogbg-molbace':34}
    if name in dataset2avg_num_node:
        return dataset2avg_num_node[name]
    else:
        print("Unknown avg_num_node of the dataset {}".format(name))


def dense2batch(x_list, adj_list, y_list):
    sampled = np.ndarray((adj_list.size(0),), dtype=object)
    for i in range(adj_list.size(0)):
        x = x_list[i]
        g = adj_list[i].nonzero().T
        y = y_list[i].view(1, -1)
        sampled[i] = (Data(x=x, edge_index=g, y=y))
    training_set = SparseTensorDataset(sampled)
    loader = DataLoader(training_set, batch_size=128, shuffle=True, num_workers=0)
    return loader


class InfIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)


class SynDataset(Dataset):
    def __init__(self, batch_data, y_syn):
        super(SynDataset, self).__init__()
        self.batch_data = batch_data
        self.y_syn = y_syn
        assert len(self.batch_data) == len(self.batch_data)

    def __getitem__(self, index):
        return self.batch_data[index], self.y_syn[index]

    def __len__(self):
        return len(self.batch_data)


class SparseTensorDataset(Dataset):
    def __init__(self, data):
        self.data  = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
