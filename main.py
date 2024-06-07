import argparse
import os
import sys
import torch
import random
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from model.mygin import G_GIN
from model.molecule_encoder import MoleculeEncoder
from utils import *
from modelpool import ModelPool
from pretrain.wrapper import get_algorithm
from pretrain.gntk import LiteNTK
from antk import TNTK
import time
from icecream import ic


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu_id)
    args.device = device

    if args.seed is None:
        args.seed = random.randint(0, 9999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dist == 'mse':
        args.loss_func = lambda x, y: F.mse_loss(x, y)
    else:
        raise NotImplementedError

    print('================================')
    print('dataset: {}'.format(args.dataset))
    print('gpc: {}'.format(args.gpc))
    print('hidden_dim: {}'.format(args.nhid))
    print('outer_epoch: {}'.format(args.outer_epoch))
    print('outer_lr: {}'.format(args.outer_lr))
    print('initialize: {}'.format(args.initialize))
    print('load_epoch: {}'.format(args.load_epoch))
    print('train_model: {}'.format(args.train_model))
    print('test_model: {}'.format(args.test_model))
    print('outer_grad_norm: {}'.format(args.outer_grad_norm))
    print('================================')

    path = '/data0/wangyuxiang/Anaconda3/.dataset/'
    if args.dataset in ['PROTEINS', 'NCI1', 'DD', 'NCI109']:
        dataset = TUDataset(path, name=args.dataset, use_node_attr=True, transform=T.Compose([T.NormalizeFeatures()]))

        """ The original split is randomly generated for the TU datasets """
        split_file = "splits/OriginalSplit/" + args.dataset + "_split.txt"
        if not os.path.exists(split_file):
            print("{} original split does not exist!".format(args.dataset))
            sys.exit()
        splits = load_split(split_file)
        training_set = dataset[splits[0]]
        training_set.x = training_set.x[splits[0]]
        val_set = dataset[splits[1]]
        test_set = dataset[splits[2]]
        nclass = dataset.num_classes
        nfeat = dataset.num_node_features
        loader_train = DataLoader(training_set, batch_size=args.outer_bs)
        loader_val = DataLoader(val_set, batch_size=64)
        loader_test = DataLoader(test_set, batch_size=64)
        args.nfeat = nfeat
        args.nclass = nclass
        args.metric = 'acc'
        evaluator = None
    elif args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace', 'ogbg-molpcba', 'ogbg-code2']:  # ogb
        dataset = PygGraphPropPredDataset(root=path, name=args.dataset,
                                          transform=T.Compose([initialize_edge_weight]))

        """ The original split is provided by the OGB datasets """
        splits = dataset.get_idx_split()
        training_set = dataset[splits["train"]]
        val_set = dataset[splits["valid"]]
        test_set = dataset[splits["test"]]
        nclass = dataset.num_classes
        nfeat = dataset.num_node_features
        loader_train = DataLoader(training_set, batch_size=args.outer_bs)
        loader_val = DataLoader(val_set, batch_size=64)
        loader_test = DataLoader(test_set, batch_size=64)
        args.nfeat = nfeat
        args.nclass = nclass
        args.task_type = dataset.task_type
        args.num_tasks = dataset.num_tasks
        args.metric = 'rocauc'
        evaluator = Evaluator(name=args.dataset)
    else:
        raise NameError

    if args.train_model == 'gin':
        target_model = G_GIN(input_dim=nfeat, hidden_dim=args.nhid,
                             output_dim=nclass, nconvs=args.layers,
                             training=False).to(device)
    elif args.train_model == 'gin_var':
        target_model = MoleculeEncoder(emb_dim=args.nhid, num_gc_layers=args.layers,
                                       drop_ratio=args.drop_ratio, pooling_type=args.pooling_type).to(device)
    else:
        raise NotImplementedError

    target_model.load_state_dict(
        torch.load(f'./pretrain/checkpoint/{args.dataset}_{args.train_model}_{str(args.load_epoch)}.cpt'),
        strict=False)

    """ Initialize the syntehtic graphs with truncated real-world graphs """
    if args.initialize == "Herding":
        file_name = "./splits/HerdingSelect/" + args.dataset + "_herding_" + str(args.gpc) + ".txt"
    elif args.initialize == "KCenter":
        file_name = "./splits/KCenterSelect/" + args.dataset + "_kcenter_" + str(args.gpc) + ".txt"
    elif args.initialize == "Random":
        file_name = "./splits/RandomSelect/" + args.dataset + "_random_" + str(args.gpc) + ".txt"
    with open(file_name, 'r', encoding='utf-8') as fin:
        selected_idx = [int(x) for x in fin.read().strip().split(' ')]

    batch_real_list = []
    y_real_list = []
    target_model.eval()
    with torch.no_grad():
        for data_train in loader_train:
            batch_real_list.append(data_train)
            data_train = data_train.to(device)
            x, edge_index, batch = data_train.x, data_train.edge_index, data_train.batch
            if args.train_model == 'gin':
                y_real = target_model(edge_index, x, batch)

            elif args.train_model == 'gin_var':
                edge_attr = data_train.edge_attr
                edge_weight = data_train.edge_weight if hasattr(data_train, 'edge_weight') else None
                y_real, _ = target_model(batch, x, edge_index, edge_attr, edge_weight)
            else:
                raise NotImplementedError
            y_real_list.append(y_real.cpu())
    real_dataset = SynDataset(batch_real_list, y_real_list)
    real_loader = DataLoader(real_dataset, batch_size=1, shuffle=True)

    selected_list = []
    for i in selected_idx:
        sub_graph = Batch.from_data_list([training_set[i]])
        sub_x = sub_graph.x

        n = sub_x.shape[0]
        d = sub_x.shape[1]
        identity = torch.eye(n)
        identity_edge_index = identity.nonzero().T
        sub_graph.edge_index = identity_edge_index
        selected_list.append(sub_graph)
    batch_syn = Batch.from_data_list(selected_list)
    batch_y_real = batch_syn.y
    # sampled = [training_set[i] for i in selected_idx]
    # batch_syn = Batch.from_data_list(sampled)
    target_model.eval()
    with torch.no_grad():
        batch_ = batch_syn.to(device)
        x, edge_index, batch = batch_.x.to(device), batch_.edge_index.to(device), batch_.batch.to(device)
        if args.train_model == 'gin':
            y_syn = target_model(edge_index, x, batch)
        elif args.train_model == 'gin_var':
            edge_attr = batch_.edge_attr
            edge_weight = batch_.edge_weight if hasattr(batch, 'edge_weight') else None
            y_syn, _ = target_model(batch, x, edge_index, edge_attr, edge_weight)
    del target_model

    x_syn, mask = to_dense_batch(batch_syn.x, batch=batch_syn.batch, max_num_nodes=None)
    adj_syn = to_dense_adj(batch_syn.edge_index, batch=batch_syn.batch, max_num_nodes=None)
    x_syn = x_syn.type(torch.cuda.FloatTensor)
    x_syn.requires_grad_(True)
    # adj_syn.requires_grad_(True)
    y_syn.requires_grad_(True)
    args.num_target_features = x_syn.shape[1]

    syn_loader = dense2batch(x_syn, adj_syn, y_syn)

    # outer opt
    if args.outer_opt == "sgd":
        outer_opt = torch.optim.SGD([x_syn, adj_syn, y_syn], lr=args.outer_lr, momentum=0.5, weight_decay=args.outer_wd)
    elif args.outer_opt == "adam":
        # outer_opt = torch.optim.Adam([x_syn, adj_syn, y_syn], lr=args.outer_lr, weight_decay=args.outer_wd)
        outer_opt = torch.optim.Adam([x_syn, y_syn], lr=args.outer_lr, weight_decay=args.outer_wd)
    else:
        raise NotImplementedError

    model_pool = ModelPool(args, device)
    model_pool.init(syn_loader)

    pretrain = get_algorithm('pretrain_ours')
    test_algo = get_algorithm(f"{args.test_algorithm}")
    train_algo = get_algorithm('pretrain_ssl')
    # init_model = pretrain.run(args, device, args.test_model, syn_loader)
    # eval_score_mean, eval_score_std = test_algo.run(args, device, args.test_model, init_model, loader_train, loader_test, evaluator)
    # print(f'Pretrain Score Mean ({args.metric}): {eval_score_mean:.4f}, Score Std: {eval_score_std:.4f}')
    # del init_model

    score_avg_list = []
    score_std_list = []
    best_score = 0

    # tntk = LiteNTK(args.layers, num_mlp_layers=1, scale=args.scale, reg_lambda=args.reg_lambda).to(device)
    tntk = TNTK(args.layers, num_mlp_layers=1, scale=args.scale, reg_lambda=args.reg_lambda).to(device)
    st = time.time()
    for outer_step in range(1, args.outer_epoch + 1):
        outer_loss, new_loader, batch_save = train_algo.run(args, device, real_loader, model_pool,
                                                (x_syn, adj_syn, y_syn), outer_opt, tntk)
    # ed = time.time()
    # cost = ed - st
    # print(f'Training Time: {cost:.1f}')
        print(f"Epoch: {outer_step}, Outer loss: {outer_loss:.4f}")
        if outer_step % args.eval_every == 0:
        #     # le
            init_model = pretrain.run(args, device, args.test_model, new_loader)
            score_mean, score_std = test_algo.run(args, device, args.test_model, init_model,
                                                  loader_train, loader_test, evaluator)

            # save syn data
            # if score_mean > best_score:
            #     batch_X_S, batch_A_S = batch_save
            #     save_name = "./synthetic_graphs/" + args.dataset + "_gpc_" + str(args.gpc) + ".pt"
            #     torch.save([batch_X_S.detach().cpu(), batch_A_S.detach().cpu(), batch_y_real], save_name)
            #     best_score = score_mean
            #     print('save success')

            score_avg_list.append(score_mean)
            score_std_list.append(score_std)
            print(f'Epoch: {outer_step}, Test Score Mean({args.metric}): {score_mean:.4f}, '
                  f'Test Score Std({args.metric}): {score_std:.4f}')
            del init_model

        #     fo1 = open(f'plot/{args.dataset}_avg.txt', 'w')
        #     for e in score_avg_list[:-1]:
        #         fo1.write(str(e) + ' ')
        #     fo1.write(str(score_avg_list[-1]))
        #     fo1.write('\n')
        #
        #     fo2 = open(f'plot/{args.dataset}_std.txt', 'w')
        #     for e in score_std_list[:-1]:
        #         fo2.write(str(e) + ' ')
        #     fo2.write(str(score_std_list[-1]))
        #     fo2.write('\n')
        # fo1.close()
        # fo2.close()

    # acc_list = np.array(score_std_list)
    # print('Best Test Score:', np.max(acc_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--dataset', type=str, default='ogbg-molbbbp', help='dataset name')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dist', type=str, default='mse')
    parser.add_argument('--initialize', type=str, default='Herding',
                        choices=['Herding', 'KCenter', 'Random'],
                        help="The initialization of the synthetic graphs")
    parser.add_argument('--gpc', type=int, default=10, help='Number of graphs per class to be synthetized.')
    parser.add_argument('--load_epoch', type=int, default=50, help='Different epochs pretrain model to be loaded.')

    # model param
    parser.add_argument('--train_model', type=str, default='gin',
                        help='gin for TUDataset, gin_var for ogbg dataset')
    parser.add_argument('--test_model', type=str, default='gin',
                        help='gin for TUDataset, gin_var for ogbg dataset')
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--drop_ratio', type=float, default=0.5)
    parser.add_argument('--pooling_type', type=str, default='standard')

    # outer
    parser.add_argument('--outer_bs', type=int, default=64)
    parser.add_argument('--outer_opt', type=str, default='adam')
    parser.add_argument('--outer_wd', type=float, default=0.0)
    parser.add_argument('--outer_lr', type=float, default=0.0001)
    parser.add_argument('--outer_epoch', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--outer_grad_norm', type=float, default=0.1)

    # hparms for online
    parser.add_argument('--online_opt', type=str, default="adam")
    parser.add_argument('--online_iteration', type=int, default=50)
    parser.add_argument('--online_lr', type=float, default=0.01)
    parser.add_argument('--online_wd', type=float, default=0.0)
    parser.add_argument('--online_batch_size', type=int, default=128)
    parser.add_argument('--online_min_iteration', type=int, default=20)
    parser.add_argument('--num_models', type=int, default=1)

    # hparms for pretrain
    parser.add_argument('--pre_epoch', type=int, default=100)
    parser.add_argument('--pre_lr', type=float, default=0.01)
    parser.add_argument('--pre_wd', type=float, default=2e-5)

    # hparms for test
    parser.add_argument('--test_algorithm', type=str, default="linear_evaluation")
    parser.add_argument('--test_opt', type=str, default="adam")
    parser.add_argument('--test_epoch', type=int, default=300)  # for linear_evaluation
    # parser.add_argument('--test_epoch', type=int, default=50) # for full_finetune
    parser.add_argument('--test_bs', type=float, default=128)
    parser.add_argument('--test_lr', type=float, default=0.001)  # for linear_evaluation
    # parser.add_argument('--test_lr', type=float, default=0.05) # for full_finetune
    parser.add_argument('--test_wd', type=float, default=2e-5)

    # hparms for GNTK
    parser.add_argument('--scale', type=str, default='uniform', choices=['uniform', 'degree'],
                        help="The normalization method of GNTK")
    parser.add_argument('--reg_lambda', type=float, default=1e-6,
                        help='the lambda hyperparameter of the KRR.')
    parser.add_argument('--orth_reg', type=float, default=1e-3,
                        help='the regularization parameter of the orthogonal_loss.')

    args = parser.parse_args()
    main(args)
