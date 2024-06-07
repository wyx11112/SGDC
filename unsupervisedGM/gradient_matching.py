import argparse
import os
import sys

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from mygin import G_GIN
from parametrized_adj import PGE
from utils import *
from aug import aug
from pretrain import EmbeddingEvaluation, TUEvaluator, run
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge, LogisticRegression
import time


def denseTobatch(x_syn, adj, original):
    # x_syn = x_syn.data.cpu()
    # adj = adj.data.cpu()
    sampled = np.ndarray((adj.size(0),), dtype=object)
    for i in range(adj.size(0)):
        x = x_syn[i]
        g = adj[i].nonzero().T
        y = original.y[i].view(1, -1)
        sampled[i] = (Data(x=x, edge_index=g, y=y))
    batch = Batch.from_data_list(sampled)
    return batch


def main(args):
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    pwd = os.path.dirname(os.path.realpath(__file__))
    par = os.path.dirname(pwd)
    path = '/data0/wangyuxiang/Anaconda3/.dataset/'
    if args.dataset in ['PROTEINS', 'NCI1', 'DD', 'NCI109']:
        # dataset = TUDataset(path, name=args.dataset, use_node_attr=True, transform=T.Compose([T.NormalizeFeatures()]))
        dataset = TUDataset(path, name=args.dataset, use_node_attr=True)

        """ The original split is randomly generated for the TU datasets """

        split_file = os.path.join(par, 'splits/OriginalSplit/' + args.dataset + '_split.txt')
        # print(cur)
        # split_file = "../splits/OriginalSplit/" + args.dataset + "_split.txt"
        if not os.path.exists(split_file):
            print("{} original split does not exist!".format(args.dataset))
            sys.exit()
        splits = load_split(split_file)
        training_set = dataset[splits[0]]
        # training_set.x = training_set.x[splits[0]]
        val_set = dataset[splits[1]]
        test_set = dataset[splits[2]]
        nclass = dataset.num_classes
        nfeat = dataset.num_node_features
        loader_train = DataLoader(training_set, batch_size=args.outer_bs)
        loader_val = DataLoader(val_set, batch_size=64)
        loader_test = DataLoader(test_set, batch_size=64)
        args.nfeat = nfeat
        args.nclass = nclass
        evaluator = TUEvaluator()
        ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True), args,
                                 evaluator, device, param_search=True)
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
        loader_train = DataLoader(training_set, batch_size=1000)
        loader_val = DataLoader(val_set, batch_size=64)
        loader_test = DataLoader(test_set, batch_size=64)
        args.nfeat = nfeat
        args.nclass = nclass
        args.task_type = dataset.task_type
        args.num_tasks = dataset.num_tasks
        args.metric = 'rocauc'
        evaluator = Evaluator(name=args.dataset)
        ee = EmbeddingEvaluation(LogisticRegression(dual=False, fit_intercept=True, max_iter=10000),
                                 args, evaluator, device)
    else:
        raise NameError

    """ Initialize the syntehtic graphs with truncated real-world graphs """
    if args.initialize == "Herding":
        file_name = os.path.join(par, "splits/HerdingSelect/" + args.dataset + "_herding_" + str(args.gpc) + ".txt")
    elif args.initialize == "KCenter":
        file_name = os.path.join(par, "splits/KCenterSelect/" + args.dataset + "_kcenter_" + str(args.gpc) + ".txt")
    elif args.initialize == "Random":
        file_name = os.path.join(par, "splits/RandomSelect/" + args.dataset + "_random_" + str(args.gpc) + ".txt")
    with open(file_name, 'r', encoding='utf-8') as fin:
        selected_idx = [int(x) for x in fin.read().strip().split(' ')]

    # sampled = [training_set[i] for i in selected_idx]
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
    syn_init = Batch.from_data_list(selected_list)

    x_syn, mask = to_dense_batch(syn_init.x, batch=syn_init.batch, max_num_nodes=None)
    adj_syn = to_dense_adj(syn_init.edge_index, batch=syn_init.batch, max_num_nodes=None)
    # pge = PGE(n_syn=x_syn.shape[0], nfeat=x_syn.shape[-1], nnodes=x_syn.shape[1], device=device, args=args).to(device)

    x_syn = x_syn.type(torch.cuda.FloatTensor)
    x_syn.requires_grad_(True)
    optimizer_feat = torch.optim.Adam([x_syn], lr=args.lr_feat)
    # optimizer_pge = torch.optim.Adam(pge.parameters(), lr=args.lr_adj)

    model = G_GIN(input_dim=args.nfeat, hidden_dim=args.nhid, output_dim=nclass, nconvs=args.layers).to(device)
    epoch = args.epochs
    test_score_list = []
    train_score_list = []
    st = time.time()
    for it in range(1, epoch + 1):
        model.train()
        model_parameters = list(model.parameters())
        optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model, weight_decay=args.weight_decay)

        outer_loop = args.outer_loop
        inner_loop = args.inner_loop
        for ol in range(outer_loop):
            loss = torch.tensor(0.0).to(device)
            # adj_syn = pge(x_syn)
            # adj_syn =
            input_syn = denseTobatch(x_syn, adj_syn, syn_init)
            input_syn.x.requires_grad_(True)

            num = 0
            for data_t in loader_train:
                num += 1
                data_t, data_t_aug = aug(data_t, args.aug)
                node_num, _ = data_t.x.size()
                data_t = data_t.to(device)
                x = model(data_t.edge_index, data_t.x, data_t.batch)
                data_t_aug = data_t_aug.to(device)
                x_aug = model(data_t_aug.edge_index, data_t_aug.x, data_t_aug.batch)
                loss_real = model.loss_cal(x, x_aug)
                # print(f'Epoch: {it}, outer loop: {ol}, loss_real: {loss_real.item():.4f}')
                gw_real = torch.autograd.grad(loss_real, model_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))

                data_syn, data_syn_aug = aug(input_syn.clone().detach(), args.aug, is_syn=True)
                data_syn, data_syn_aug = data_syn.to(device), data_syn_aug.to(device)
                output_syn = model(data_syn.edge_index, data_syn.x, data_syn.batch)
                output_syn_aug = model(data_syn_aug.edge_index, data_syn_aug.x, data_syn_aug.batch)
                loss_syn = model.loss_cal(output_syn, output_syn_aug)
                # print(f'Epoch: {it}, outer loop: {ol}, loss_syn: {loss_syn.item():.4f}')
                gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                loss += match_loss(gw_syn, gw_real, args, device)
                # print(f'Epoch: {it}, outer loop: {ol}, match loss: {loss.item():.4f}, num: {num}')

            print(f'Epoch: {it}, Gradient matching loss: {loss.item():.4f}', )

            x_syn_inner = x_syn.detach()
            # print(x_syn_inner)
            # adj_syn_inner = pge(x_syn_inner)
            input_syn_inner = denseTobatch(x_syn_inner, adj_syn, input_syn)
            input_syn_inner, input_syn_inner_aug = aug(input_syn_inner.cpu(), args.aug, is_syn=True)
            for i in range(inner_loop):
                optimizer_model.zero_grad()
                input_syn_inner, input_syn_inner_aug = input_syn_inner.to(device), input_syn_inner_aug.to(device)
                x_inner, edge_index_inner, batch_inner = input_syn_inner.x, input_syn_inner.edge_index, input_syn_inner.batch
                output_syn_inner = model(edge_index_inner, x_inner, batch_inner)
                x_inner_aug, edge_index_inner_aug, batch_inner_aug = input_syn_inner_aug.x, input_syn_inner_aug.edge_index, input_syn_inner_aug.batch
                output_syn_inner_aug = model(edge_index_inner_aug, x_inner_aug, batch_inner_aug)
                loss_syn_inner = model.loss_cal(output_syn_inner, output_syn_inner_aug)
                loss_syn_inner.backward()
                optimizer_model.step()

            optimizer_feat.zero_grad()
            loss.backward()
            if it % 2 == 0:
                optimizer_feat.step()
    ed = time.time()
    print('time cost: ', ed - st)
        if it % args.eval_every == 0:
            if args.dataset in ['PROTEINS', 'NCI1', 'DD', 'NCI109']:
                score_avg_list = []
                score_std_list = []
                score_mean, score_std = run(args, device, loader_train, loader_test)
                print(f'Epoch: {it}, score_mean: {score_mean:.4f}, score_std: {score_std:.4f}')
                score_avg_list.append(np.round(score_mean, 4))
                score_std_list.append(np.round(score_std, 2))
                fo1 = open(f'plot/{args.dataset}_avg.txt', 'w')
                for e in score_avg_list[:-1]:
                    fo1.write(str(e) + ' ')
                fo1.write(str(score_avg_list[-1]))
                fo1.write('\n')

                fo2 = open(f'plot/{args.dataset}_std.txt', 'w')
                for e in score_std_list[:-1]:
                    fo2.write(str(e) + ' ')
                fo2.write(str(score_std_list[-1]))
                fo2.write('\n')
            else:
                train_score, test_score = ee.embedding_evaluation(input_syn_inner, loader_val, loader_test)
                train_score_list.append(round(train_score, 2))
                test_score_list.append(round(test_score, 2))
                print(f'Epoch: {it}, train_score: {train_score:.4f}, test_score: {test_score:.4f}')

    # fo = open(f'log/{args.dataset}_test.txt', 'w')
    # for e in test_score_list[:-1]:
    #     fo.write(str(e) + ' ')
    # fo.write(str(test_score_list[-1]))
    # fo.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--dataset', type=str, default='NCI1', help='dataset name')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--dis_metric', type=str, default='mse')
    parser.add_argument('--initialize', type=str, default='Herding',
                        choices=['Herding', 'KCenter', 'Random'],
                        help="The initialization of the synthetic graphs")
    parser.add_argument('--gpc', type=int, default=50, help='Number of graphs per class to be synthetized.')
    parser.add_argument('--load_epoch', type=int, default=50, help='Different epochs pretrain model to be loaded.')
    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--outer_loop', type=int, default=10)
    parser.add_argument('--inner_loop', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)

    # model param
    parser.add_argument('--train_model', type=str, default='gin',
                        help='gin for TUDataset, gin_var for ogbg dataset')
    parser.add_argument('--test_model', type=str, default='gin',
                        help='gin for TUDataset, gin_var for ogbg dataset')
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--lr_feat', type=float, default=0.0001)
    parser.add_argument('--lr_model', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    # parser.add_argument('--lr_model', type=float, default=0.0001)
    # parser.add_argument('--lr_model', type=float, default=0.0001)
    # outer
    parser.add_argument('--outer_bs', type=int, default=64)
    parser.add_argument('--outer_opt', type=str, default='adam')
    parser.add_argument('--outer_wd', type=float, default=0.0)
    parser.add_argument('--outer_lr', type=float, default=0.0001)

    parser.add_argument('--test_algorithm', type=str, default="linear_evaluation")
    parser.add_argument('--test_opt', type=str, default="adam")
    parser.add_argument('--test_epoch', type=int, default=300)  # for linear_evaluation
    # parser.add_argument('--test_epoch', type=int, default=50) # for full_finetune
    parser.add_argument('--test_bs', type=float, default=128)
    parser.add_argument('--test_lr', type=float, default=0.001)  # for linear_evaluation
    # parser.add_argument('--test_lr', type=float, default=0.05) # for full_finetune
    parser.add_argument('--test_wd', type=float, default=2e-5)
    parser.add_argument('--outer_grad_norm', type=float, default=0.1)






    args = parser.parse_args()
    main(args)
