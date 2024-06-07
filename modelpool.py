import numpy as np

import torch

from model.wrapper import get_model


class ModelPool:
    def __init__(self, args, device):
        self.device = device
        self.online_iteration = args.online_iteration
        self.online_min_iteration = args.online_min_iteration
        self.online_batch_size = args.online_batch_size
        self.num_models = args.num_models
        self.model_name = args.train_model

        # model func
        self.model_func = lambda _: get_model(args.train_model, args, args.num_target_features).to(device)

        # opt func
        if args.online_opt == "sgd":
            self.opt_func = lambda param: torch.optim.SGD(param, lr=args.online_lr, momentum=0.9,
                                                          weight_decay=args.online_wd)
        elif args.online_opt == "adam":
            self.opt_func = lambda param: torch.optim.Adam(param, lr=args.online_lr, weight_decay=args.online_wd)
        else:
            raise NotImplementedError

        # loss func
        self.loss_func = args.loss_func

        self.iterations = [0] * self.num_models
        self.models = [self.model_func(None) for _ in range(self.num_models)]
        self.opts = [self.opt_func(self.models[i].parameters()) for i in range(self.num_models)]

    def init(self, loader):
        for idx in range(self.num_models):
            online_iteration = np.random.randint(self.online_min_iteration, self.online_iteration + 1)
            self.iterations[idx] = online_iteration
            model = self.models[idx]
            opt = self.opts[idx]

            model.train()
            for _ in range(online_iteration):
                for data_syn in loader:
                    x, edge_index = data_syn.x.to(self.device), data_syn.edge_index.to(self.device)
                    batch, y = data_syn.batch.to(self.device), data_syn.y.to(self.device)
                    if self.model_name == 'gin':
                        loss = self.loss_func(model(edge_index, x, batch), y)
                    elif self.model_name == 'gin_var':
                        edge_attr = data_syn.edge_attr.to(self.device)
                        edge_weight = data_syn.edge_weight if hasattr(data_syn, 'edge_weight') else None
                        x, _ = model(batch, x, edge_index, edge_attr, edge_weight)
                        loss = self.loss_func(x, y)
                    else:
                        raise NotImplementedError
                    opt.zero_grad()
                    # print(loss.item())
                    loss.backward()
                    opt.step()

    def update(self, idx, loader):
        # reset
        if self.iterations[idx] >= self.online_iteration:
            self.iterations[idx] = self.online_min_iteration
            self.models[idx] = self.model_func(None)
            self.opts[idx] = self.opt_func(self.models[idx].parameters())
            model = self.models[idx]
            opt = self.opts[idx]

            model.train()
            for _ in range(self.online_min_iteration):
                for data_syn in loader:
                    x, edge_index = data_syn.x.to(self.device), data_syn.edge_index.to(self.device)
                    batch, y = data_syn.batch.to(self.device), data_syn.y.to(self.device)
                    if self.model_name == 'gin':
                        loss = self.loss_func(model(edge_index, x, batch), y)
                    elif self.model_name == 'gin_var':
                        edge_attr = data_syn.edge_attr.to(self.device)
                        edge_weight = data_syn.edge_weight if hasattr(data_syn, 'edge_weight') else None
                        x, _ = model(batch, x, edge_index, edge_attr, edge_weight)
                        loss = self.loss_func(x, y)
                    else:
                        raise NotImplementedError
                    # print('Inner loss:', loss.item())
                    loss.backward()
                    opt.step()

        # train the model for 1 step
        else:
            self.iterations[idx] = self.iterations[idx] + 1
            model = self.models[idx]
            opt = self.opts[idx]

            model.train()
            for data_syn in loader:
                x, edge_index = data_syn.x.to(self.device), data_syn.edge_index.to(self.device)
                batch, y = data_syn.batch.to(self.device), data_syn.y.to(self.device)
                if self.model_name == 'gin':
                    loss = self.loss_func(model(edge_index, x, batch), y)
                elif self.model_name == 'gin_var':
                    edge_attr = data_syn.edge_attr.to(self.device)
                    edge_weight = data_syn.edge_weight if hasattr(data_syn, 'edge_weight') else None
                    x, _ = model(batch, x, edge_index, edge_attr, edge_weight)
                    loss = self.loss_func(x, y)
                else:
                    raise NotImplementedError
