import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from mygin import G_GIN
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader


def run(args, device, dl_train, dl_test):
    model = G_GIN(input_dim=args.nfeat, hidden_dim=args.nhid, output_dim=args.nhid,
                  nconvs=args.layers).to(device)
    if hasattr(model, "project"):
        del model.project
    model.project = nn.Identity()
    score_list = []

    model.eval()
    with torch.no_grad():
        # tr feature
        X_val, Y_val = [], []
        for data in dl_train:
            data = data.to(args.device)
            x, edge_index, = data.x, data.edge_index
            batch = data.batch
            emb = model(edge_index, x, batch)
            X_val.append(emb)
            Y_val.append(data.y)
        X_val, Y_val = torch.cat(X_val, dim=0), torch.cat(Y_val, dim=0)
        num_features = X_val.shape[-1]
        loader_emb_val = DataLoader(TensorDataset(X_val, Y_val), batch_size=args.test_bs,
                                    shuffle=True, num_workers=0, )

        # te feature
        X_test, Y_test = [], []
        for data in dl_test:
            data = data.to(args.device)
            x, edge_index, = data.x, data.edge_index
            batch = data.batch
            emb = model(edge_index, x, batch)
            X_test.append(emb)
            Y_test.append(data.y)
        X_test, Y_test = torch.cat(X_test, dim=0), torch.cat(Y_test, dim=0)
        loader_emb_test = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.test_bs,
                                     shuffle=False, num_workers=0, )

    # -------------------------------------------------------------------------------------------------------------------------------------------------------#

    """LINEAR EVALUATION"""

    pred_head = nn.Linear(num_features, args.nclass).to(args.device)
    if args.test_opt == "sgd":
        opt = torch.optim.SGD(pred_head.parameters(), lr=args.test_lr, momentum=0.9, weight_decay=args.test_wd)
    elif args.test_opt == "adam":
        opt = torch.optim.Adam(pred_head.parameters(), lr=args.test_lr, weight_decay=args.test_wd)
    else:
        raise NotImplementedError

    # print("Linear Evaluation")
    pred_head.train()
    for _ in range(5):
        for _ in range(args.test_epoch):
            for x, y in loader_emb_val:
                loss = F.nll_loss(torch.log_softmax(pred_head(x), dim=1), y.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()

        pred_head.eval()
        with torch.no_grad():
            meta_loss, meta_acc, denominator = 0., 0., 0.
            true_list = []
            pred_list = []
            for x, y in loader_emb_test:
                l = pred_head(x)
                meta_loss += F.nll_loss(torch.log_softmax(l, dim=1), y.view(-1), reduction="sum")
                pred = l.argmax(dim=-1)
                pred_list.append(pred.cpu().numpy())
                true_list.append(y.view(-1).cpu().numpy())
            #     meta_acc += torch.eq(pred, y.view(-1)).float().sum()
            #     denominator += x.shape[0]
            # meta_loss /= denominator
            # meta_acc /= denominator
            true_list = np.concatenate(true_list, 0)
            pred_list = np.concatenate(pred_list, 0)
            score = accuracy_score(true_list, pred_list)
            score_list.append(score)
    return np.mean(score_list), np.std(score_list)

class TUEvaluator:
    def __init__(self):
        self.num_tasks = 1
        self.eval_metric = 'accuracy'

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'accuracy':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks should be {} but {} given'.format(self.num_tasks,
                                                                                             y_true.shape[1]))

            return y_true, y_pred
        else:
            raise ValueError('Undefined eval metric %s ' % self.eval_metric)

    def _eval_accuracy(self, y_true, y_pred):
        '''
            compute Accuracy score averaged across tasks
        '''
        acc_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            acc = accuracy_score(y_true[is_labeled], y_pred[is_labeled])
            acc_list.append(acc)

        return {'accuracy': sum(acc_list) / len(acc_list)}

    def eval(self, input_dict):
        y_true, y_pred = self._parse_and_check_input(input_dict)
        return self._eval_accuracy(y_true, y_pred)


def evaluation(args, device, data_syn):

    """PRETRAIN"""
    # model and opt
    model = G_GIN(input_dim=args.nfeat, hidden_dim=args.nhid, output_dim=args.nhid, nconvs=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    pre_epoch = 50
    # pretrain
    # print("Pretrain")
    model.train()
    for _ in range(1, pre_epoch + 1):
        x, edge_index = data_syn.x.to(device), data_syn.edge_index.to(device)
        batch, y = data_syn.batch.to(device), data_syn.y.to(device)
        output = model(edge_index, x, batch)
        loss = F.nll_loss(torch.log_softmax(output, -1), y.view(-1))
        # print(loss)
        # update
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


class EmbeddingEvaluation:
    def __init__(self, base_classifier, args, evaluator, device,
                 param_search=True):
        self.args = args
        self.base_classifier = base_classifier
        self.evaluator = evaluator
        self.eval_metric = evaluator.eval_metric
        self.device = device
        self.param_search = param_search
        if self.eval_metric == 'rmse':
            self.gscv_scoring_name = 'neg_root_mean_squared_error'
        elif self.eval_metric == 'mae':
            self.gscv_scoring_name = 'neg_mean_absolute_error'
        elif self.eval_metric == 'rocauc':
            self.gscv_scoring_name = 'roc_auc'
        elif self.eval_metric == 'accuracy':
            self.gscv_scoring_name = 'accuracy'
        else:
            raise ValueError('Undefined grid search scoring for metric %s ' % self.eval_metric)

        self.classifier = None

    def scorer(self, y_true, y_raw):
        input_dict = {"y_true": y_true, "y_pred": y_raw}
        score = self.evaluator.eval(input_dict)[self.eval_metric]
        return score

    def ee_binary_classification(self, train_emb, train_y, test_emb):
        if self.param_search:
            params_dict = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            self.classifier = make_pipeline(StandardScaler(),
                                            GridSearchCV(self.base_classifier, params_dict, cv=5,
                                                         scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
                                            )
        else:
            self.classifier = make_pipeline(StandardScaler(), self.base_classifier)

        self.classifier.fit(train_emb, np.squeeze(train_y))

        if self.eval_metric == 'accuracy':
            train_raw = self.classifier.predict(train_emb)
            test_raw = self.classifier.predict(test_emb)
        else:
            train_raw = self.classifier.predict_proba(train_emb)[:, 1]
            test_raw = self.classifier.predict_proba(test_emb)[:, 1]

        return np.expand_dims(train_raw, axis=1), np.expand_dims(test_raw, axis=1)

    def embedding_evaluation(self, data_syn, train_loader, test_loader):
        encoder = evaluation(self.args, self.device, data_syn)
        encoder.eval()
        train_emb, train_y = encoder.get_emb(train_loader, self.device)
        test_emb, test_y = encoder.get_emb(test_loader, self.device)

        train_raw, test_raw = self.ee_binary_classification(train_emb, train_y, test_emb)
        if train_y.ndim < 2:
            train_y = np.expand_dims(train_y, 1)
        if test_y.ndim < 2:
            test_y = np.expand_dims(test_y, 1)
        train_score = self.scorer(train_y, train_raw)
        test_score = self.scorer(test_y, test_raw)

        return train_score, test_score






