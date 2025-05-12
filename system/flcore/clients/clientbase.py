import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from flcore.losses.focal_loss import FocalLoss


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # 学习率预热相关
        self.learning_rate_warmup = args.learning_rate_warmup
        self.warmup_epochs = args.warmup_epochs
        self.current_epoch = 0

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # 设置损失函数
        if hasattr(args, 'use_focal_loss') and args.use_focal_loss:
            # 如果需要类别权重，可以根据数据分布计算
            if hasattr(args, 'focal_alpha') and args.focal_alpha is not None:
                # 将字符串转换为浮点数列表，例如"0.1,0.2,0.3" -> [0.1, 0.2, 0.3]
                alpha = torch.tensor([float(x) for x in args.focal_alpha.split(',')]).to(self.device)
            else:
                # 自动计算类别权重
                alpha = self._compute_class_weights()
            self.loss = FocalLoss(alpha=alpha, gamma=args.focal_gamma if hasattr(args, 'focal_gamma') else 2.0)
        else:
            self.loss = nn.CrossEntropyLoss()
        
        # 根据参数选择优化器
        if args.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        elif args.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True
            )
        elif args.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=args.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {args.optimizer} not supported")

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                # 收集预测和真实标签用于计算BACC
                all_predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        # 计算BACC，不乘以test_num
        bacc = metrics.balanced_accuracy_score(all_targets, all_predictions)
        
        return test_acc, test_num, auc, bacc  # 直接返回bacc，不乘以test_num

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def _compute_class_weights(self):
        """
        根据训练数据分布自动计算类别权重
        使用有效样本数（effective number）方法计算权重
        Reference: "Class-Balanced Loss Based on Effective Number of Samples"
        """
        # 统计每个类别的样本数
        class_counts = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        
        for _, y in trainloader:
            for label in y:
                class_counts[label] += 1
        
        # 避免除零错误
        class_counts = torch.clamp(class_counts, min=1)
        
        # 计算有效样本数
        beta = 0.9999  # 论文建议的参数
        effective_nums = (1.0 - beta ** class_counts) / (1.0 - beta)
        
        # 计算权重
        weights = (1.0 / effective_nums) / torch.sum(1.0 / effective_nums)
        
        return weights.to(self.device)

    def adjust_learning_rate(self):
        """调整学习率（包括预热）"""
        if self.learning_rate_warmup and self.current_epoch < self.warmup_epochs:
            # 线性预热
            lr = self.learning_rate * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            
        self.current_epoch += 1
