import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from sklearn import metrics


class clientFML(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.alpha = args.alpha
        self.beta = args.beta

        self.global_model = copy.deepcopy(args.model)
        self.optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_g, 
            gamma=args.learning_rate_decay_gamma
        )

        self.KL = nn.KLDivLoss()

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        self.global_model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                output_g = self.global_model(x)
                loss = self.loss(output, y) * self.alpha + self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) * (1-self.alpha)
                loss_g = self.loss(output_g, y) * self.beta + self.KL(F.log_softmax(output_g, dim=1), F.softmax(output, dim=1)) * (1-self.beta)

                self.optimizer.zero_grad()
                self.optimizer_g.zero_grad()
                loss.backward(retain_graph=True)
                loss_g.backward()
                # prevent divergency on specifical tasks
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 10)
                self.optimizer.step()
                self.optimizer_g.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_g.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
    def set_parameters(self, global_model):
        for new_param, old_param in zip(global_model.parameters(), self.global_model.parameters()):
            old_param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
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
        
        # 计算BACC，不乘以test_num
        bacc = metrics.balanced_accuracy_score(all_targets, all_predictions)
        
        return test_acc, test_num, 0, bacc  # 直接返回bacc，不乘以test_num

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()
        self.global_model.eval()

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
                output_g = self.global_model(x)
                loss = self.loss(output, y) * self.alpha + self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) * (1-self.alpha)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num