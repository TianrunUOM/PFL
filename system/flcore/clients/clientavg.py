import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # 用于跟踪训练损失
        self.train_loss_dict = {'total_loss': 0, 'count': 0}
        self.train_loss = 0  # 平均训练损失
        
    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        # 记录每轮的训练损失
        epoch_losses = []
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            batch_losses = []
            
            # 调整学习率
            self.adjust_learning_rate()
            
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
            
            # 记录每个epoch的平均损失
            epoch_losses.append(np.mean(batch_losses))
            print(f'Client {self.id} - Epoch {epoch + 1}/{max_local_epochs}, Loss: {epoch_losses[-1]:.4f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        # 记录这轮训练的平均损失
        self.train_loss = np.mean(epoch_losses)
