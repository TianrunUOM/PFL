import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现
        
        Args:
            alpha (Tensor, optional): 各类别的权重系数。shape应为[num_classes]。默认为None
            gamma (float): focusing参数。默认为2.0
            reduction (str): 'none' | 'mean' | 'sum'。默认为'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input: 模型输出的logits，shape为[N, C]
            target: 真实标签，shape为[N]
        """
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss 