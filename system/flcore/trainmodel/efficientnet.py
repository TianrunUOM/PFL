import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetWithHead(nn.Module):
    """
    专门为联邦学习设计的EfficientNet，避免使用复杂的模型分离
    """
    def __init__(self, model_name='efficientnet_b0', num_classes=8, pretrained=True):
        super(EfficientNetWithHead, self).__init__()
        
        # 创建基础EfficientNet模型
        if model_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet_b1':
            base_model = models.efficientnet_b1(pretrained=pretrained) 
        elif model_name == 'efficientnet_b2':
            base_model = models.efficientnet_b2(pretrained=pretrained)
        elif model_name == 'efficientnet_b3':
            base_model = models.efficientnet_b3(pretrained=pretrained)
        elif model_name == 'efficientnet_b4':
            base_model = models.efficientnet_b4(pretrained=pretrained)
        elif model_name == 'efficientnet_b5':
            base_model = models.efficientnet_b5(pretrained=pretrained)
        elif model_name == 'efficientnet_b6':
            base_model = models.efficientnet_b6(pretrained=pretrained)
        elif model_name == 'efficientnet_b7':
            base_model = models.efficientnet_b7(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # 直接从原始模型提取所需的部分
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # 获取特征维度
        if hasattr(base_model, 'classifier') and isinstance(base_model.classifier, nn.Sequential):
            in_features = base_model.classifier[1].in_features
        elif hasattr(base_model, 'classifier') and isinstance(base_model.classifier, nn.Linear):
            in_features = base_model.classifier.in_features
        elif hasattr(base_model, 'fc'):
            in_features = base_model.fc.in_features
        else:
            # 使用默认值
            in_features = 1280
        
        # 创建自定义的分类头
        head_classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )
        
        # 设置所有需要的属性
        self.classifier = head_classifier
        self.head = head_classifier  # 与fc相同，提供多个访问点
        self.fc = head_classifier
    
    def forward(self, x):
        """完整的前向传播"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def base(self, x):
        """只执行特征提取部分"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def create_efficientnet(model_name='efficientnet_b0', num_classes=8, pretrained=True):
    """
    创建专用于联邦学习的EfficientNet模型
    """
    return EfficientNetWithHead(model_name, num_classes, pretrained) 