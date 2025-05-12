# Weights & Biases 集成指南

本项目已集成[Weights & Biases](https://wandb.ai/)工具来跟踪和可视化联邦学习实验。

## 安装依赖

首先，确保已安装wandb：

```bash
pip install wandb
```

## 登录Weights & Biases

在使用前，需要先登录到wandb账户：

```bash
wandb login
```

## 使用方法

在运行实验时，添加`-uw True`参数启用wandb监控：

```bash
python main.py -data ISIC -m ResNet18 -algo FedAvg -gr 100 -nc 10 -jr 1.0 -uw True
```

其他wandb相关参数：

- `-uw` 或 `--use_wandb`: 是否启用wandb，默认为False
- `-wp` 或 `--wandb_project`: 项目名称，默认为"PFL"
- `-we` 或 `--wandb_entity`: 组织名称，默认为None（使用个人账户）

## 跟踪的指标

通过wandb可以监控的主要指标包括：

### 全局指标
- 全局测试准确率和AUC
- 全局训练损失
- 每轮训练时间

### 客户端指标
- 每个客户端的测试准确率和AUC（按轮次）
- 每轮选择的客户端ID列表
- 每轮成功参与聚合的客户端ID列表 
- 客户端准确率分布的可视化

### 统计指标
- 最终模型性能
- 多次运行的统计结果（平均值和标准差）

## 查看客户端变化

在wandb界面中，您可以：

1. 查看特定客户端的性能变化：
   - 在图表区选择 `client_X/test_accuracy` 指标查看单个客户端的准确率随轮次的变化
   - 使用 `client_X/test_auc` 查看AUC变化

2. 比较多个客户端：
   - 在图表中同时选择多个客户端的性能指标进行比较
   - 查看每轮的 `round_X/client_performance` 表格获取详细数据

3. 了解客户端参与情况：
   - 检查 `selected_clients` 和 `active_client_ids` 表格，了解每轮参与的客户端
   - 对比 `num_selected_clients` 和 `active_clients` 了解客户端的掉线情况

## 可视化

实验结果可在[wandb.ai](https://wandb.ai)网站上查看。登录后选择对应的项目，可以看到实验的详细记录、图表和比较。

## 示例wandb仪表盘

在wandb网站上，您可以创建自定义仪表盘来比较不同算法的性能，例如：

- 不同联邦学习算法的收敛速度对比
- 客户端间性能差异的可视化
- 通信开销与模型性能的权衡分析
- 客户端性能随时间的变化趋势分析

python main.py -data ISIC_2019 -m CNN -algo FedAvg -gr 200 -did 0 -uw True