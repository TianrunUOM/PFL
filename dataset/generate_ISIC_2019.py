# generate_ISIC2019.py

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.dataset_utils import check, separate_data, split_data, save_file
import gc

def classify_label(y, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, label in enumerate(y):
        list1[int(label)].append(idx)
    return list1


def clients_indices(list_label2indices: list, num_classes: int, num_clients: int, non_iid_alpha: float, seed=None):
    indices2targets = []
    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))

    batch_indices = build_non_iid_by_dirichlet(seed=seed,
                                               indices2targets=indices2targets,
                                               non_iid_alpha=non_iid_alpha,
                                               num_classes=num_classes,
                                               num_indices=len(indices2targets),
                                               n_workers=num_clients)
    import functools
    indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)

    list_client2indices = partition_balance(indices_dirichlet, num_clients)

    return list_client2indices

def partition_balance(idxs, num_split: int):

    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts

def build_non_iid_by_dirichlet(
    seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    random_state = np.random.RandomState(seed)
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    import math
    num_splits = math.ceil(n_workers / n_auxi_workers)

    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        #n_workers=10
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        _idx_batch = None
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch


class SkinDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        assert self.mode in ["train", "test", "all"]  # 修改这里，允许all模式
        self.transform = transform
        csv_file = os.path.join(root, "/mnt/vhd/ctr/pFL+MOE/Dataset/ISIC_2019/ISIC_2019_Training_GroundTruth.csv")
        
        self.file = pd.read_csv(csv_file)

        self.images = self.file["image"].values
        self.labels = self.file.iloc[:, 1:].values.astype("int")
        self.targets = np.argmax(self.labels, axis=1)

        initial_len = len(self.images)

        # data split
        np.random.seed(0)
        idxs = np.random.permutation(initial_len)
        self.images = self.images[idxs]
        self.targets = self.targets[idxs]
        
        if self.mode == "all":
                # 使用所有数据
            pass  # 不修改images和targets
        elif self.mode == "train":
            self.images = self.images[:int(0.8*initial_len)]
            self.targets = self.targets[:int(0.8*initial_len)]
        else:  # test
            self.images = self.images[int(0.8*initial_len):]
            self.targets = self.targets[int(0.8*initial_len):]
            
            
        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 8

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = os.path.join(
            self.root, "ISIC_2019_Training_Input", self.images[index] + ".jpg")
        img = Image.open(image_name).convert("RGB")
        label = self.targets[index]
        if self.transform is not None:
            if not isinstance(self.transform, list):
                img = self.transform(img)
            else:
                img0 = self.transform[0](img)
                img1 = self.transform[1](img)
                img = [img0, img1]
        return img, label

    def __len__(self):
        return len(self.images)
    
    
def generate_dataset(dir_path, num_clients, niid, balance, partition, num_classes):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)   
    
    
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return
    
    # import pdb; pdb.set_trace()
    

    #  数据处理
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
    
    
    # 修改数据处理，确保所有图像大小一致
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 改用固定大小的resize
        # 或者用这个也可以：
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),   
        normalize               
    ])
    
    
    # all_dataset = SkinDataset(root='/mnt/vhd/ctr/pFL+MOE/Dataset/ISIC_2019', mode="all", transform=basic_transform)
    
    # dataset_image = all_dataset.images
    # dataset_label = all_dataset.targets
    
    trainset = SkinDataset(root='/mnt/vhd/ctr/pFL+MOE/Dataset/ISIC_2019', mode="train", transform=basic_transform)
    testset = SkinDataset(root='/mnt/vhd/ctr/pFL+MOE/Dataset/ISIC_2019', mode="test", transform=basic_transform)
    
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    # import pdb; pdb.set_trace()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.targets), shuffle=False, num_workers=16)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.targets), shuffle=False, num_workers=16)
    
    
    print(f'trainloader批次数: {len(trainloader)}')
    print(f'testloader批次数: {len(testloader)}')
    
    

    for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data
    dataset_image = []
    dataset_label = []
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    
    # 保存处理好的数据
    save_dir = '/mnt/vhd/ctr/pFL+MOE/saved_data'      
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'isic_dataset_full.npz')
    
    # print(f"正在保存数据到 {save_path}...")
    # print(f"数据形状: 图像 {dataset_image.shape}, 标签 {dataset_label.shape}")
    # print(f"划分方法: {'非独立同分布' if niid else '独立同分布'}, {'平衡' if balance else '不平衡'}, 分区: {partition}")
    
    np.savez_compressed(
        save_path,
        images=dataset_image,
        labels=dataset_label
    )
    print(f"数据保存完成!")



   
    # train_data, test_data = split_data(X, y)
    # save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
    #     statistic, niid, balance, partition)




 
import sys
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    # dir_path = "/mnt/vhd/ctr/pFL+MOE/Dataset/ISIC_2019"
    dir_path = "./ISIC_2019/"
    num_clients = 10
    num_classes = 8
    
    # First: 生成global数据
    # generate_dataset(dir_path, num_clients, niid, balance, partition, num_classes)
    
    # Second: 处理global数据
    #################################### 保存好数据后处理 #################################### 
    # 加载数据
    loaded_data = np.load('/mnt/vhd/ctr/pFL+MOE/saved_data/isic_dataset_full.npz')
    dataset_image = loaded_data['images']
    dataset_label = loaded_data['labels']
    import pdb; pdb.set_trace()
    
     # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    
    # 在保存文件前创建目录
    train_dir = './ISIC_2019/train/'
    test_dir = './ISIC_2019/test/'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 前面自己处理
    # X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
    #                                 niid, balance, partition, class_per_client=8)
    X, y, statistic = separate_data(
        data=(dataset_image, dataset_label),
        num_clients=num_clients,
        num_classes=num_classes,
        niid=True,
        balance=True,
        partition='equal_dir',
        class_per_client=8
    )
    import pdb; pdb.set_trace()
    
    del dataset_image
    del dataset_label
    gc.collect()
    train_data, test_data = split_data(X, y)
    
    del X
    del y
    gc.collect()
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)

    

   
   


