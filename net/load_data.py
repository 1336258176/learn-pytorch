# -*- coding: utf-8 -*-
# @Time    : 2024/10/24
# @Author  : Bin Li
# @Email   : lybin1336258176@outlook.com
# @File    : load_data.py

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

data_root = '../data/'
num_works = 4


def load_data(batch_size: int = 100):
    train_data = torchvision.datasets.FashionMNIST(data_root, train=True, download=True,
                                                   transform=transforms.ToTensor())
    test_data = torchvision.datasets.FashionMNIST(data_root, train=False, download=True,
                                                  transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = load_data(100)
    it = iter(train_loader)
    x, y = next(it)
    print(x.shape, y.shape)
