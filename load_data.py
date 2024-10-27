# -*- coding: utf-8 -*-
# @Time    : 2024/10/24
# @Author  : Bin Li
# @Email   : lybin1336258176@outlook.com
# @File    : load_data.py

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torch.utils.tensorboard as tensorboard

data_root = './data/'
log_dir = './logs'


def get_dataloader_workers() -> int:
    return 4


def download_CIFAR10(is_train: bool, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    dataset = torchvision.datasets.CIFAR10(root=data_root, train=is_train, download=True, transform=trans)
    return dataset


def download_FashionMNIST(is_train: bool, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    dataset = torchvision.datasets.FashionMNIST(root=data_root, train=is_train, download=True, transform=trans)
    return dataset


def load_data(name: str, batch_size: int = 100, resize=None):
    if name == 'FashionMNIST':
        train_dataset = download_FashionMNIST(is_train=True, resize=resize)
        test_dataset = download_FashionMNIST(is_train=False, resize=resize)
    elif name == 'CIFAR10':
        train_dataset = download_CIFAR10(is_train=True, resize=resize)
        test_dataset = download_CIFAR10(is_train=False, resize=resize)
    else:
        raise ValueError(f"string mismatch, expected one of {'CIFAR10', 'FashionMNIST'}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=get_dataloader_workers(), drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=get_dataloader_workers(), drop_last=False)
    return train_loader, test_loader


if __name__ == '__main__':
    dataset_name = 'FashionMNIST'
    train_loader, test_loader = load_data(name=dataset_name, batch_size=100, resize=224)
    # writer = tensorboard.SummaryWriter(log_dir=log_dir)
    # for idx, (images, labels) in enumerate(test_loader):
    #     writer.add_images(dataset_name + '/images', images, idx)
    # writer.flush()
    # writer.close()
    it = iter(test_loader)
    images, labels = next(it)
    print(images.shape)
    print(labels.shape)
