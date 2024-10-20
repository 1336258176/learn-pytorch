# -*- coding: utf-8 -*-
# @Time    : 2024/10/19
# @Author  : Bin Li
# @Email   : lybin1336258176@outlook.com
# @File    : LeNet.py
import torch
import torch.nn.functional as F
import torch.nn as nn

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
