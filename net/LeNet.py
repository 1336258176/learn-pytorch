# -*- coding: utf-8 -*-
# @Time    : 2024/10/24
# @Author  : Bin Li
# @Email   : lybin1336258176@outlook.com
# @File    : LeNet.py

import torch
import torch.nn as nn
import pytorch_model_summary


class LeNet(nn.Module):
    def __init__(self, in_planes: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

        def init_params(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        self.net.apply(init_params)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    net = LeNet(1)
    print(pytorch_model_summary.summary(net, torch.rand(size=(1, 1, 28, 28), dtype=torch.float32), show_input=True,
                                        show_hierarchical=True))
