# -*- coding: utf-8 -*-
# @Time    : 2024/10/24
# @Author  : Bin Li
# @Email   : lybin1336258176@outlook.com
# @File    : AlexNet.py

import torch
import torch.nn as nn
import pytorch_model_summary


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10))

        def init_params(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

        self.net.apply(init_params)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    print(pytorch_model_summary.summary(AlexNet(),
                                        torch.rand(size=(1, 1, 224, 224), dtype=torch.float32),
                                        show_input=True,
                                        show_hierarchical=True))
