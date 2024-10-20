import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 1)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


if __name__ == "__main__":
    model = NN()
    X = torch.rand(5, 2)
    print(X)
    y = model(X)
    print(y)
