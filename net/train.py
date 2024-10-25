# -*- coding: utf-8 -*-
# @Time    : 2024/10/24
# @Author  : Bin Li
# @Email   : lybin1336258176@outlook.com
# @File    : train.py

import os.path
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim.lr_scheduler

import load_data
import LeNet

epochs = 50
batch_size = 100
dropout = 0.5
lr = 0.1
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = load_data.load_data(batch_size=batch_size)
model = LeNet.LeNet(1).to(dev)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.1)

model_root = '../model'
model_name = model.__class__.__name__


def train_loop(data_loader: data.DataLoader, model, loss_fn, optimizer):
    model.train()
    size = len(data_loader.dataset)
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(dev), y.to(dev)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_idx + 1) % batch_size == 0:
            loss, current = loss.item(), batch_idx * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(data_loader: data.DataLoader, model, loss_fn, lr_scheduler: torch.optim.lr_scheduler = None):
    model.eval()
    size = len(data_loader.dataset)
    num_batches = data_loader.batch_size
    test_loss, correct = 0., 0.

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(dev), y.to(dev)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    if lr_scheduler and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler.step(test_loss)
    elif lr_scheduler and not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler.step()
    else:
        pass

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    print(f"running on {dev}")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n-------------------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn, lr_scheduler=lr_scheduler)
    print("Training Done!")

    if os.path.exists(os.path.join(model_root, model_name + '.pth')):
        os.remove(os.path.join(model_root, model_name + '.pth'))
        print("Delete an existing model parameter.")
    torch.save(model.state_dict(), os.path.join(model_root, model_name + '.pth'))
    print("Model Saved!")
