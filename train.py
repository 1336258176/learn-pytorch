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
import torch.utils.tensorboard as tensorboard

import load_data
from net import AlexNet
from net import LeNet

epochs = 50
batch_size = 100
dropout = 0.5
lr = 0.001
resize = 224
dataset_name = 'FashionMNIST'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = tensorboard.SummaryWriter('./logs')
train_loader, test_loader = load_data.load_data(name=dataset_name, batch_size=batch_size, resize=resize)
# model = LeNet.LeNet(1).to(dev)
model = AlexNet.AlexNet().to(dev)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.1)

model_root = './model'
model_name = model.__class__.__name__


def train_loop(data_loader: data.DataLoader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.
    size = len(data_loader.dataset)
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % batch_size == 0:
            loss, current = loss.item(), batch_idx * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss / len(data_loader)


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
    return correct


if __name__ == '__main__':
    print(f"running on {dev}")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n-------------------------------------------")
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        acc = test_loop(test_loader, model, loss_fn, lr_scheduler=lr_scheduler)
        writer.add_scalar(dataset_name + '/' + model_name + '/train loss', train_loss, epoch)
        writer.add_scalar(dataset_name + '/' + model_name + '/accuracy rate', acc, epoch)
    writer.flush()
    writer.close()
    print("Training Done!")

    if os.path.exists(os.path.join(model_root, model_name + '.pth')):
        os.remove(os.path.join(model_root, model_name + '.pth'))
        print("Delete an existing model parameter.")
    torch.save(model.state_dict(), os.path.join(model_root, model_name + '.pth'))
    print("Model Saved!")
