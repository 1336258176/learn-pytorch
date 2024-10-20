import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data_dir = './data'
model_save_dir = './model'
batch_size = 64
epochs = 200
lr = 0.003

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


def download_data():
    training_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor()
    )
    return training_data, test_data


def load_data(data_arrays, batch_sizes):
    return torch.utils.data.DataLoader(data_arrays, batch_size=batch_sizes, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net_stack(x)


def train_loop(data_loader: torch.utils.data.DataLoader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(dev), y.to(dev)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(data_loader: torch.utils.data.DataLoader, model):
    model.eval()
    batch_size = len(data_loader)
    size = len(data_loader.dataset)

    loss = 0.
    correct = 0.

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(dev), y.to(dev)
            y_pred = model(X)
            loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= batch_size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f} \n")


if __name__ == '__main__':
    training_data, testing_data = download_data()
    train_dataloader = load_data(training_data, batch_size)
    test_dataloader = load_data(testing_data, batch_size)
    model = NeuralNetwork().to(dev)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(epochs):
        print(f"Epoch {i + 1}\n------------------------------------")
        train_loop(train_dataloader, model, loss_fn, opt)
        test_loop(test_dataloader, model)
    print("Training Done!")

    if os.path.exists(os.path.join(model_save_dir, 'FashionMNIST.pth')):
        print('A model with the same name already exists and will be replaced with a new one.')
        os.remove(os.path.join(model_save_dir, 'FashionMNIST.pth'))
    torch.save(model.state_dict(), os.path.join(model_save_dir, 'FashionMNIST.pth'))
    # onnx_program = torch.onnx.dynamo_export(model, train_dataloader.dataset)
    # onnx_program.save(os.path.join(model_save_dir, "FashionMNIST.onnx"))
