import torch
import torch.nn as nn
import torch.utils.data as data

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


def generateData(w, b, sample_nums):
    X = torch.normal(0, 1, (sample_nums, len(w)))
    y = torch.matmul(X, w) + b
    y_offset = y + torch.normal(0, 0.1, y.shape)
    return X, y_offset.reshape(-1, 1)


def load_array(data_arrays, batch_sizes, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size=batch_sizes, shuffle=is_train)


if __name__ == '__main__':
    # init
    SampleNum = 100
    Epochs = 10
    w_true = torch.tensor([2, -3.4])
    b_true = 4.2
    batch_size = 10

    features, labels = generateData(w_true, b_true, SampleNum)
    data_iter = load_array((features, labels), batch_size)
    for epoch in range(Epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f"Epoch: {epoch + 1}, Loss: {l:f}")
