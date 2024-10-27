from d2l import torch as d2l
import torchvision
import torch


if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.05, 10, 100
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(torchvision.models.AlexNet(10), train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
