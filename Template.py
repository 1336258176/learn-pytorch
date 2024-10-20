import torch
import torch.nn as nn


def sigmoid(x: torch.Tensor):
    """
    y = 1 / (1 + e^(-x))
    常被用作神经网络的激活函数，将变量映射到[0,1]之间。
    """
    return 1 / (1 + torch.exp(-x))


def ReLU(x: torch.Tensor):
    """
    y = max(0, x)
    线性整流函数，又称修正线性单元ReLU，是一种人工神经网络中常用的激活函数
    """
    return torch.max(x, torch.zeros_like(x))


def softmax(x: torch.Tensor):
    """
    y = e^x / sum(e^x)
    用于多类分类问题的激活函数，在多类分类问题中，超过两个类标签则需要类成员关系。
    对于长度为K的任意实向量，Softmax函数可以将其压缩为长度为K，值在[0,1]范围内，并且向量中元素的总和为1的实向量。
    """
    x_exp = torch.exp(x)
    sum = torch.sum(x_exp, dim=1, keepdim=True)
    return x_exp / sum


def dropout(x: torch.Tensor, p: float):
    """
    在前向传播的时候，让某个神经元的激活值以一定的概率p（伯努利分布）停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征
    """
    assert 0.0 <= p <= 1.0
    if p == 0.0:
        return torch.zeros_like(x)
    elif p == 1.0:
        return x
    mask = (torch.rand(x.shape) > p).float()
    return (x * mask) / (1.0 - p)


def corr2d(X: torch.Tensor, K: torch.Tensor):
    """
    二维卷积运算，卷积主要用于图像的特征提取
    卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出。 所以，卷积层中的两个被训练的参数是卷积核权重和标量偏置。
    用卷积层代替全连接层的一个好处是：模型更简洁、所需的参数更少。
    """
    h, w = K.shape
    res = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return res


def pooling2d(X: torch.Tensor, pooling_size: int, mode: str = 'max'):
    """
    二维池化运算
    """
    h, w = pooling_size
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = (X[i:i + h, j:j + w]).max()
            elif mode == 'avg':
                Y[i, j] = (X[i:i + h, j:j + w]).mean()
            else:
                return None
    return Y


if __name__ == '__main__':
    # x = torch.rand((2, 3))
    # print(x)
    # print(dropout(x, 0.5))
    # print(softmax(x))
    # print(ReLU(x))
    # X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    # K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    # print(corr2d(X, K))
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    print(pooling2d(X, (2, 2), 'avg'))
