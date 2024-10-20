# -*- coding: utf-8 -*-
# @Time    : 2024/10/18
# @Author  : Bin Li
# @Email   : lybin1336258176@outlook.com
# @File    : main.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch

print(torch.__version__)

import onnxscript

print(onnxscript.__version__)

from onnxscript import opset18  # opset 18 is the latest (and only) supported version for now

import onnxruntime

print(onnxruntime.__version__)
