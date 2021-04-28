import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Sequence, Iterable
import math
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair
from torchvision import models
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt


# class Early_Exit_net(nn.Module):

#     def __init__(self, layers, classes):

#         # layers is a list of sizes of the layers, the first term is the size of the input and last is the number of classes

#         self.classes = classes
#         super().__init__()

#         self.layers = []
#         self.classes = layers[-1]

#         for i in range(layers-1):
#             self.layers.append(Early_exit_lin_layer(layers[i],layers[i+1], classes))

#         def forward(x):
#             out = x
#             for layer in self.layers:
#                 out = layer(out)
                
#             return out

class Early_exit_conv_layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_kernel, pool_stride, pool_padding):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.mp = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def output_size(self,B,C,H,W):

        conv_H = ((H + 2*self.padding[0] - self.kernel_size[0])/self.stride[0]) + 1
        conv_W = ((W + 2*self.padding[1] - self.kernel_size[1])/self.stride[1]) + 1
        conv_C = self.out_channels

        mp_H = ((conv_H + 2*self.pool_padding[0] - self.pool_kernel[0])/self.pool_stride[0]) + 1
        mp_W = ((conv_W + 2*self.pool_padding[1] - self.pool_kernel[1])/self.pool_stride[1]) + 1

        return B, conv_C, mp_H, mp_W

    def forward(self, x):

        out = self.conv(x)
        out = self.mp(out)

        return out
    

class Early_exit_lin_layer(nn.Module):

    def __init__(self, in_size, out_size):

        self.in_size = in_size
        self.out_size = out_size
        super().__init__()

        self.lin_layer = nn.Linear(self.in_size, self.out_size, True)
        self.bn = nn.LayerNorm(self.out_size)

    def forward(self, x):

        out = self.lin_layer(x) 
        out = torch.relu(out)
        out = self.bn(out)

        return out



class Early_exit_classifier(nn.Module):

    def __init__(self, in_size, classes):

        self.in_size = in_size
        self.classes = classes
        super().__init__()

        self.lin_layer = nn.Linear(self.in_size, self.classes, True)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):

        out = self.lin_layer(x)
        out = self.sm(out)
        

        return out

class Early_exit_confidence_layer(nn.Module):

    def __init__(self, in_size):

        self.in_size = in_size
        super().__init__()

        self.lin_layer = nn.Linear(self.in_size, 1, True)

    def forward(self, x):

        out = self.lin_layer(x)
        out = torch.sigmoid(out)

        return out

