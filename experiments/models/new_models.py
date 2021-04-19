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


class Early_exit_lin_layer(nn.Module):

    def __init__(self, in_size, out_size, classes):

        self.in_size = in_size
        self.out_size = out_size
        self.classes = classes
        self.classifer = Early_exit_classifier(in_size, classes)
        self.confidence_layer = Early_exit_confidence_layer(in_size)
        super().__init__()

        self.lin_layer = nn.Linear(self.in_size, self.out_size, True)

    def forward(x):

        out = self.lin_layer(x) 
        out = torch.relu(out)
        self.classify = self.classifer(out)
        self.confidence = self.confidence_layer(out)

        return out




class Early_exit_classifier(nn.Module):

    def __init__(self, in_size, classes):

        self.in_size = in_size
        self.classes = classes
        super().__init__()

        self.lin_layer = nn.Linear(self.in_size, self.classes, True)

    def forward(x):

        out = self.lin_layer(x)

        return out

class Early_exit_confidence_layer(nn.Module):

    def __init__(self, in_size):

        self.in_size = in_size
        super().__init__()

        self.lin_layer = nn.Linear(self.in_size, 1, True)

    def forward(x):

        out = self.lin_layer(x)

        return out
