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
from models.base_cnn_models import *
import matplotlib.pyplot as plt

# BEGIN flat reflexor experiment
class FlattenedConvEncoderModulator(nn.Module):

    def __init__(self, reflexor_size):

        super().__init__()

        self.encoder = ConvolutionalEncoder(reflexor_size)
        self.sigmoid = torch.sigmoid

    def forward(self, x):

        out = self.encoder(x)
        out = out.view(-1, 10 * 16 ** 2)
        out = self.sigmoid(out)

        return out

class FlattenedConvolutionalEncoder(nn.Module):

    def __init__(self, reflexor_size):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.encoder = ConvolutionalEncoder(10)
        self.fc = nn.Linear(10 * 16 ** 2, reflexor_size)

    def forward(self, x):

        out = self.encoder(x)
        out = out.view(-1, 10 * 16 ** 2)
        out = self.fc(out)

        return out

class ModulatedFlattenedConvolutionalEncoder(nn.Module):

    def __init__(self, reflexor_size):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.encoder = ConvolutionalEncoder(10)
        self.mod = FlattenedConvEncoderModulator(10)
        self.fc = nn.Linear(10 * 16 ** 2, reflexor_size)

    def forward(self, x):
        mod = self.mod(x)
        out = self.encoder(x)
        out = out.view(-1, 10 * 16 ** 2)
        out = out * mod
        out = self.fc(out)

        return out

class FlattenedConvolutionalDecoder(nn.Module):

    def __init__(self, reflexor_size):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.fc = nn.Linear(reflexor_size, 10 * 16 ** 2)
        self.decoder = ConvolutionalDecoder(10)

    def forward(self, x):

        out = self.fc(x)
        out = out.view(-1, 10, 16, 16)
        out = self.decoder(out)

        return out

class FlattenedEncoderClassifier(nn.Module):
    def __init__(self, reflexor_size, n_classes):

      self.reflexor_size = reflexor_size

      super().__init__()

      self.fc1 = nn.Linear(reflexor_size, 500)
      self.fc2 = nn.Linear(500, n_classes)
      self.softmax = nn.Softmax(dim=1)
      self.batchnorm = nn.BatchNorm1d(500)
      self.relu = torch.relu

    def forward(self, x):
      out = self.fc1(x)
      out = self.batchnorm(out)
      out = self.relu(out)
      out = self.fc2(out)
      out = self.softmax(out)

      return out
# END flat reflexor experiment

# BEGIN pre reflexor layer experiment
class PreReflexorConvEncoderModulator(nn.Module):

    def __init__(self, reflexor_size):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.relu = torch.relu
        self.sigmoid = torch.sigmoid
        self.batchnorm = nn.BatchNorm2d(32)
        self.batchnormr = nn.BatchNorm2d(reflexor_size)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.batchnorm(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return out

class PreReflexorModulatedConvEncoder(nn.Module):

    def __init__(self, reflexor_size):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.mod = PreReflexorConvEncoderModulator(reflexor_size)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, reflexor_size, 3, padding=1)
        self.relu = torch.relu
        self.sigmoid = torch.sigmoid
        self.batchnorm = nn.BatchNorm2d(32)
        self.batchnormr = nn.BatchNorm2d(reflexor_size)
        self.reflexor = Reflexor(reflexor_size)

    def forward(self, x):

        mod = self.mod(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.batchnorm(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out * mod
        out = self.reflexor(out)

        return out
# END pre reflexor layer experiment

# BEGIN modulator only prereflexor experiment
class PreReflexorConvEncoder(nn.Module):

    def __init__(self, reflexor_size):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.mod = PreReflexorConvEncoderModulator(reflexor_size)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, reflexor_size, 3, padding=1)
        self.relu = torch.relu
        self.sigmoid = torch.sigmoid
        self.batchnorm = nn.BatchNorm2d(32)
        self.batchnormr = nn.BatchNorm2d(reflexor_size)
        self.reflexor = Reflexor(reflexor_size)

    def forward(self, x):

        mod = self.mod(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.batchnorm(out)
        out = self.conv2(out)
        out = self.relu(out)
        # out = out * mod
        # out = self.reflexor(out)

        return out
# END modulator only prereflexor experiment

