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

# BASIC AUTOENCODER COMPONENTS
class Reflexor(nn.Module):
    def __init__(self, reflexor_size):
        super().__init__()
        self.conv = nn.Conv2d(32, reflexor_size, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(reflexor_size)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        return out

class ConvolutionalEncoder(nn.Module):

    def __init__(self, reflexor_size):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, reflexor_size, 3, padding=1)
        self.relu = torch.relu
        self.sigmoid = torch.sigmoid
        self.batchnorm = nn.BatchNorm2d(32)
        self.batchnormr = nn.BatchNorm2d(reflexor_size)
        self.reflexor = Reflexor(reflexor_size)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.batchnorm(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.reflexor(out)

        return out

class ConvolutionalDecoder(nn.Module):

    def __init__(self, reflexor_size):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.conv1 = nn.Conv2d(reflexor_size, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 3, 1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.relu = torch.relu
        self.sigmoid = torch.sigmoid
        self.batchnorm = nn.BatchNorm2d(32)

    def forward(self, x):

        out = self.upsample(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.batchnorm(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return out


# MODULATORS
# Basic modulator
class ConvModulator(nn.Module):

    def __init__(self, reflexor_size):

        super().__init__()

        self.soft = torch.nn.Softmax(dim=1)
        self.reflexor_size = reflexor_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(10,10), padding=5)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=reflexor_size, kernel_size=3, stride=2)

        self.relu = nn.ReLU()

    def forward(self, x):


        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)

        return out

# Basic modulator using pooling
class ConvModulatorWithPooling(nn.Module):

    def __init__(self, reflexor_size):

        super().__init__()

        self.soft = torch.nn.Softmax(dim=1)
        self.reflexor_size = reflexor_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(10,10), padding=5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=reflexor_size, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):


        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)

        return out

# Conv encoder as modulator
class ConvEncoderModulator(nn.Module):

    def __init__(self, reflexor_size):

        super().__init__()

        self.encoder = ConvolutionalEncoder(reflexor_size)
        self.sigmoid = torch.sigmoid

    def forward(self, x):

        out = self.encoder(x)
        out = self.sigmoid(out)

        return out

# ENCODER MODELS
class ModulatedConvolutionalEncoder(nn.Module):

    def __init__(self, reflexor_size, modulator=None):

        self.reflexor_size = reflexor_size

        super().__init__()

        if(modulator == None):
            self.mod = ConvModulator(reflexor_size)
        else:
            self.mod = modulator

        self.encoder = ConvolutionalEncoder(reflexor_size)

    def forward(self, x):

        mod = self.mod(x)
        out = self.encoder(x)
        out = out * mod

        return out

# FULL MODELS (not used right now)
class ConvolutionalAutoEncoder(nn.Module):

    def __init__(self, reflexor_size, encoder=None, decoder=None):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.encoder = ConvolutionalEncoder(reflexor_size)
        self.decoder = ConvolutionalDecoder(reflexor_size)

    def forward(self, x):

        out = self.encoder(x)
        out = self.decoder(out)

        return out

class ModulatedConvolutionalAutoEncoder(nn.Module):

    def __init__(self, reflexor_size, encoder=None, decoder=None):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.encoder = ModulatedConvolutionalEncoder(reflexor_size)
        self.decoder = ConvolutionalDecoder(reflexor_size)

    def forward(self, x):

        mod = self.mod(x)
        out = self.encoder(out)
        out = self.decoder(out)

        return out

# CLASSIFIER
class ConvolutionalEncoderClassifier(nn.Module):
    def __init__(self, reflexor_size, n_classes):

      self.reflexor_size = reflexor_size

      super().__init__()

      self.fc1 = nn.Linear(reflexor_size * 16 ** 2, 1000)
      self.fc2 = nn.Linear(1000, 500)
      self.fc3 = nn.Linear(500, n_classes)
      self.softmax = nn.Softmax(dim=1)
      self.batchnorm1 = nn.BatchNorm1d(1000)
      self.batchnorm2 = nn.BatchNorm1d(500)
      self.relu = torch.relu

    def forward(self, x):

      out = x.view(-1, self.reflexor_size * 16 ** 2)
      out = self.fc1(out)
      out = self.batchnorm1(out)
      out = self.relu(out)
      out = self.fc2(out)
      out = self.batchnorm2(out)
      out = self.relu(out)
      out = self.fc3(out)
      out = self.softmax(out)

      return out
