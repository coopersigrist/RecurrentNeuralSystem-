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

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.batchnorm(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.batchnormr(out)

        return out

class ModulatedConvolutionalEncoder(nn.Module):

    def __init__(self, reflexor_size):

        self.reflexor_size = reflexor_size

        super().__init__()

        self.mod = ConvModulator(reflexor_size)
        self.encoder = ConvolutionalEncoder(reflexor_size)

    def forward(self, x):

        mod = self.mod(x)
        out = self.encoder(x)
        out = out * mod

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

class ConvolutionalEncoderClassifier(nn.Module):
    def __init__(self, reflexor_size, n_classes):

      self.reflexor_size = reflexor_size

      super().__init__()

      self.fc1 = nn.Linear(reflexor_size * 16 ** 2, 500)
      self.fc2 = nn.Linear(500, n_classes)
      self.sigmoid = torch.sigmoid
      self.batchnorm = nn.BatchNorm1d(500)
      self.relu = torch.relu

    def forward(self, x):

      out = x.view(-1, self.reflexor_size * 16 ** 2)
      out = self.fc1(out)
      out = self.batchnorm(out)
      out = self.relu(out)
      out = self.fc2(out)
      out = self.sigmoid(out)

      return out

class recurrentLayer(nn.Module):

    def __init__(self, in_size, out_size, next_size, previous_size, reflexor_size, cdu_thresh, depth=1):

        print("new layer")

        self.in_size = in_size
        self.out_size = out_size
        self.next_size = next_size
        self.previous_size = previous_size
        self.reflexor_size = reflexor_size
        self.cdu_thresh = cdu_thresh
        self.depth = depth



        super().__init__()

        self.soft = torch.nn.Softmax(dim=1)

        if(self.in_size <= self.reflexor_size or depth > 3):
            self.trivial = True
            self.simple_layer = nn.Linear(self.in_size, self.out_size)
        else:
            self.trivial = False
            self.next = recurrentLayer(in_size=self.next_size, out_size=reflexor_size, next_size=math.floor(self.next_size/2), previous_size=math.floor(self.out_size/2), reflexor_size=math.ceil(self.reflexor_size * 2), cdu_thresh=self.cdu_thresh, depth=self.depth+1)
            self.previous = recurrentLayer(in_size=self.reflexor_size, next_size=math.floor(self.reflexor_size * 2), previous_size=math.floor(self.previous_size/2), out_size=self.previous_size, reflexor_size=math.floor(self.reflexor_size * 2), cdu_thresh=self.cdu_thresh, depth=self.depth+1)
            self.in_layer = nn.Linear(self.in_size, self.next_size, True)
            self.out_layer = nn.Linear(self.previous_size, self.out_size, True)


    def forward(self, x):
        if(self.trivial):
            return self.simple_layer(x)

        out = self.in_layer(x)
        out = torch.sigmoid(out)
        out = self.next.forward(out)
        out = torch.sigmoid(out)
        out = self.previous.forward(out)
        out = torch.sigmoid(out)
        out = self.out_layer(out)
        out = self.soft(out)

        return out

class RegularAutoEncoder(nn.Module):

    def __init__(self, in_size, out_size, reflexor_size):

        self.in_size = in_size
        self.out_size = out_size
        self.reflexor_size = reflexor_size



        super().__init__()

        self.soft = torch.nn.Softmax(dim=1)

        self.fc1 = nn.Linear(in_size, 300)
        self.fc2 = nn.Linear(300, reflexor_size)
        self.fc3 = nn.Linear(reflexor_size, 300)
        self.fc4 = nn.Linear(300, out_size)


    def forward(self, x):


        out = self.fc1(x.view(-1, self.in_size))
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        out = torch.relu(out)
        out = self.fc4(out)

        return out

class Modulator(nn.Module):

    def __init__(self, in_size, conv_size, out_size):

        self.in_size = in_size
        self.out_size = out_size
        self.conv_size = conv_size

        super().__init__()

        self.soft = torch.nn.Softmax(dim=1)

        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(10,10), padding=(1,1))
        self.fc1 = nn.Linear(21*21, out_size)

        self.relu = nn.ReLU()



    def forward(self, x):


        out = self.conv(x)
        self.relu(out)
        out = out.view(-1, 21 * 21)
        out = self.fc1(out)
        out = torch.sigmoid(out)

        return out

class LinModulator(nn.Module):

    def __init__(self, in_size, out_size):

        self.in_size = in_size
        self.out_size = out_size

        super().__init__()

        self.soft = torch.nn.Softmax(dim=1)

        self.fc1 = nn.Linear(in_size, out_size)


    def forward(self, x):

        out = self.fc1(x.view(-1, self.in_size))
        out = torch.sigmoid(out)


        return out

class ModulatedAutoEncoder(nn.Module):

    def __init__(self, in_size, out_size, reflexor_size):

        self.in_size = in_size
        self.out_size = out_size
        self.reflexor_size = reflexor_size



        super().__init__()

        self.soft = torch.nn.Softmax(dim=1)


        self.mod = Modulator(in_size, (4,4), reflexor_size)
        self.fc1 = nn.Linear(in_size, 200)
        self.fc2 = nn.Linear(200, reflexor_size)
        self.fc3 = nn.Linear(reflexor_size, 200)
        self.fc4 = nn.Linear(200, out_size)


    def forward(self, x):

        mod = self.mod(x)
        out = self.fc1(x.view(-1, self.in_size))
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = out * mod
        out = self.fc3(out)
        out = torch.relu(out)
        out = self.fc4(out)

        return out

class PseudoRecLayer(nn.Module):

    def __init__(self, in_size, out_size, reflexor_size, modulator):

        self.in_size = in_size
        self.out_size = out_size
        self.reflexor_size = reflexor_size





        super().__init__()

        self.fc1 = nn.Linear(in_size, reflexor_size)
        self.fc2 = nn.Linear(reflexor_size, out_size)
        self.mod = modulator
        self.relu = nn.ReLU()



    def forward(self, x):

        mod = self.mod(x)
        out = self.fc1(x.view(-1, self.in_size))
        out = self.relu(out)
        out = out * mod
        out = self.fc2(out)

        return out

class PseudoRecAutoEncoder(nn.Module):

    def __init__(self, in_size, out_size, reflexor_size):

        self.in_size = in_size
        self.out_size = out_size
        self.reflexor_size = reflexor_size




        super().__init__()


        self.mod1 = LinModulator(in_size, reflexor_size)
        self.mod2 = LinModulator(in_size//2, reflexor_size//2)
        self.mod3 = LinModulator(in_size//2, reflexor_size//2)

        print("made modulators")

        self.rec1 = PseudoRecLayer(in_size//2, reflexor_size, reflexor_size//2, self.mod2)
        self.rec2 = PseudoRecLayer(in_size//2, reflexor_size, reflexor_size//2, self.mod3)

        print("made recurrers")

        self.fc1 = nn.Linear(reflexor_size, 200)
        self.fc2 = nn.Linear(200, out_size)

        self.relu = nn.ReLU()


    def forward(self, x):

        mod1 = self.mod1(x)
        out = x.view(-1, self.in_size)

        # FIRST RECURENCE LAYER
        first_half = out[:, :self.in_size//2]
        rec1 = self.rec1(first_half)

        # SECOND RECURENCE LAYER
        second_half = out[:, self.in_size//2:]
        rec2 = self.rec2(second_half)

        out = rec1 * rec2
        out = self.relu(out)
        out = out * mod1
        out = self.relu(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)


        return out
