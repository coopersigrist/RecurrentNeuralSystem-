# -*- coding: utf-8 -*-
"""ReNS experiments - CIFAR10 [conv]

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WwT0q8ojrAJ4zHy82CK4ST2iZ-gmLsAu

# SETUP
"""

#@title Insatlling Pyorch

!pip install torch
!pip install torchvision

#@title Import Dependencies
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

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

"""# MODELS

"""

class Modulator(nn.Module):

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

        self.mod = Modulator(reflexor_size)
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

        self.mod = Modulator(reflexor_size)
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

      out = x.view(-1, reflexor_size * 16 ** 2)
      out = self.fc1(out)
      out = self.batchnorm(out)
      out = self.relu(out)
      out = self.fc2(out)
      out = self.sigmoid(out)

      return out

"""# TRAINING"""

# Hyperparams
batch_size = 64
num_epochs = 10
reflexor_size = 10
image_size = 32
channels = 3

transform = transforms.Compose(
    [transforms.ToTensor()])

# Load MNIST data.
train_data = dsets.CIFAR10(root = './data', train = True,
                        transform = transform, download = True)

test_data = dsets.CIFAR10(root = './data', train = False,
                       transform = transform)

train_gen = torch.utils.data.DataLoader(dataset = train_data,
                                             batch_size = batch_size,
                                             shuffle = True)

test_gen = torch.utils.data.DataLoader(dataset = test_data,
                                      batch_size = batch_size, 
                                      shuffle = False)

encoder1 = ConvolutionalEncoder(reflexor_size)
decoder1 = ConvolutionalDecoder(reflexor_size)
classifier1 = ConvolutionalEncoderClassifier(reflexor_size, 10)
auto_params1 = list(encoder1.parameters()) + list(decoder1.parameters())

encoder2 = ModulatedConvolutionalEncoder(reflexor_size)
decoder2 = ConvolutionalDecoder(reflexor_size)
classifier2 = ConvolutionalEncoderClassifier(reflexor_size, 10)
auto_params2 = list(encoder2.parameters()) + list(decoder2.parameters())

net1 = [encoder1, decoder1, classifier1, auto_params1]
net2 = [encoder2, decoder2, classifier2, auto_params2]

lr = 1e-5 # size of step 
loss_function = nn.MSELoss()

# Unnormalize the image to display it
def img_fix(img):
  return np.transpose((img).numpy(), (1, 2, 0))

# Commented out IPython magic to ensure Python compatibility.
auto_train_losses = [[],[],[]]
auto_test_losses = [[],[],[]]
class_train_losses = [[],[],[]]
class_test_losses = [[],[],[]]

real_imgs = [[],[],[]]
reconstructed_imgs = [[],[],[]]

param_counts = np.ones(3)

steps = [[],[],[]]

for num, net in enumerate([net1, net2]):
  encoder, decoder, classifier, params = net

  autoencoder_optimizer = torch.optim.Adam(params, lr=lr)
  classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
  param_counts[num] = (sum(p.numel() for p in params if p.requires_grad))

  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_gen):
      
      autoencoder_optimizer.zero_grad()
      classifier_optimizer.zero_grad()
      
      # Generate encoded features
      encoded = encoder(images)

      # Train autoencoder
      decoded = decoder(encoded)
      decoder_loss = loss_function(decoded, images)
      decoder_loss.backward()
      autoencoder_optimizer.step()

      # Train classifier
      outputs = classifier(encoded.detach())
      labels = torch.nn.functional.one_hot(labels, num_classes=10).type(torch.FloatTensor)
      output_loss = loss_function(outputs, labels)
      output_loss.backward()
      classifier_optimizer.step()
      
      if (i+1) % 300 == 0:
        auto_loss = decoder_loss.item()
        class_loss = output_loss.item()
        print('Epoch [%d/%d], Step [%d/%d], class_loss: %.4f, auto_loss: %.4f,'
#                   %(epoch+1, num_epochs, i+1, len(train_data)//batch_size, class_loss, auto_loss))
        dupe = Variable(decoded[0].data, requires_grad=False)
        # plt.imshow(img_fix(images[0]))
        # plt.show()
        # plt.imshow(img_fix(dupe))
        # plt.show()
        auto_train_losses[num].append(auto_loss)
        class_train_losses[num].append(class_loss)
        steps[num].append((50000 * epoch) + ((i + 1) * batch_size))

        real_imgs[num].append(img_fix(images[0].clone()))
        reconstructed_imgs[num].append(img_fix(dupe.clone()))

        # Test Data
        # Calculate train loss for image generation
        score = 0
        total = 0
        for images, labels in test_gen:
          output = decoder(encoder(images))
          score += loss_function(output, images).item()
        auto_test_losses[num].append((score))

        # Calculate train loss for image classification
        score = 0
        total = 0
        for images, labels in test_gen:
          output = classifier(encoder(images))
          labels = torch.nn.functional.one_hot(labels, num_classes=10).type(torch.FloatTensor)
          score += loss_function(output, labels).item()
        class_test_losses[num].append((score))

plt.plot(steps[0], auto_train_losses[0], label= "Baseline")
plt.plot(steps[1], auto_train_losses[1], label= "Modulated")
plt.plot(steps[2], auto_train_losses[2], label= "Recurrent with Modulation")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Autoencoder training loss history')
plt.legend()
plt.show()

plt.plot(steps[0], class_train_losses[0], label= "Baseline")
plt.plot(steps[1], class_train_losses[1], label= "Modulated")
plt.plot(steps[2], class_train_losses[2], label= "Recurrent with Modulation")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Classification training loss history')
plt.legend()
plt.show()

plt.plot(steps[0], auto_test_losses[0], label= "Baseline")
plt.plot(steps[1], auto_test_losses[1], label= "Modulated")
plt.plot(steps[2], auto_test_losses[2], label= "Recurrent with Modulation")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Autoencoder training loss history')
plt.legend()
plt.show()

plt.plot(steps[0], class_test_losses[0], label= "Baseline")
plt.plot(steps[1], class_test_losses[1], label= "Modulated")
plt.plot(steps[2], class_test_losses[2], label= "Recurrent with Modulation")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Classification test loss history')
plt.legend()
plt.show()

for num,count in enumerate(param_counts):
  param_counts[num] /= 1000

plt.bar(["Base", "Modulated", "ReNS"], param_counts)
plt.xlabel('Model')
plt.ylabel('# of thousands of Parameters')
plt.show()

from mpl_toolkits.axes_grid1 import ImageGrid

num_smaples = len(real_imgs[0])


for num in [0, 1]:
  fig = plt.figure(figsize=(20.,20.))
  grid = ImageGrid(fig, 111,  # similar to subplot(111)
                  nrows_ncols=(2, num_smaples),  # creates 2x2 grid of axes
                  axes_pad=0.1,  # pad between axes in inch.
                  )

  for ax, im in zip(grid, real_imgs[num]+reconstructed_imgs[num]):
      # Iterating over the grid returns the Axes.
      ax.imshow(im)
      ax.axis("off")

  plt.show()