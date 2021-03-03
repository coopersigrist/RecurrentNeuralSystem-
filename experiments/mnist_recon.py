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
from models.models import RegularAutoEncoder, ModulatedAutoEncoder, PseudoRecAutoEncoder

batch_size = 32
num_epochs = 2

# Load MNIST data.
train_data = dsets.MNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

test_data = dsets.MNIST(root = './data', train = False,
                       transform = transforms.ToTensor())

train_gen = torch.utils.data.DataLoader(dataset = train_data,
                                             batch_size = batch_size,
                                             shuffle = True)

test_gen = torch.utils.data.DataLoader(dataset = test_data,
                                      batch_size = batch_size,
                                      shuffle = False)

# net = recurrentLayer(784, 784, 10, 5, 10, 0)
net1 = RegularAutoEncoder(784, 784, reflexor_size)
net2 = ModulatedAutoEncoder(784, 784, reflexor_size)
net3 = PseudoRecAutoEncoder(784, 784, reflexor_size)



lr = .0001 # size of step
loss_function = nn.MSELoss()

train_losses = [[],[],[]]
test_losses = [[],[],[]]

real_imgs = [[],[],[]]
reconstructed_imgs = [[],[],[]]

param_counts = np.ones(3)

steps = [[],[],[]]

for num, net in enumerate([net1, net2, net3]):

  optimizer = torch.optim.Adam( net.parameters(), lr=lr)
  param_counts[num] = (sum(p.numel() for p in net.parameters() if p.requires_grad))

  for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_gen):
      #images = Variable(images.view(-1,28*28))
      labels = Variable(images.view(-1,28*28))

      optimizer.zero_grad()
      outputs = net(images)
      loss = loss_function(outputs, labels)
      loss.backward()
      optimizer.step()

      if (i+1) % 300 == 0:
        temp_loss = loss.item()
        print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, len(train_data)//batch_size, temp_loss))
        dupe = Variable(outputs[0].data, requires_grad=False)
        # plt.imshow(images[0].view(28, 28))
        # plt.show()
        # plt.imshow(dupe.view(28, 28))
        # plt.show()
        train_losses[num].append(temp_loss)
        steps[num].append((50000 * epoch) + ((i + 1) * batch_size))

        real_imgs[num].append(images[0].view(28, 28))
        reconstructed_imgs[num].append(dupe.view(28, 28))

        # Test Data

        score = 0
        total = 0
        for images,labels in test_gen:
          #images = Variable(images.view(-1,784))

          output = net(images)
          score += loss_function(output, images.view(-1, 784)).item()
        test_losses[num].append((score))

then plt.plot(steps[0], train_losses[0], label= "Baseline")
plt.plot(steps[1], train_losses[1], label= "Modulated")
plt.plot(steps[2], train_losses[2], label= "Recurrent with Modulation")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.legend()
plt.show()

plt.plot(steps[0], test_losses[0], label= "Baseline")
plt.plot(steps[1], test_losses[1], label= "Modulated")
plt.plot(steps[2], test_losses[2], label= "Recurrent with Modulation")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Testing loss history')
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


for num in [0,1,2]:
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
