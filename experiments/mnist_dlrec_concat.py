from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models.models import RegularAutoEncoder, ModulatedAutoEncoder, RecDepthLimited, RecDepthLimitedConcat, Modulator
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

batch_size = 32
num_epochs = 2

# Load MNIST data.
train_data = dsets.MNIST(root='./data', train=True,
                         transform=transforms.ToTensor(), download=True)

test_data = dsets.MNIST(root='./data', train=False,
                        transform=transforms.ToTensor())

train_gen = torch.utils.data.DataLoader(dataset=train_data,
                                        batch_size=batch_size,
                                        shuffle=True)

test_gen = torch.utils.data.DataLoader(dataset=test_data,
                                       batch_size=batch_size,
                                       shuffle=False)

reflexor_size = 100

if torch.cuda.is_available():
    net1 = RegularAutoEncoder(784, 784, reflexor_size).cuda()
    net2 = ModulatedAutoEncoder(784, 784, reflexor_size).cuda()
    mod = Modulator(784, (4, 4), reflexor_size).cuda()
    net4 = RecDepthLimited(784, 784, reflexor_size, mod, 0, 3).cuda()
    mod2 = Modulator(784, (4, 4), reflexor_size).cuda()
    net5 = RecDepthLimitedConcat(784, 784, reflexor_size, mod, 0, 3).cuda()
else:
    net1 = RegularAutoEncoder(784, 784, reflexor_size)
    net2 = ModulatedAutoEncoder(784, 784, reflexor_size)
    mod = Modulator(784, (4, 4), reflexor_size)
    net4 = RecDepthLimited(784, 784, reflexor_size, mod, 0, 3)
    mod2 = Modulator(784, (4, 4), reflexor_size)
    net5 = RecDepthLimitedConcat(784, 784, reflexor_size, mod, 0, 3)

nets = [net1, net2, net4, net5]
names = ["Regular AutoEncoder", "Modulated AutoEncoder", "RecDepthLimited 3", "RecDepthLimited 3 Concat"]
num_nets = len(nets)

lr = .0001  # size of step
loss_function = nn.MSELoss()

train_losses = []
test_losses = []
real_imgs = []
reconstructed_imgs = []
steps = []
for n in range(num_nets):
    train_losses.append([])
    test_losses.append([])
    real_imgs.append([])
    reconstructed_imgs.append([])
    steps.append([])

param_counts = np.ones(num_nets)

for num, net in enumerate(nets):
    print("Net ", num, ",", names[num], "\n")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    param_counts[num] = (sum(p.numel()
                             for p in net.parameters() if p.requires_grad))

    start = time.time()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_gen):
            #images = Variable(images.view(-1,28*28))
            labels = Variable(images.view(-1, 28*28))

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 300 == 0:
                temp_loss = loss.item()
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch+1, num_epochs, i+1, len(train_data)//batch_size, temp_loss))
                dupe = Variable(outputs[0].data, requires_grad=False)
                if torch.cuda.is_available():
                    dupe = dupe.cpu()
                # plt.imshow(images[0].view(28, 28))
                # plt.show()
                # plt.imshow(dupe.view(28, 28))
                # plt.show()
                train_losses[num].append(temp_loss)
                steps[num].append((50000 * epoch) + ((i + 1) * batch_size))

                if torch.cuda.is_available():
                    images = images.cpu()
                real_imgs[num].append(images[0].view(28, 28))
                reconstructed_imgs[num].append(dupe.view(28, 28))

                # Test Data

                score = 0
                total = 0
                for images, labels in test_gen:
                    #images = Variable(images.view(-1,784))

                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()

                    output = net(images)
                    score += loss_function(output, images.view(-1, 784)).item()
                test_losses[num].append((score))
print("Training time:", round(time.time() - start), "seconds")

for i in range(num_nets):
    plt.plot(steps[i], train_losses[i], label=names[i])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.legend()
plt.show()

for i in range(num_nets):
    plt.plot(steps[i], test_losses[i], label=names[i])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Testing loss history')
plt.legend()
plt.show()

for num, count in enumerate(param_counts):
    param_counts[num] /= 1000

plt.bar(names, param_counts)
plt.xlabel('Model')
plt.ylabel('# of thousands of Parameters')
plt.show()

num_smaples = len(real_imgs[0])

for num in range(num_nets):
    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, num_smaples),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, real_imgs[num] + reconstructed_imgs[num]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis("off")

    plt.show()
