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
from models.Early_exit_models import Early_exit_lin_layer, Early_exit_classifier, Early_exit_confidence_layer



def main():

    batch_size = 1
    num_epochs = 5

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

    classifiers = {}
    linear_layers = {}
    confidences = {}

    # Sets the size of the main body layers which determines the size of the whole network
    layers = [3072,400,200,10]

    classes = layers[-1]
 
    # List of parameters of the whole model (for optimizer)
    # seperate ones for each because they all train at different times
    lin_params = []
    class_params = []
    confidence_params = []

    # Set the learning rate hyperparams -- we can change them all differently if we wanna make more work for otherselves
    lin_lr = 1e-7
    class_lr = 1e-7
    confidence_lr = 1e-7

    graph_iter = 10000

    # They'll all use the same loss (for now)
    loss_function = nn.MSELoss()

    gpu = torch.cuda.is_available()
    if gpu:
        print("CUDA available")
    

    # create the modules of each layer and store lists of their hyperparams
    for i, layer in enumerate(layers[:-1]):

        #add all of the layers -- the main body, the classifiers, and the decision/confidence layer
        linear_layers["lin_"+str(i)] = Early_exit_lin_layer(layers[i], layers[i+1])
        classifiers["class_"+str(i)] = Early_exit_classifier(layers[i+1], classes)
        confidences["confidence_"+str(i)] = Early_exit_confidence_layer(layers[i+1])

        if gpu:
            linear_layers["lin_"+str(i)] = linear_layers["lin_"+str(i)].to('cuda')
            classifiers["class_"+str(i)] = classifiers["class_"+str(i)].to('cuda')
            confidences["confidence_"+str(i)] = confidences["confidence_"+str(i)].to('cuda')


        # Make lists of params (you'll get the naming convention)
        lin_params += (list(linear_layers["lin_"+str(i)].parameters()))
        class_params += (list(classifiers["class_"+str(i)].parameters()))
        confidence_params += (list(confidences["confidence_"+str(i)].parameters()))





    # different optimizers for each module, but the same between layers for memories sake
    lin_and_class_optimizer = torch.optim.Adam( lin_params + class_params, lr=lin_lr)
    confidence_optimizer = torch.optim.Adam( confidence_params, lr=confidence_lr)

    print("Training:")

    for epoch in range(num_epochs):
        for i, (images, labels) in tqdm(enumerate(train_gen)):

            images = images.view((batch_size, 3072))
            labels = torch.nn.functional.one_hot(labels, num_classes=classes).type(torch.FloatTensor)
            labels = labels.view((batch_size, 10))

            if gpu:

                images = images.to('cuda')
                labels = labels.to('cuda')

            class_losses_instance = (train_linear_and_classifier(images, linear_layers, classifiers, labels, lin_and_class_optimizer, loss_function))
            decider_losses_instance = (train_decider(images, linear_layers, classifiers, confidences, labels, confidence_optimizer, loss_function, classes))

            class_losses_instance = np.expand_dims(class_losses_instance, axis=1)
            decider_losses_instance = np.expand_dims(decider_losses_instance, axis=1)

            if i == 0 and epoch == 0:
                class_losses = class_losses_instance
                decider_losses = decider_losses_instance

            elif i % graph_iter == 0:

                class_losses = np.concatenate((class_losses, class_losses_instance), axis=1)
                decider_losses = np.concatenate((decider_losses, decider_losses_instance), axis=1)


                for j in range(len(layers) - 1):
                    plt.plot(np.arange((i//graph_iter) + 2)*graph_iter/2, class_losses[j], label="class loss layer:"+str(j))
                    plt.plot(np.arange((i//graph_iter) + 2)*graph_iter/2, decider_losses[j], label="decider loss layer:"+str(j))

                plt.xlabel('Batches')
                plt.ylabel('Loss')
                plt.title('train loss history')
                plt.legend()
                plt.show()

    
    test_losses = np.ones(len(test_gen))

    print("Testing:")

    for i, (images, labels) in tqdm(enumerate(test_gen)):

        images = images.flatten()
        test_losses[i] = test(images, labels, linear_layers, classifiers, confidences, threshold)

    plt.plot(np.arange(len(test_gen)), test_losses)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('test loss history')
    plt.show()



def train_linear_and_classifier(inputs, linear_layers, classifiers, labels, optimizer, loss_function):

    # outputs = []
    optimizer.zero_grad()
    layers = len(linear_layers)

    class_losses = np.ones((layers))

    for i in range(layers):

        optimizer.zero_grad()


        classifier = classifiers["class_"+str(i)]
        lin_layer = linear_layers["lin_"+str(i)]

        # append the output of the linear layer (and just this layer) to the list of layer outpus  
        # outputs.append(layer(inputs)) # -- optional (see return)
        
        inputs = lin_layer(inputs)


        # use that output to get class accuracy for grad on classifier and linear
        outputs = classifier(inputs)
        class_loss = loss_function(outputs, labels)
        class_losses[i] = class_loss.item()
        class_loss.backward(retain_graph=True)
        optimizer.step()



    return class_losses # outputs # -- uncomment (and make other changes) if you'd like to train deider at same time. 

def train_decider(inputs, linear_layers, classifiers, confidences, labels, confidence_optimizer, loss_function, classes ):

    # outputs = []
    layers = len(linear_layers)
    batch_size = len(inputs)
    class_scores = np.ones((layers, batch_size, classes))

    outputs = [inputs]

    decider_losses = np.ones((layers))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(layers):

        # lin_optimizer.zero_grad()
        # class_optimizer.zero_grad()
        confidence_optimizer.zero_grad()

        classifier = classifiers["class_"+str(i)]
        lin_layer = linear_layers["lin_"+str(i)]


        # append the output of the linear layer (and just this layer) to the list of layer outpus  
        # outputs.append(layer(inputs)) # -- optional (see return)       
        outputs.append(lin_layer(outputs[-1].detach()))


        # use that output to get class accuracy for grad on classifier and linear
        class_score = classifier(outputs[-1])



        # class_loss = loss_function(class_score, labels)
        # class_loss.backward(retain_graph=True)
        # lin_optimizer.step()
        # class_optimizer.step()

        class_scores[i] = np.array(class_score.detach())


    # softmax over all layers 
    class_score_sm = np.exp(class_scores)
    class_score_sm /= np.sum(class_score_sm)
    
    
    for i in range(layers):
        
        decider = confidences["confidence_"+str(i)]

        decision_output = decider(outputs[i+1])

        class_score = torch.from_numpy(class_score_sm[i])
        class_sm_loss = loss_function(class_score, labels).item() 


        decision_label = torch.from_numpy((np.ones((decision_output.shape)) * class_sm_loss)).type(torch.float32)

        decider_loss = loss_function(decision_output, decision_label)
        decider_losses[i] = decider_loss.item()
        decider_loss.backward(retain_graph = True)
        confidence_optimizer.step()

    return decider_losses

def test(inputs, labels, linear_layers, classifiers, confidences, threshold=None):

    layers = len(linear_layers)
    for i in range(layers-1):

        classifier = classifiers["class_"+str(i)]
        lin_layer = linear_layers["lin_"+str(i)]
        decider = confidences["confidence_"+str(i)]

        main_outs = lin_layer(inputs)

        if threshold is not None:

            confidence = decider(main_outs)

            if (confidence > threshold):

                classifications = classifier(main_outs)
                return classifications
    
    classifications = classifier(main_outs)
    return classifications1

main()