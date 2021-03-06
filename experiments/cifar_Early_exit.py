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
import os
from models.Early_exit_models import Early_exit_lin_layer, Early_exit_classifier, Early_exit_confidence_layer

from cifar_Early_exit_tests import run_test, load_and_test_and_graph 



def train_and_save(batch_size=50,num_epochs=10,balance=false,layers=None,classes=10,class_lr=1e-6,conf_lr=1e-7,save_name='temp_'):


    # Hyper parameters ########

    batch_size = batch_size
    num_epochs = num_epoches
    balance = balance   # Whether or not to balance decider training (equal 1s and 0s) 

    ###########################


    # Normalization transformation for data ###
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load data.
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

    # These will become the 
    classifiers = {}
    linear_layers = {}
    confidences = {}

    # Sets the size of the main body layers which determines the size of the whole network, passed as param
    layers = layers

    # passed as param
    classes = classes
 
    # List of parameters of the whole model (for optimizer)
    # seperate ones for each because they all train at different times
    lin_params = []
    class_params = []
    confidence_params = []

    # Set the learning rate hyperparams -- we can change them all differently if we wanna make more work for otherselves
    class_lr = class_lr
    confidence_lr = conf_lr

    # They'll all use the same loss (for now)
    loss_function = nn.BCELoss()

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


    # How many more have been accepted than rejected for Early exit by each layer -- used to balance decider training
    more_accepted = np.zeros((len(layers) - 1))


    # different optimizers for each module, but the same between layers for memories sake
    lin_and_class_optimizer = torch.optim.Adam( lin_params + class_params, lr=class_lr)
    confidence_optimizer = torch.optim.Adam( confidence_params, lr=confidence_lr)


    print("Training Main and Classifiers:")

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(tqdm(train_gen)):

            # Just reshaping the images and labels as well as changing the labels to be one-hot encoding
            images = images.view((batch_size, 3072))
            labels = torch.nn.functional.one_hot(labels, num_classes=classes).type(torch.FloatTensor)
            labels = labels.view((batch_size, 10))

            if gpu:
                images = images.to('cuda')
                labels = labels.to('cuda')

            # These two function calls train the classifier+Main body then Decider layer respectively
            train_linear_and_classifier(images, linear_layers, classifiers, labels, lin_and_class_optimizer, loss_function)
            more_accepted = train_decider(images, linear_layers, classifiers, confidences, labels, confidence_optimizer, loss_function, classes, more_accepted)

        # Saves the network that is currently trained in checkpoints, with the name given as a param
        save_all(linear_layers, classifiers, confidences, save_name + str(epoch+1))


    # print("Training Deciders:")  ### UNCOMMENT FROM HERE DOWN TO SEPERATE CLASS / DECIDER TRAINING

    # # change to MSE for decider
    # loss_function = nn.MSELoss()

    # for epoch in range(num_epochs):
    #     for i, (images, labels) in enumerate(tqdm(train_gen)):

    #         images = images.view((batch_size, 3072))
    #         labels = torch.nn.functional.one_hot(labels, num_classes=classes).type(torch.FloatTensor)
    #         labels = labels.view((batch_size, 10))

    #         if gpu:

    #             images = images.to('cuda')
    #             labels = labels.to('cuda')

    #         # train_linear_and_classifier(images, linear_layers, classifiers, labels, lin_and_class_optimizer, loss_function)
    #         train_decider(images, linear_layers, classifiers, confidences, labels, confidence_optimizer, loss_function, classes)


            # if (i % graph_iter == 0):

    #             save_all(linear_layers, classifiers, confidences, "d_"+str(epoch + 1))
    #             # load_and_test_and_graph(layers=layers, path='../checkpoints/'+str(i), threshold=0.5)

    #             # print("Testing:")
    #             # test(linear_layers, classifiers, confidences, threshold=0.8)


def save_all(linear_layers, classifiers, confidences, save_name):

    '''
    Saves the current state of the network (linear, classifier, confidences layers) in checkpoints folder

    Params:

    - Linear_layers: a dictionary of all of the linear layer modules
    - Classifiers: a dictionary of all of the classifier layer modules
    - confidences: a dict of all of the decider layer modules
    - save_name: an arbitrary name for what to save the current state as: convention is *something*_numberofepoches (i.e. rem_5 is the removed data network at 5th epoch)
    '''

    path = '../checkpoints/'+save_name

    if not os.path.exists(path):
        os.makedirs(path, mode=0o777)
        os.chmod(path, mode=0o777)

    for i in range(len(linear_layers)):
        torch.save({"lin_"+str(i) : linear_layers["lin_"+str(i)].state_dict(),
        "class_"+str(i) : classifiers["class_"+str(i)].state_dict(),
        "confidence_"+str(i) : confidences["confidence_"+str(i)].state_dict()}, path+'/layer'+str(i)+'.pt')

    return


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
        
        detached_inputs = inputs.detach()
        inputs = lin_layer(inputs)


        # use that output to get class accuracy for grad on classifier and linear
        outputs = classifier(inputs)
        class_loss = loss_function(outputs, labels)
        class_losses[i] = class_loss.item()
        class_loss.backward(retain_graph=True)
        optimizer.step()

    # print(class_losses)

    return class_losses # outputs # -- uncomment (and make other changes) if you'd like to train deider at same time. 

def train_decider(inputs, linear_layers, classifiers, confidences, labels, confidence_optimizer, loss_function, classes, more_accepted):

    # outputs = []
    layers = len(linear_layers)
    batch_size = len(inputs)
    class_losses = np.ones((layers, batch_size))

    outputs = [inputs]

    decider_losses = np.ones((layers))

    for i in range(layers):

        # Zero the optimizer each time to not double up on gradients
        confidence_optimizer.zero_grad()

        # defines which classifier and main body piece (lin_layer) are used on this layer
        classifier = classifiers["class_"+str(i)]
        lin_layer = linear_layers["lin_"+str(i)]


        # append the output of the linear layer (and just this layer) to the list of layer outpus         
        outputs.append(lin_layer(outputs[-1].detach()))


        # get output of the classifier -- needed for decider training
        class_score = classifier(outputs[-1])

        # convert scores to losses to train the decider
        class_losses[i] = loss_function(class_score, labels).item()


    # softmax over all layers 
    class_loss_softmin = nn.functional.softmin(torch.from_numpy(class_losses).detach().type(torch.float32), dim=0)

    # Find which layer has max confidence
    max_conf_index = torch.argmax(class_loss_softmin)

    # More accepted gets +1 for the layer that had highest confidence
    more_accepted[max_conf_index] += 1
    

    # Loop over all layers to train respective deciders
    for i in range(layers):

        # When 'balance' is true we only allow the decider to train on an equal number of 1s and 0s to balance training
        if balance is false or more_accepted[i] > 0 :

            # 'Confidences' is our dictionary of decider layers -- this gets current layer's decider
            decider = confidences["confidence_"+str(i)]


            decision_output = decider(outputs[i+1])

            decision_label = class_loss_softmin[i]

            decider_loss = loss_function(decision_output, decision_label)
            decider_losses[i] = decider_loss.item()
            decider_loss.backward(retain_graph = True)
            confidence_optimizer.step()

            if i is not max_conf_index:
                more_accepted[i] -= 1

    return more_accepted




train_and_save(batch_size=50,num_epochs=10,balance=false,layers=[3072,400,200,10],classes=10,class_lr=1e-6,conf_lr=1e-7,save_name='temp_')

layers = [3072,400,200,10]

cds = ['cd_1','cd_2','cd_3','cd_4','cd_5','cd_6','cd_7']
det = ['det_1','det_2','det_3','det_4','det_5','det_6','det_7','det_8','det_9','det_10']
rem = ['rem_1','rem_2','rem_3','rem_4','rem_5','rem_6','rem_7','rem_8','rem_9','rem_10']
alt = ['a_1','a_2','a_3','a_4','a_5','a_6','a_7','a_8','a_9','a_10']
comparison = ['cd_7','det_10','rem_10','a_10']

large_layers = [3072,2000,1000,400,200,10]

large = ['t_cd_1','t_cd_2','t_cd_3','t_cd_4','t_cd_5','t_cd_6','t_cd_7','t_cd_8','t_cd_9','t_cd_10']


for threshold in [0.3, 0.5, 0.6, 0.7, None]:

    for name, checkpoints in zip(['simultaneous','detached','removed','alternating(1 per 100)','comparison'],[cds,det,rem,alt,comparison]):

        load_and_test_and_graph(layers=layers, checkpoints=checkpoints, name=name, threshold=threshold, classes=10)
    
    load_and_test_and_graph(layers=large_layers, checkpoints=large, name='large', threshold=threshold, classes=10)


exit()