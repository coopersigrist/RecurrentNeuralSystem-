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



def main():

    batch_size = 1
    num_epochs = 10

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
    lin_lr = 1e-6
    class_lr = 1e-6
    confidence_lr = 1e-7

    graph_iter = 100
    threshold = 0.5
    graphing = False

    # They'll all use the same loss (for now)
    class_loss_function = nn.BCELoss()
    decider_loss_function = nn.MSELoss()

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

    test_losses = []

    print("Training Main and Classifiers:")

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(tqdm(train_gen)):

            images = images.view((batch_size, 3072))
            labels = torch.nn.functional.one_hot(labels, num_classes=classes).type(torch.FloatTensor)
            labels = labels.view((batch_size, 10))

            if gpu:

                images = images.to('cuda')
                labels = labels.to('cuda')

            train_all(images, linear_layers, classifiers, confidences, labels, lin_and_class_optimizer, class_loss_function, confidence_optimizer, decider_loss_function, threshold)
            # train_linear_and_classifier(images, linear_layers, classifiers, labels, lin_and_class_optimizer, class_loss_function)
            # train_decider(images, linear_layers, classifiers, confidences, labels, confidence_optimizer, loss_function, classes)


        save_all(linear_layers, classifiers, confidences, "rem_" + str(epoch+1))


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

    path = '../checkpoints/'+save_name

    if not os.path.exists(path):
        os.makedirs(path, mode=0o777)
        os.chmod(path, mode=0o777)

    for i in range(len(linear_layers)):
        torch.save({"lin_"+str(i) : linear_layers["lin_"+str(i)].state_dict(),
        "class_"+str(i) : classifiers["class_"+str(i)].state_dict(),
        "confidence_"+str(i) : confidences["confidence_"+str(i)].state_dict()}, path+'/layer'+str(i)+'.pt')

    return

def sig(x):
    return 1/(1 + np.exp(-x))


def train_all(inputs, linear_layers, classifiers, confidences, labels, class_optimizer, class_loss_function, decider_optimizer, decider_loss_function, threshold):

    # outputs = []
    class_optimizer.zero_grad()
    decider_optimizer.zero_grad()
    layers = len(linear_layers)

    for i in range(layers):

        class_optimizer.zero_grad()
        decider_optimizer.zero_grad()


        classifier = classifiers["class_"+str(i)]
        lin_layer = linear_layers["lin_"+str(i)]
        decider = confidences["confidence_"+str(i)]

        # append the output of the linear layer (and just this layer) to the list of layer outpus  
        # outputs.append(layer(inputs)) # -- optional (see return)
        
        detached_inputs = inputs.detach()
        inputs = lin_layer(inputs)

        conf = decider(inputs.detach())[0].item()

        if conf > threshold:

            outputs = classifier(inputs)
            class_loss = class_loss_function(outputs, labels)
            class_loss.backward(retain_graph=True)
            class_optimizer.step()

            decider_label = torch.FloatTensor([float(1 - sig(class_loss.item()))]) 
            decider_loss = decider_loss_function(conf, decider_label)
            decider_loss.backward()
            decider_optimizer.step()

            break


        # use that output to get class accuracy for grad on classifier and linear


    # print(class_losses)

    return  # outputs # -- uncomment (and make other changes) if you'd like to train deider at same time. 


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

def train_decider(inputs, linear_layers, classifiers, confidences, labels, confidence_optimizer, loss_function, classes ):

    # outputs = []
    layers = len(linear_layers)
    batch_size = len(inputs)
    class_losses = np.ones((layers, batch_size))

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

        class_losses[i] = loss_function(class_score, labels).item()


    # softmax over all layers 
    class_loss_softmin = nn.functional.softmin(torch.from_numpy(class_losses).detach().type(torch.float32), dim=0)

    
    
    for i in range(layers):
        
        decider = confidences["confidence_"+str(i)]

        decision_output = decider(outputs[i+1])

        decision_label = class_loss_softmin[i]

        decider_loss = loss_function(decision_output, decision_label)
        decider_losses[i] = decider_loss.item()
        decider_loss.backward(retain_graph = True)
        confidence_optimizer.step()

    return decider_losses

def test(linear_layers, classifiers, confidences, threshold=None):

    batch_size = 1
    n_layers = len(linear_layers)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_data = dsets.CIFAR10(root = './data', train = False,
                    transform = transform)
    
    test_gen = torch.utils.data.DataLoader(dataset = test_data,
                                    batch_size = batch_size,
                                    shuffle = False)

# Run test over all the test data #

    exit_counts = np.zeros(n_layers)

    correct = np.zeros(n_layers)


    for i, (image, label) in tqdm(enumerate(test_gen)):

        # Called main_outs because this will be the inputs/outputs of the main body (linear layers)
        main_outs = image.view((batch_size, 3072))

        max_conf = 0 # for threshold = None


        for i in range(n_layers):

            classifier = classifiers["class_"+str(i)]
            lin_layer = linear_layers["lin_"+str(i)]
            decider = confidences["confidence_"+str(i)]

            main_outs = lin_layer(main_outs)

            if threshold is not None:

                confidence = decider(main_outs).detach()[0]

                if (confidence >= threshold):

                    # Add 1 to count for early exit on EE'd layer
                    exit_counts[i] += 1

                    scores = list(classifier(main_outs).detach())
                    scores = list(scores[0].detach())
                    prediction = scores.index(max(scores))

                    if (prediction == label):
                        correct[i] += 1
                    # else:
                        # print("chose", prediction,"not",label)
                        # print("had confidence:", confidence)

                    break
                
                elif(i == n_layers-1):
                    # Adds Count for exit on final layer (none had enough confidence)
                    exit_counts[i] += 1

                    scores = list(classifier(main_outs).detach())
                    scores = list(scores[0].detach())
                    prediction = scores.index(max(scores))

                    if (prediction == label):
                        correct[i] += 1
 
            else:

                conf = decider(main_outs).detach()[0]
                if  conf > max_conf:

                    max_conf = conf
                    max_conf_ind = i
                    scores = list(classifier(main_outs).detach())
                    scores = list(scores[0].detach())
                    
                    max_conf_pred = scores.index(max(scores))
                
        if threshold is None:

            # Add 1 to count for early exit on EE'd layer
            exit_counts[max_conf_ind] += 1

            prediction = max_conf_pred

            if (prediction == label):
                correct[max_conf_ind] += 1
            # else:
                # print("chose", prediction,"not",label)
                # print("had confidence:", max_conf)

            

    total_accuracy = np.sum(correct)/len(test_gen)

    print("accuracy=", total_accuracy)

    accuracy = correct/exit_counts

    for i in range(n_layers):


        print("number of exits at layer",i,":",exit_counts[i])
        print("accuracy of exits at layer",i,":",accuracy[i])

    return exit_counts, accuracy, total_accuracy

def load_and_test_and_graph(layers, checkpoints, threshold=0.8):

    # Layers here is number of layers

    paths = []

    for cp in checkpoints:
        paths.append('../checkpoints/'+str(cp))

    linear_layers = {}
    classifiers = {}
    confidences = {}

    lin_params = []
    class_params = []
    confidence_params = []

    n_layers = len(layers) - 1
    classes = layers[-1]

    for i in range(n_layers):

        #load all of the layers -- the main body, the classifiers, and the decision/confidence layer

        linear_layers["lin_"+str(i)] = Early_exit_lin_layer(layers[i], layers[i+1])
        classifiers["class_"+str(i)] = Early_exit_classifier(layers[i+1], classes)
        confidences["confidence_"+str(i)] = Early_exit_confidence_layer(layers[i+1])

    n_paths = len(paths)

    exit_counts = np.zeros((n_paths, n_layers))
    layer_accuracies = np.zeros((n_paths, n_layers))
    total_accuracies = np.zeros((n_paths))


    for n, path in enumerate(paths):

        for i in range(n_layers):

            layer = torch.load(path + '/layer'+str(i)+'.pt')

            linear_layers["lin_"+str(i)].load_state_dict(layer["lin_"+str(i)])
            classifiers["class_"+str(i)].load_state_dict(layer["class_"+str(i)])
            confidences["confidence_"+str(i)].load_state_dict(layer["confidence_"+str(i)]) 

            # Make lists of params (you'll get the naming convention)
            lin_params += (list(linear_layers["lin_"+str(i)].parameters()))
            class_params += (list(classifiers["class_"+str(i)].parameters()))
            confidence_params += (list(confidences["confidence_"+str(i)].parameters()))
        
        exit_counts[n], layer_accuracies[n], total_accuracies[n] = test(linear_layers, classifiers, confidences, threshold=0.8)

    width = 1/ n_paths

    for n in range(n_paths):
        plt.bar(np.arange(n_layers) + n*width, exit_counts[n], width=width, label=checkpoints[n])

    plt.legend()
    plt.title("early exit counts by layer -- threshold =" +str(threshold))
    plt.xlabel('layer number')
    plt.ylabel('number of early exits')
    plt.show()

    for n in range(n_paths):
        plt.bar(np.arange(n_layers) + n*width, layer_accuracies[n], width=width, label=checkpoints[n])

    plt.legend()
    plt.title("Accuracy of layers output when chosen for EE -- threshold =" +str(threshold))
    plt.xlabel('layer number')
    plt.ylabel('avg accuracy of layer when chosen')
    plt.show()

    for n in range(n_paths):
        plt.bar(np.arange(1) + n*width, total_accuracies[n], width=width, label=checkpoints[n])

    plt.legend()
    plt.title("total test accuracy with early exit -- threshold =" +str(threshold))
    plt.xlabel('layer number')
    plt.ylabel('test accuracy')
    plt.show()

    quit()

# main()

layers = [3072,400,200,10]

checkpoints = ['bal_2', 'cd_2', 'a_2', 'det_2']


load_and_test_and_graph(layers=layers, checkpoints=checkpoints, threshold=0.01)
# main()

exit()