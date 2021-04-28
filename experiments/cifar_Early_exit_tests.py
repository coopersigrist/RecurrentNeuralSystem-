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


def run_test(linear_layers, classifiers, confidences, threshold=None):

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

    sum_confidences = np.zeros(n_layers)


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

                confidence = decider(main_outs).detach()[0].item()

                sum_confidences[i] += confidence

                # print("layer:",i,"confidence:",confidence,"threshold:",threshold,"accepted:",confidence >= threshold)

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

    avg_confidences = np.zeros(n_layers)

    for i in range(n_layers):
        avg_confidences[i] = sum_confidences[i] / np.sum(exit_counts[i:])            

    total_accuracy = np.sum(correct)/len(test_gen)

    print("accuracy=", total_accuracy)

    accuracy = correct/exit_counts

    for i in range(n_layers):


        print("number of exits at layer",i,":",exit_counts[i])
        print("accuracy of exits at layer",i,":",accuracy[i])

    return exit_counts, accuracy, total_accuracy, avg_confidences

def load_and_test_and_graph(layers, checkpoints, name='', threshold=0.8, classes=10):

    '''
    This function will load all of the pretrained networks give in the checkpoint param and will run a pass through the test data with it
    and will create a few figures on their test performance on a few metric. It will save these figures in the 'early_exit_results' folder

    Params: 

    - Layers: the layer sizes of our network in list form (e.g. [3072,400,10] would have (3072,400) as the size of linear layer 1) 

    - Checkpoints: a list of markers for checkpoints to use to get networks from the checkpoint folder (e.g. ['a_1','a_2'] would be the alternating net at 1 and 2 epochs )

    - Name: the name you'd like to save the figure under -- this will be the folder name 

    - Threshold: The threshold for acceptance at any layer compared to decider output

    '''

    paths = []

    # Creates a list of paths to the checkpoints 
    for cp in checkpoints:
        paths.append('../checkpoints/'+str(cp))

    # This is the path we will save results too, nested folder of threshold and the given name
    save_path = '../early_exit_results/threshold='+str(threshold)+'/'+str(name)+'/'

    # This will make the save path if it isn't made yet
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
        os.chmod(save_path, mode=0o777)

    # Creating an empty dictionary for each type of layer
    linear_layers = {}
    classifiers = {}
    confidences = {}


    n_layers = len(layers) - 1
    classes = classes

    for i in range(n_layers):

        #load all of the layers -- the main body, the classifiers, and the decision/confidence layer

        linear_layers["lin_"+str(i)] = Early_exit_lin_layer(layers[i], layers[i+1])
        classifiers["class_"+str(i)] = Early_exit_classifier(layers[i+1], classes)
        confidences["confidence_"+str(i)] = Early_exit_confidence_layer(layers[i+1])

    n_paths = len(paths)

    exit_counts = np.zeros((n_paths, n_layers))
    layer_accuracies = np.zeros((n_paths, n_layers))
    total_accuracies = np.zeros((n_paths))
    avg_confidences = np.zeros((n_paths, n_layers))


    for n, path in enumerate(paths):

        for i in range(n_layers):

            layer = torch.load(path + '/layer'+str(i)+'.pt')

            linear_layers["lin_"+str(i)].load_state_dict(layer["lin_"+str(i)])
            classifiers["class_"+str(i)].load_state_dict(layer["class_"+str(i)])
            confidences["confidence_"+str(i)].load_state_dict(layer["confidence_"+str(i)]) 
        
        exit_counts[n], layer_accuracies[n], total_accuracies[n], avg_confidences[n] = run_test(linear_layers, classifiers, confidences, threshold=threshold)

    width = 1/ n_paths

    ## PLOTTING VARIOUS THINGS AND SAVING THEM IN 'early_exit_results' ################


    # Plotting the number of early exits by layer #####################################

    fig = plt.figure()

    for n in range(n_paths):
        plt.bar(np.arange(n_layers) + n*width, exit_counts[n], width=width, label=checkpoints[n])

    plt.legend()
    plt.title("early exit counts by layer -- threshold =" +str(threshold))
    plt.xlabel('layer number')
    plt.ylabel('number of early exits')

    fig.savefig(save_path+'early_exits.png')
    plt.close()

    # Plotting the avg accuracy of exited data by layer ###############################

    fig = plt.figure()

    for n in range(n_paths):
        plt.bar(np.arange(n_layers) + n*width, layer_accuracies[n], width=width, label=checkpoints[n])

    plt.legend()
    plt.title("Accuracy of layers output when chosen for EE -- threshold =" +str(threshold))
    plt.xlabel('layer number')
    plt.ylabel('avg accuracy of layer when chosen')

    fig.savefig(save_path+'early_exits_accuracies.png')
    plt.close()

    # Plotting the avg confidence by layer #############################################

    fig = plt.figure()

    for n in range(n_paths):
        plt.bar(np.arange(n_layers) + n*width, avg_confidences[n], width=width, label=checkpoints[n])

    plt.legend()
    plt.title("Average Confidences by layer with threshold=" +str(threshold))
    plt.xlabel('layer number')
    plt.ylabel('avg confidence of layer on seen data')

    fig.savefig(save_path+'avg_confidences.png')
    plt.close()

    # Plotting the total accuracy of the model ###########################################

    fig = plt.figure()

    for n in range(n_paths):
        plt.bar(np.arange(1) + n*width, total_accuracies[n], width=width, label=checkpoints[n])

    plt.legend()
    plt.title("total test accuracy with early exit -- threshold =" +str(threshold))
    plt.xlabel('layer number')
    plt.ylabel('test accuracy')

    fig.savefig(save_path+'total_accuracy.png')
    plt.close()
