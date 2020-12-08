

###########################################################################################################################################
#a)ANN is used to solve a classification question. Hope it can predict if certain variations of a SNP will lead to a disease or not
# Iteratively process samples in the training set
# Compare the predicted value and the actual value of the output layer after the neural network
# Backpropagation from the output layer through the hidden layer to the input layer to minimize the error and update the connection weights


#b)Pseudo code
#  import dataset and split data
#  build network and initialize weights
#  feedforward propagate compute each one neuron outputs
#  backward propagate transfer errors and update weights
#  train model using train sets and save weights to compute accuracy of test set
###########################################################################################################################################


#c) the code

#!/usr/bin/env python3

import sys
import math
import random

#O(1)
def activationSigmoid(z):
    ''' The sigmoid function '''
    return (1.0 / (1.0 + math.exp(-z)))
#O(1)
def dsigmoid1(y):
  '''
  y = activationSigmoid(z)
  a part of computing derivative
  '''
  return y * (1 - y)
#O(1)
def activationR(x):
  '''
  LeakyReLU function
  '''
  if x >= 0:
    x =x
  if x < 0:
    x = 0.2*x
  return x
#O(1)  
def getFileName(data):
  #Nicola
    ''' get the filename from command line arguments or ask for input '''
    if len(sys.argv) == 1:
        filename = input( "Enter filename containing the dataset:")
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        print("Too many arguments.")
        sys.exit(1)
    return filename
#O(m*i), where m is the number of line in dataset, i is the number of elements in a line
def importDataset(file):
  #Nicola
    ''' Assuming there is no header fill a 2D matrix with float numbers '''
    dataset = list()
    for line in file:
        # do not add empty lines
        if line.strip() != "":
            dataset.append(line.split())
            for i in range(len(dataset[-1])):
                dataset[-1][i] = float(dataset[-1][i])
    return dataset
#O(1)
def initializeNN (n_inputlayer,n_hiddenlayer,n_outputlayer):
    #reference Nicola's code structure
    
    '''
    n_inputlayer: the number of input layer neuros
    n_hiddenlayer: the number of hidden layer neuros
    n_outputlayer: the number of output layer neuros
    '''
    network = list()
    hidden_layer = dict()
    output_layer = dict()
    #build a weights matrix and save in network
    #n_inputlayer + 1: the number of input layer neurons add 1 bias neuron
    hidden_layer = [{'weights':[[random.random() for i in range (n_inputlayer + 1)] for j in range(n_hiddenlayer)]}]
    network.append (hidden_layer)
    #n_hiddenlayer + 1: the number of hidden layer neurons add 1 bias neuron
    output_layer = [{'weights': [[random.random() for i in range(n_hiddenlayer + 1)] for j in range(n_outputlayer)]}]
    network.append(output_layer)
    return network
#O(56*27)
def z_inputlayer ( hidden_layer, inputs):
    '''
    Compute inputs of each neuron of hidden layer based on outputs of input layer
    and save in list z_input
    inputlayer_bias neuros is 1
    '''
    z_input0 = 0.0   
    z_input = []
    
    for i in range(len(hidden_layer[0]['weights'])):
        inputlayer_bias = hidden_layer[0]['weights'][i][-1]
        for j in range(len(hidden_layer[0]['weights'][0])-1):
            z_input0 += hidden_layer[0]['weights'][i][j]*inputs[j]
        z_input0 +=  inputlayer_bias  
        z_input.append(z_input0)    
    return z_input
#O(1*56)
def z_hiddenlayer (output_layer, z_input):
    '''
    Compute inputs of each neuron based on outputs of hidden layer
    and save in list z_hidden
    hiddenlayer_bias neuron is 1
    '''
    z_hidden0 = 0.0
    z_hidden = []
    
    for i in range(len(output_layer[0]['weights'])):
        hiddenlayer_bias = output_layer[0]['weights'][i][-1]
        for j in range(len(output_layer[0]['weights'][0])-1):
            z_hidden0 += output_layer[0]['weights'][i][j]*activationR(z_input[j]) 
        z_hidden0 += hiddenlayer_bias 
        z_hidden.append(z_hidden0)  
    return z_hidden
#O(56*27 + 1*56 + 56 + 1 = 1625)
def ForwardPropagate(network, inputs):
    '''
    return computed results of output value in each layer
    through activated function and save in list n_inputs
    '''
    hidden_layer = network[0]
    output_layer = network[1]
    #O(56*27)
    z_input = z_inputlayer (hidden_layer, inputs)
    #O(1*56)
    z_hidden = z_hiddenlayer (output_layer, z_input)
    
    n0_inputs=[]
    n1_inputs=[]
    
    for i in range(len(hidden_layer[0]['weights'])):
        n0_inputs.append (activationR(z_input[i]))
    for j in range(len(output_layer[0]['weights'])):
       n1_inputs.append(activationSigmoid(z_hidden[j]))
    n_inputs = [n0_inputs]+[n1_inputs]
    
    return n_inputs
#O(1+56+56+7 = 119)  
def backward_propagate_error(network, actual, neurons):
    '''
    compute error and delta
    '''
    hidden_layer = network[0]
    output_layer = network[1]
    
    sig_wp = 0.0
    sig_wp0 = 0.0
    errors0 = []
    errors1 = []
    errors= [errors0] + [errors1]
    neuron = {}

    delta1 = []
    delta0 = []
    
    neuron['output'] = neurons
   #for output layer:
    errors1.append(actual - neuron['output'][1][0])
    delta1.append(errors1[0] * dsigmoid1(neuron['output'][1][0]))
    #for hidden layer:
    for i in range(len(output_layer[0]['weights'][0])-1): 
          sig_wp = activationSigmoid(output_layer[0]['weights'][0][i]*neuron['output'][0][i])
          errors0.append(output_layer[0]['weights'][0][i]*(sig_wp - actual)*sig_wp*sig_wp*(1/sig_wp-1))
    
    for i in range(len(output_layer[0]['weights'][0])-1): 
            delta0.append(errors0[i]*dsigmoid1(neuron['output'][0][i]))
     # compute bias delta       
    sig_wp0 = activationSigmoid(output_layer[0]['weights'][0][-1])
    delta0.append(output_layer[0]['weights'][0][-1]*(sig_wp0 - actual)*sig_wp0*sig_wp0*(1/sig_wp0-1)*dsigmoid1(1)) 
    ndeltas  = [delta0]+[delta1]
    
    return ndeltas
#O(56*27 +56 +1 = 1569) 
def update_weight(network, inputs, rate, neurons, deltas,actual):
    '''
    input a sample data
    rate: learning rate
    update degree of weight: delta*output(last layer)
    '''
    neuron = {}
    hidden_layer = network[0]
    output_layer = network[1]
    neuron['output'] = neurons
    #for hidden layer:
    for i in range(len(hidden_layer[0]['weights'])):
        for j in range(len(hidden_layer[0]['weights'][0])-1):
              hidden_layer[0]['weights'][i][j] += rate * deltas[0][i]*inputs[j]
        hidden_layer[0]['weights'][i][-1] += rate * deltas[0][-1]
    #for output layer: 
    for i in range(len(output_layer[0]['weights'])):
        for j in range(len(output_layer[0]['weights'][0])-1):
            output_layer[0]['weights'][i][j] += rate * deltas[1][i]* neuron['output'][0][j]
        output_layer[0]['weights'][i][-1] += rate * deltas[1][-1]
    
    network[0] = hidden_layer 
    network[1] = output_layer
    
    return network

#O(n*(m*(1625 + 119 + 1569) + 4*m_test) where n is the number of epochs
                                        #and m is the size of dataset
def train (dataset, epochs, n_hiddenlayer, rate):
    '''
    Iteratively process samples until the train sets samples used up.
    finishing one epoch, save weights
    and use test set to do feedforward propagation
    and compute predicted accuracy of test set.
    '''
    network = initializeNN(n_inputlayer,n_hiddenlayer,n_outputlayer)
    
    for i in range(epochs):
        acc_rate = []
        for row in dataset:
            inputs = row[0:27]
            actual = row[-1]
           
            neurons = ForwardPropagate(network, inputs)
            n_deltas = backward_propagate_error(network, actual, neurons)
            network = update_weight(network, inputs, rate,neurons,n_deltas,actual)
        
        accuracy_test = test (network, testdata) 
        print("\nTHIS IS", i, "TRAIN ITEM", accuracy_test, "\n")

    return network    
#O(3*m_test + m_test)   where m_test is the size of test set   
def test (network, testdata):
    '''
    use test set to do feedforward propagation and compute predicted accuracy of test set
    '''
    inputs = [testdata[i][0:27] for i in range(len(testdata))]
    predicted = [ForwardPropagate(network, row)[1] for row in inputs]
    actual = [testdata[i][-1] for i in range(len(testdata))]
    
    accuracy = accuracy_metric(actual, predicted)
    
    return accuracy

#O(m)  where m is the size of dataset
def accuracy_metric (actual, predicted):
    # part of Nicola
    correct = 0
    
    for i in range(len(predicted)):
      if predicted[i][0] >= 0.5:
            predicted[i][0] = 1.0
            
      elif predicted[i][0] < 0.5:
             predicted[i][0] = 0.0
             
      if actual[i] == predicted[i][0]:
            correct += 1
           
    accuracy_percentage = correct / float(len(actual)) * 100.0
    
    return accuracy_percentage

###main code###
#O(2 + m*i + 2 + m_test*i + n*(m*(3313) + 4*m_test)
#where n is the number of epochs and m is the size of dataset
# i is the number of elements in a line of data set 
try:
  #Nicola
    fileName_train = getFileName(train)
    dataFile_train = open(fileName_train, 'r')
    dataset = importDataset(dataFile_train)
    
    fileName_test = getFileName(test)
    dataFile_test = open(fileName_test, 'r')
    
    testdata = importDataset(dataFile_test)
except IOError as e:
    print("Can't open file \n Reason: " + str(e))
    sys.exit(1)


rate = 0.0001
epochs = 10000
n_hiddenlayer = 56
n_inputlayer = 27
n_outputlayer = 1

#O(n*(m*(3313) + 4*m_test) where n is the number of epochs
                            #and m is the size of dataset
network = train (dataset, epochs, n_hiddenlayer, rate)
