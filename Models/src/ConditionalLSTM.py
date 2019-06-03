import collections
import os

import numpy as np
import sklearn.metrics as sk
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

# Defining file paths
filePath = os.path.dirname(__file__)
fullDataPath = os.path.join(filePath, '../resources/quote2vec/fullDataset/')
polSubsetPath = os.path.join(filePath, '../resources/quote2vec/nationalPolicy/')

# Defining hyperparameter space
embSize = 300
noClasses = 3
LSTMLayersVar = [1, 2, 3]
LSTMDimsVar = [50, 100, 200]
ReLuLayersVar = [1, 2, 3]
ReLuDimsVar = [50, 100, 200]
epochsVar = [30, 50, 70, 100, 200, 300]
L2Var = [0.0, 0.0001, 0.0003]
dropoutVar = [0.0, 0.2, 0.5, 0.7]


# Conditional LSTM architecture with variable dimensions in the form of number of LSTM layers and dimensions, ReLU
# layers and dimensions. The LSTM takes a single word embedding as input at each time step, and is conditioned on a
# one-hot vector
# Inspired by https://discuss.pytorch.org/t/example-of-many-to-one-lstm/1728/4 and
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class ConditionalLSTM(nn.Module):
    def __init__(self, LSTMLayers, LSTMDims, ReLULayers, ReLUDims):
        super(ConditionalLSTM, self).__init__()
        self.LSTMLayers = LSTMLayers
        self.LSTMDims = LSTMDims
        self.ReLuLayers = ReLULayers
        self.ReLuDims = ReLUDims
        # Initialize LSTM model using the PyTorch LSTM class and user-specified parameters
        self.lstm = nn.LSTM(embSize, LSTMDims, LSTMLayers)
        # Initialize initial hidden state of the LSTM, all values being zero
        self.hiddenState = self.initializeHiddenState()

        # Initialize linear layers mapping from LSTM model to RelU Layers, and initialize ReLu layers
        denseLayers = collections.OrderedDict()
        denseLayers["linear0"] = torch.nn.Linear(LSTMDims, ReLUDims)
        denseLayers["ReLU0"] = torch.nn.ReLU()
        for i in range(ReLULayers - 1):
            denseLayers['linear{}'.format(i + 1)] = nn.Linear(ReLUDims, ReLUDims)
            denseLayers['ReLU{}'.format(i + 1)] = nn.ReLU()
        # Initialize dropout layer
        denseLayers['dropOut'] = nn.Dropout(p=0.5)
        # Final layer mapping from last ReLU layer to labels
        denseLayers['linear{}'.format(ReLULayers)] = nn.Linear(ReLUDims, noClasses)

        # Define hidden layers as a single unit
        self.hiddenLayers2Labels = nn.Sequential(denseLayers)

    # A forward pass over the full model, running the words through LSTM layers one at a time, and running the output
    # of running the final word through the LSTM layers through the remaining hidden layers, returning class
    # probabilities as 'score'
    def forward(self, quote):
        for word in quote:
            lstmOut, self.hiddenState = self.lstm(word.view(len(word), 1, -1), self.hiddenState)
            labelSpace = self.hiddenLayers2Labels(lstmOut.view(len(word), -1))
        score = f.log_softmax(labelSpace, dim=1)
        return score

    # Initializes empty hidden state with dimensions [number of layers, batch size, LSTM dimensions], contains cell
    # state and output
    def initializeHiddenState(self):
        return torch.zeros(self.LSTMLayers, 1, self.LSTMDims), torch.zeros(self.LSTMLayers, 1, self.LSTMDims)


# Parse data from data file. Must have a single quote on each line, feature vector separated by '], [', and each value
# in said features separated by ', ', and the actual label of the quote as the last word. Returns a matrix of word
# embeddings, and a vector of quote labels
def loadData(path):
    with open(path, 'r', encoding='utf-8') as inFile:
        data = []
        for tempFeatureMatrix in inFile.readlines():
            features = tempFeatureMatrix.split(']\', \'[')
            featureMatrix = []
            for feature in features:
                feature = feature.replace('[', '').replace(']', '').replace('\'', '').replace('\n', '').split(', ')
                feature = [float(i) for i in feature]
                featureMatrix.append(feature)
            data.append((featureMatrix[:-1], int(featureMatrix[-1][0])))
        return data


def train(data, model, lossFunction, optimizer, epochs):
    epochLoss = 0.0
    for epoch in range(epochs):
        for quote, label in data:
            # Clear out gradients and hidden layers
            model.zero_grad()
            model.hiddenLayers = model.initializeHiddenState()

            # Build tensors for quotes and labels
            target = torch.tensor([label])
            features = []
            for feature in quote:
                features.append([feature])

            # Extract class probability distributions for training data
            labelScores = model(torch.tensor(features))

            # Calculate loss using the defined loss function, backpropagate loss and optimize model based on loss
            loss = lossFunction(labelScores, target)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
        print('Epoch %d, loss: %.5f' % (epoch + 1, epochLoss / 1000))
        epochLoss = 0


# Tests a pre-trained model
def test(data, model):
    predictedLabels = []
    actualLabels = []

    # Code is run with torch.no_grad(), as the model is not to be trained during testing
    with torch.no_grad():
        for quote, label in data:
            # Generate quote feature vector
            features = []
            for feature in quote:
                features.append([feature])

            # Extract class probability distributions for test data
            labelScores = model(torch.tensor(features))

            # Extract prediction, and add prediction and actual label to each their own list
            predicted = torch.argmax(labelScores.data, dim=1)
            predictedLabels.extend(predicted.numpy())
            actualLabels.append(label)

    # Generate confusion matrix, and extract evaluation measures using scikit-learn
    cMatrix = sk.confusion_matrix(actualLabels, predictedLabels, labels=[0, 1, 2])
    print("Confusion matrix:")
    print(cMatrix)
    cm = cMatrix.astype('float') / cMatrix.sum(axis=1)[:, np.newaxis]
    classAcc = cm.diagonal()
    acc = sk.accuracy_score(actualLabels, predictedLabels)
    f1Macro = sk.f1_score(actualLabels, predictedLabels, average='macro')
    f1Micro = sk.f1_score(actualLabels, predictedLabels, average='micro')
    print("Class acc:", classAcc)
    print("Accuracy: %.5f" % acc)
    print("F1-macro:", f1Macro)
    print("F1-micro:", f1Micro)
    return classAcc, acc, f1Macro


# Method for running benchmarking for the full hyperparameter space
def runFullBenchmark():
    with open(os.path.join(filePath, '../out/LSTM_benchmarkNoAvg.csv'), 'w') as outFile:
        outFile.write("epochs,LSTMLayers,LSTMDims,ReLULayers,ReLUDims,L2,totalAcc,f1,For,Against,Neutral\n")
        for LSTMLayer in LSTMLayersVar:
            for LSTMDim in LSTMDimsVar:
                for ReLULayer in ReLuLayersVar:
                    for ReLUDim in ReLuDimsVar:
                        for L2 in L2Var:
                            runSpecificBenchmark(fullDataPath, LSTMLayer, LSTMDim, ReLULayer, ReLUDim, L2, True,
                                                 outFile)


# Method for running a benchmark using specific parameters
def runSpecificBenchmark(path, LSTMLayers, LSTMDims, ReLULayers, ReLUDims, L2, fullRun, outFile):
    # Load training and test data
    trainingData = loadData(path + 'trainData.txt')
    testData = loadData(path + 'testData.txt')

    # Initiate the LSTM model using user-defined parameters
    model = ConditionalLSTM(LSTMLayers, LSTMDims, ReLULayers, ReLUDims)

    # Initialize optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=L2)
    lossFunction = nn.NLLLoss()
    if not fullRun:
        outFile.write("epochs,LSTMLayers,LSTMDims,ReLULayers,ReLUDims,L2,totalAcc,f1,For,Against,Neutral\n")
    # Run train and test for model, and print out benchmark at each epoch count in the epoch hyperparameter space
    for i in range(len(epochsVar)):
        if i == 0:
            # Train for x epochs and print  benchmark, x being the first value in the epoch hyperparameter space
            train(trainingData, model, lossFunction, optimizer, epochsVar[i])
            classAcc, totalAcc, f1 = test(testData, model)
            outFile.write(
                "%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" %
                (epochsVar[i], LSTMLayers, LSTMDims, ReLULayers, ReLUDims, L2, totalAcc, f1, classAcc[0],
                 classAcc[1], classAcc[2]))
            outFile.flush()
        # Train for x1-x0 epochs, x1 being the epoch value at the current index i in the epoch hyperparameter space, x0
        # being the value at index i-1, and print the resulting benchmark
        else:
            train(trainingData, model, lossFunction, optimizer, epochsVar[i] - epochsVar[i - 1])
            classAcc, totalAcc, f1 = test(testData, model)
            outFile.write(
                "%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" %
                (epochsVar[i], LSTMLayers, LSTMDims, ReLULayers, ReLUDims, L2, totalAcc, f1, classAcc[0],
                 classAcc[1], classAcc[2]))
            outFile.flush()
