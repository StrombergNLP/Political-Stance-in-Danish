import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import collections
import sklearn.metrics as sk
import os

filePath = os.path.dirname(__file__)
fullDataPath = os.path.join(filePath, '../resources/avgQuote2Vec/fullDataset/')
polSubsetPath = os.path.join(filePath, '../resources/avgQuote2Vec/nationalPolicy/')

embSize = 372  # 300 sentence embeddings, 63 politician embeddings and 9 party embeddings
noClasses = 3
LSTMLayersVar = [1, 2]
LSTMDimsVar = [50, 100, 200]
ReLuLayersVar = [1, 2]
ReLuDimsVar = [50, 100, 200]
epochsVar = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
# epochsVar = [30, 50, 70, 100, 200, 300]
L2Var = [0.0, 0.0001, 0.0003]
dropoutVar = [0.0, 0.2, 0.5, 0.7]


# Inspired by https://discuss.pytorch.org/t/example-of-many-to-one-lstm/1728/4 and
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class LSTM(nn.Module):
    def __init__(self, LSTMLayers, LSTMDims, ReLULayers, ReLUDims, biDirectional):
        super(LSTM, self).__init__()
        self.LSTMLayers = LSTMLayers
        self.LSTMDims = LSTMDims
        self.ReLuLayers = ReLULayers
        self.ReLuDims = ReLUDims
        self.noDirections = 1
        if biDirectional:
            self.noDirections = 2
        self.biDirectional = biDirectional
        self.lstm = nn.LSTM(embSize, LSTMDims, LSTMLayers, bidirectional=biDirectional)
        # Initialize initial hidden state of the LSTM, all values being zero
        self.hiddenLayers = self.initializeHiddenLayers()

        # Initialize linear layers mapping to RelU Layers, and initialize ReLu layers
        denseLayers = collections.OrderedDict()
        denseLayers["linear0"] = torch.nn.Linear(LSTMDims*self.noDirections, ReLUDims*self.noDirections)
        denseLayers["ReLU0"] = torch.nn.ReLU()
        for i in range(ReLULayers-1):
            denseLayers['linear{}'.format(i+1)] = nn.Linear(ReLUDims*self.noDirections, ReLUDims*self.noDirections)
            denseLayers['ReLU{}'.format(i+1)] = nn.ReLU()
        # Initialize dropout layer
        denseLayers['dropOut'] = nn.Dropout(p=0.5)
        # Final layer mapping from last ReLU layer to labels
        denseLayers['linear{}'.format(ReLULayers)] = nn.Linear(ReLUDims*self.noDirections, noClasses)
        self.hiddenLayers2Labels = nn.Sequential(denseLayers)

    def forward(self, quote):
        lstmOut, self.hiddenLayers = self.lstm(quote.view(len(quote), 1, -1), self.hiddenLayers)
        labelSpace = self.hiddenLayers2Labels(lstmOut.view(len(quote), -1))
        score = f.log_softmax(labelSpace, dim=1)
        return score

    def initializeHiddenLayers(self):
        return torch.zeros(self.LSTMLayers*self.noDirections, 1,  self.LSTMDims), \
               torch.zeros(self.LSTMLayers*self.noDirections, 1, self.LSTMDims)


def loadData(path):
    with open(path, 'r', encoding='utf-8') as inFile:
        data = []
        for quoteVec in inFile.readlines():
            quoteVec = quoteVec.replace('[', '').replace(']', '').replace('\'', '')
            quoteVec = quoteVec.split(', ')
            quoteVec = [float(i) for i in quoteVec]
            data.append((quoteVec[:-1], int(quoteVec[-1])))
        return data


def train(data, model, lossFunction, optimizer, epochs):
    epochLoss = 0.0
    for epoch in range(epochs):
        for quote, label in data:
            # Clear out gradients and hidden layers
            model.zero_grad()
            model.hiddenLayers = model.initializeHiddenLayers()
            target = torch.tensor([label])
            inputs = torch.tensor([quote])
            labelScores = model(inputs)
            loss = lossFunction(labelScores, target)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
        print('Epoch %d, loss: %.5f' % (epoch + 1, epochLoss / 1000))
        epochLoss = 0


def test(data, model):
    predictedLabels = []
    actualLabels = []
    with torch.no_grad():
        for quote, label in data:
            inputs = torch.tensor([quote])
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, dim=1)
            predictedLabels.extend(predicted.numpy())
            actualLabels.append(label)

    # Generate confusion matrix
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


def runFullBenchmark(biDirectional):
    with open(os.path.join(filePath, '../out/LSTM_benchmark.csv'), 'w') as outFile:
        outFile.write("epochs,LSTMLayers,LSTMDims,ReLULayers,ReLUDims,L2,totalAcc,f1,For,Against,Neutral\n")
        for LSTMLayer in LSTMLayersVar:
            for LSTMDim in LSTMDimsVar:
                for ReLULayer in ReLuLayersVar:
                    for ReLUDim in ReLuDimsVar:
                        for L2 in L2Var:
                            runSpecificBenchmark(fullDataPath, LSTMLayer, LSTMDim, ReLULayer, ReLUDim, L2, True,
                                                 outFile, biDirectional)


def runSpecificBenchmark(path, LSTMLayers, LSTMDims, ReLULayers, ReLUDims, L2, fullRun, outFile, biDirectional):
    lossFunction = nn.NLLLoss()
    trainingData = loadData(path + 'trainData.txt')
    testData = loadData(path + 'testData.txt')
    model = LSTM(LSTMLayers, LSTMDims, ReLULayers, ReLUDims, biDirectional)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=L2)
    if not fullRun:
        outFile.write("epochs,LSTMLayers,LSTMDims,ReLULayers,ReLUDims,L2,totalAcc,f1,For,Against,Neutral\n")
    for i in range(len(epochsVar)):
        if i == 0:
            train(trainingData, model, lossFunction, optimizer, epochsVar[i])
            classAcc, totalAcc, f1 = test(testData, model)
            outFile.write(
               "%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" %
               (epochsVar[i], LSTMLayers, LSTMDims, ReLULayers, ReLUDims, L2, totalAcc, f1, classAcc[0],
                classAcc[1], classAcc[2]))
            outFile.flush()
        else:
            train(trainingData, model, lossFunction, optimizer, epochsVar[i]-epochsVar[i-1])
            classAcc, totalAcc, f1 = test(testData, model)
            outFile.write(
                "%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" %
                (epochsVar[i], LSTMLayers, LSTMDims, ReLULayers, ReLUDims, L2, totalAcc, f1, classAcc[0],
                 classAcc[1], classAcc[2]))
            outFile.flush()


# runFullBenchmark(biDirectional=True)
with open(os.path.join(filePath, '../out/LSTM_benchmark.csv'), 'w') as outFile:
    runSpecificBenchmark('../resources/avgQuote2Vec/fullDataset/', 1, 50, 1, 100, 0,
                         outFile=outFile, fullRun=False, biDirectional=False)
