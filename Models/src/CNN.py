import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import sklearn.metrics as sk


fullDataPath = '../resources/fullDataset/'
polSubsetPath = '../resources/nationalPolicy/'

embSize = 372  # 300 sentence embeddings, 63 politician embeddings and 9 party embeddings
layersVar = [1, 2, 3]
kernelSize = 2
stride = 1
padding = 0
epochsVar = [30, 50, 70]  # Also test 100 and 300
L2Var = [0.0, 1e-4, 3e-4]
dropoutVar = [0.0, 0.2, 0.5, 0.7, 1.0]

# Inspired by https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/ and
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
class CNN(nn.Module):
    def __init__(self, layers, embDims, noLabels):
        super(CNN, self).__init__()
        self.layers = layers
        self.stride = stride
        self.padding = padding
        self.embDims = embDims
        self.CNNLayers = collections.OrderedDict()
        self.CNNLayers['CNNLayer0'] = nn.Sequential(
            nn.Conv2d(embSize, 1, kernel_size=(1, embDims), stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, kernelSize), stride=stride)
        )
        for i in range(layers-1):
            self.CNNLayers['CNNLayer{}'.format(i+1)] = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(1, embDims), stride=stride, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, kernelSize), stride=stride)
            )
        self.CNNLayers['dropOut'] = nn.Dropout(p=0.5)
        # Generate fully connected linear layers
        self.CNNLayers['linear0'] = nn.Linear(in_features=64, out_features=1000)
        self.CNNLayers['linear1'] = nn.Linear(in_features=1000, out_features=noLabels)
        self.CNNLayers = nn.ModuleDict(self.CNNLayers)

    def forward(self, quote):
        out = self.CNNLayers['CNNLayer0'](quote.view(1, embSize, 1))
        for i in range(self.layers-1):
            out = self.CNNLayers['CNNLayer{}'.format(i+1)](out)
        out = self.CNNLayers['dropOut'](out)
        out = self.CNNLayers['linear0'](out)
        out = self.CNNLayers['linear1'](out)
        return out


def loadData(path):
    with open(path, 'r', encoding='utf-8') as inFile:
        data = []
        for quoteVec in inFile.readlines():
            quoteVec = quoteVec.strip('\n').strip('[').strip(']').split(', ')
            quoteVec = [float(i) for i in quoteVec]
            data.append((quoteVec[:-1], int(quoteVec[-1])))
        return data


def run(path, layers, noClasses, L2, epochs):
    trainingData = loadData(path + 'trainData.txt')
    testData = loadData(path + 'testData.txt')
    model = CNN(layers, embSize, noClasses)
    lossFunction = nn.NLLLoss()
    print(type(model.parameters()))
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=L2)
    train(trainingData, model, lossFunction, optimizer, epochs)
    #return test(testData, model)


def train(data, model, lossFunction, optimizer, epochs):
    epochLoss = 0.0
    for epoch in range(epochs):
        for quote, label in data:
            # Clear out gradients and hidden layers
            model.zero_grad()
            inputs = torch.tensor([quote])
            target = torch.tensor([label])
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
    f1 = sk.f1_score(actualLabels, predictedLabels, average='macro')
    print("Class acc:", classAcc)
    print("Accuracy: %.5f" % acc)
    print("F1-macro:", f1)
    return classAcc, acc, f1


run(fullDataPath, layersVar[0], 3, L2Var[0], epochsVar[0])
