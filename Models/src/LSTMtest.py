import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import sklearn.metrics as sk

avgFullDataPath = '../resources/avgQuote2Vec/fullDataset/'
avgPolSubsetPath = '../resources/avgQuote2Vec/nationalPolicy/'
fullDataPath = '../resources/quote2vec/fullDataset/'
polSubsetPath = '../resources/quote2vec/nationalPolicy/'

embSizeVar = [300, 372]  # 300 sentence embeddings, 63 politician embeddings and 9 party embeddings
LSTMLayersVar = [1]
LSTMDimsVar = [200]
ReLuLayersVar = [1]
ReLuDimsVar = [100, 200]
epochsVar = [200]# [30, 50, 70] Also test 300
L2Var = [0.0, 0.0001, 0.0003]
dropoutVar = [0.0, 0.2, 0.5, 0.7, 1.0]


# Inspired by https://discuss.pytorch.org/t/example-of-many-to-one-lstm/1728/4 and
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class LSTM(nn.Module):
    def __init__(self, LSTMLayers, LSTMDims, ReLULayers, ReLUDims, embDims, noLabels):
        super(LSTM, self).__init__()
        self.LSTMLayers = LSTMLayers
        self.LSTMDims = LSTMDims
        self.ReLuLayers = ReLULayers
        self.ReLuDims = ReLUDims
        self.embDims = embDims
        self.lstm = nn.LSTM(embDims, LSTMDims, LSTMLayers)
        # Initialize initial hidden state of the LSTM, all values being zero
        self.hiddenLayers = self.initializeHiddenLayers()

        # Initialize linear layers mapping to RelU Layers, and initialize ReLu layers
        denseLayers = collections.OrderedDict()
        denseLayers["linear0"] = torch.nn.Linear(LSTMDims, ReLUDims)
        denseLayers["ReLU0"] = torch.nn.ReLU()
        for i in range(ReLULayers-1):
            denseLayers['linear{}'.format(i+1)] = nn.Linear(ReLUDims, ReLUDims)
            denseLayers['ReLU{}'.format(i+1)] = nn.ReLU()
        # Initialize dropout layer
        denseLayers['dropOut'] = nn.Dropout(p=0.5)
        # Final layer mapping from last ReLU layer to labels
        denseLayers['linear{}'.format(ReLULayers)] = nn.Linear(ReLUDims, noLabels)
        self.hiddenLayers2Labels = nn.Sequential(denseLayers)

    def forward(self, quote):
        lstmOut, self.hidden = self.lstm(quote.view(len(quote), 1, -1), self.hidden)
        labelSpace = self.hiddenLayers2Labels(lstmOut.view(len(quote), -1))
        score = F.log_softmax(labelSpace, dim=1)
        return score

    def initializeHiddenLayers(self):
        return torch.zeros(self.LSTMLayers, 1,  self.LSTMDims), torch.zeros(self.LSTMLayers, 1, self.LSTMDims)


def loadAvgVectors(path):
    with open(path, 'r', encoding='utf-8') as inFile:
        data = []
        for quoteVec in inFile.readlines():
            quoteVec = quoteVec.replace('\n', '').replace('[', '').replace(']', '').replace(', ', '')
            quoteVec = [float(i) for i in quoteVec]
            data.append((quoteVec[:-1], int(quoteVec[-1])))
        return data


def loadVectors(path):
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


def run(path, LSTMLayers, LSTMDims, ReLULayers, ReLUDims, noClasses, L2, epochs, avgQuote2Vec):
    lossFunction = nn.NLLLoss()
    if avgQuote2Vec:
        trainingData = loadAvgVectors(path + 'trainData.txt')
        testData = loadAvgVectors(path + 'testData.txt')
        model = LSTM(LSTMLayers, LSTMDims, ReLULayers, ReLUDims, embSizeVar[1], noClasses)
    else:
        trainingData = loadVectors(path + 'trainData.txt')
        testData = loadVectors(path + 'testData.txt')
        model = LSTM(LSTMLayers, LSTMDims, ReLULayers, ReLUDims, embSizeVar[0], noClasses)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=L2)
    train(trainingData, model, lossFunction, optimizer, epochs)
    return test(testData, model)


def train(data, model, lossFunction, optimizer, epochs):
    epochLoss = 0.0
    for epoch in range(epochs):
        for quote, label in data:
            # Clear out gradients and hidden layers
            model.zero_grad()
            model.hidden = model.initializeHiddenLayers()
            target = torch.tensor([label])
            for feature in quote:
                labelScores = model(torch.tensor([feature]))
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
            for feature in quote:
                labelScores = model(torch.tensor([feature]))
            predicted = torch.argmax(labelScores.data, dim=1)
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


def LSTMBenchmark(outPath, avgQuote2Vec):
    with open(outPath, 'w') as outFile:
        outFile.write("epochs,LSTMLayers,LSTMDims,ReLULayers,ReLUDims,L2,totalAcc,f1,For,Against,Neutral\n")
        for LSTMLayer in LSTMLayersVar:
            for LSTMDim in LSTMDimsVar:
                for ReLULayer in ReLuLayersVar:
                    for ReLUDim in ReLuDimsVar:
                        for epoch in epochsVar:
                            for L2 in L2Var:
                                if avgQuote2Vec:
                                    classAcc, totalAcc, f1 = run(avgFullDataPath, LSTMLayer, LSTMDim, ReLULayer, ReLUDim,
                                                                 3, L2, epoch, avgQuote2Vec)
                                else:
                                    classAcc, totalAcc, f1 = run(fullDataPath, LSTMLayer, LSTMDim, ReLULayer, ReLUDim,
                                                                 3, L2, epoch, avgQuote2Vec)
                                outFile.write(
                                    "%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" %
                                    (epoch, LSTMLayer, LSTMDim, ReLULayer, ReLUDim, L2, totalAcc, f1, classAcc[0],
                                     classAcc[1], classAcc[2]))
                                # Flushing to force write for each test, avoiding lost progress on system failure
                                outFile.flush()


LSTMBenchmark('../out/LSTM_benchmarkNoAvg.csv', avgQuote2Vec=False)
#run(fullDataPath, LSTMLayersVar[0], LSTMDimsVar[0], ReLuLayersVar[0], ReLuDimsVar[0], 3, L2Var[0], epochsVar[0], avgQuote2Vec=False)
