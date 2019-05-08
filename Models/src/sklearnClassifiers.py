from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import numpy as np

fullDataPath = '../resources/fullDataset/'
polSubsetPath = '../resources/nationalPolicy/'


def run(path, classifierType):
    if classifierType == 'randomForest':
        classifier = RandomForestClassifier(random_state=0)
    elif classifierType == 'GNB':
        classifier = GaussianNB()
    else:
        print('Incorrect classifier type')
        return
    train(classifier, path)
    testResults, classLabels = test(classifier, path)
    # Generate 3x3 confusion matrix and populate with zeroes
    cMatrix = [[0]*3 for i in range(3)]
    # Increment cells in confusion matrix according to predictions
    for i in range(testResults.__len__()):
        cMatrix[int(testResults[i])][int(classLabels[i])] = cMatrix[int(testResults[i])][int(classLabels[i])] + 1
    trueClass = cMatrix[0][0]+cMatrix[1][1]+cMatrix[2][2]
    falseClass = cMatrix[0][1]+cMatrix[0][2]+cMatrix[1][0]+cMatrix[1][2]+cMatrix[2][0]+cMatrix[2][1]
    print('Correct classifications {} \nMis-classifications {} \nAccuracy {}'
          .format(trueClass, falseClass, trueClass/len(testResults)))
    print('\t\tActual class labels\n'
          '     For\t   Against\t   Neutral\n'
          'For\t    {}\t    {}\t        {}\n'
          'Against\t{}\t    {}\t        {}\n'
          'Neutral\t{}\t    {}\t        {}'.format(cMatrix[0][0], cMatrix[0][1], cMatrix[0][2], cMatrix[1][0],
                                                   cMatrix[1][1], cMatrix[1][2], cMatrix[2][0], cMatrix[2][1],
                                                   cMatrix[2][2]))


def train(model, path):
    with open(path+'trainData.txt', 'r', encoding='utf-8') as testFile:
        trainData = [np.array(x.replace('\n', '').replace('[', '').replace(']', '').split(', ')).astype(np.float) for x
                     in testFile.readlines()]
        classLabels = [x[-1] for x in trainData]
        trainData = [x[:-1] for x in trainData]
        model.fit(trainData, classLabels)
    return model


def test(model, path):
    with open(path+'testData.txt', 'r', encoding='utf-8') as testFile:
        testData = [np.array(x.replace('\n', '').replace('[', '').replace(']', '').split(', ')).astype(np.float) for x
                    in testFile.readlines()]
        classLabels = [x[-1] for x in testData]
        testData = [x[:-1] for x in testData]
        return list(model.predict(testData)), classLabels

# randomForest or GNB
run(polSubsetPath, 'GNB')

