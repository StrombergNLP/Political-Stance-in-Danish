import numpy as np
import sklearn.metrics as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Defining file paths
fullDataPath = '../resources/avgQuote2Vec/fullDataset/'
polSubsetPath = '../resources/avgQuote2Vec/nationalPolicy/'


# Run the given classifier until it converges on a result
def run(path, classifierType):
    # Initialize classifier of type defined by user, or give error message
    if classifierType == 'randomForest':
        classifier = RandomForestClassifier(random_state=0)
    elif classifierType == 'GNB':
        classifier = GaussianNB()
    else:
        print('Incorrect classifier type')
        return

    # Train classifier using training data, and test using test data, extracting predicted and actual labels
    train(classifier, path)
    predictedLabels, actualLabels = test(classifier, path)

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


def train(model, path):
    # Parse data from data file. Must have a single quote on each line, each word separated by ', ', and the actual
    # label of the quote as the last word. Returns a matrix of word embeddings, and a vector of quote labels
    with open(path + 'trainData.txt', 'r', encoding='utf-8') as testFile:
        trainData = [np.array(x.replace('\n', '').replace('[', '').replace(']', '').split(', ')).astype(np.float) for x
                     in testFile.readlines()]
        classLabels = [x[-1] for x in trainData]
        trainData = [x[:-1] for x in trainData]
        # Train model
        model.fit(trainData, classLabels)
    return model


# Tests a pre-trained model
def test(model, path):
    # Parse data from data file. Must have a single quote on each line, each word separated by ', ', and the actual
    # label of the quote as the last word. Returns a matrix of word embeddings, and a vector of quote labels
    with open(path + 'testData.txt', 'r', encoding='utf-8') as testFile:
        testData = [np.array(x.replace('\n', '').replace('[', '').replace(']', '').split(', ')).astype(np.float) for x
                    in testFile.readlines()]
        classLabels = [x[-1] for x in testData]
        testData = [x[:-1] for x in testData]
        # Return predicted and actual class labels
        return list(model.predict(testData)), classLabels


# randomForest or GNB
run(fullDataPath, 'GNB')
