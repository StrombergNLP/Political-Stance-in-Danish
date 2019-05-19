import nltk
import pandas as pd


def preprocessQuotes(rawQuotes, polSubset, avgEmbeddings):
    global w2vmodel
    w2vmodel = {}
    loadModel()
    inFile = pd.read_csv('../Resources/parsedQuotes/' + rawQuotes, sep=';', encoding='UTF-16', header=0, quoting=1)
    # Drop quotes identified as false positives from data
    inFile = inFile.dropna()
    if polSubset:
        inFile = inFile[inFile['subTopic'] == 'p']

    with open('../out/featurevecs.txt', 'w', encoding='utf-8') as outFile:
        with open('../resources/fanMapping.txt', 'r', encoding='utf-8') as fanMapFile:
            fanVec = fanMapFile.readlines()
            fanMap = {x.split(';')[0]: x.split(';')[1].replace('\n', '') for x in fanVec}
            for index, row in inFile.iterrows():
                quoteTokens = nltk.word_tokenize(row['quote'])
                if avgEmbeddings:
                    embeddedQuote = quote2AvgVec(quoteTokens)
                    finalVec = embeddedQuote + genFeatureVec(row) + [int(fanMap[row['fan']])]
                    outFile.write(str(finalVec)+'\n')
                else:
                    embeddedQuote = quote2vec(quoteTokens)
                    featureVec = [0.0]*300
                    polPartyVec = genFeatureVec(row)
                    for i in range(len(polPartyVec)):
                        featureVec[i] = polPartyVec[i]
                    finalVec = [featureVec] + embeddedQuote + [[int(fanMap[row['fan']])]]
                    outFile.write(str(finalVec)+'\n')



def loadModel():
    global w2vmodel
    with open('../resources/w2vModel/filteredModel.csv', encoding='utf-8') as inFile:
        for i, line in enumerate(inFile):
            line = line.split(';')
            w2vmodel.update({line[0]: line[1]})


def quote2AvgVec(quoteTokens):
    global w2vmodel
    # Embeddings of size 300
    emb = [0.0]*300
    # Count number of words in quote
    n = 0
    for word in quoteTokens:
        if word in w2vmodel:
            n = n + 1
            wordEmb = w2vmodel[word].split(', ')
            vector = [float(i) for i in wordEmb]
            emb = [x + y for x, y in zip(emb, vector)]
    return [x / n for x in emb]


def quote2vec(quoteTokens):
    global w2vmodel
    quoteEmb = []
    for word in quoteTokens:
        if word in w2vmodel:
            wordEmb = w2vmodel[word].split(', ')
            vector = [float(i) for i in wordEmb]
            quoteEmb.append(vector)
    return quoteEmb


def genFeatureVec(quoteVec):
    polMap = pd.read_csv('../Resources/mapping/politicianMapping.csv', sep=';', encoding='UTF-16', quoting=1,
                         header=None, index_col=0)
    partyMap = pd.read_csv('../Resources/mapping/partyMapping.csv', sep=';', encoding='UTF-16', quoting=1, header=None,
                           index_col=0)
    polVec = [0]*63
    partyVec = [0]*9
    # Extract one-hot feature for politician of quote
    polVec[int((polMap.loc[quoteVec.politician, 1]))] = 1
    # Extract one-hot feature for party affiliation of politician of quote
    partyVec[int((partyMap.loc[quoteVec.party, 1]))] = 1
    return polVec + partyVec


def generateModelSubset():
    inFile = pd.read_csv('../Resources/parsedQuotes/quote_db.csv', sep=';', encoding='UTF-16', header=0, quoting=1)
    # Drop quotes identified as false positives from data
    inFile = inFile.dropna()
    words = set()
    wordEmbeddings = list()
    wordsWOembeddings = list()
    for index, row in inFile.iterrows():
        quoteTokens = nltk.word_tokenize(row['quote'])
        for word in quoteTokens:
            words.add(word.lower())
    with open('../Resources/w2vmodel/daft.vec', 'r', encoding='utf-8') as modelFile:
        for i, line in enumerate(modelFile):
            line = line.split(' ')
            word = line[0]
            vector = str(line[1:-1]).replace('[', '').replace(']', '').replace('\'', '')
            if word in words:
                wordEmbeddings.append([word, str(vector)])
            else:
                wordsWOembeddings.append(word)
    pd.DataFrame(wordEmbeddings).to_csv('../resources/w2vmodel/filteredModel.csv', encoding='utf-8', index=None, sep=';'
                                        , header=['word', 'vector'])
    pd.DataFrame(wordsWOembeddings).to_csv('../resources/missingEmbeddings.csv', encoding='utf-8')


def splitTrainingTestData(avgEmbeddings):
    with open('../out/featurevecs.txt', 'r', encoding='utf-8') as inFile:
        vectors = [x.replace('\n', '').split(', ') for x in inFile.readlines()]
        # Splitting feature vectors into training and test set, with 80 % in training and 20 % in test
        trainSet = vectors[:int(vectors.__len__()*0.8)]
        testSet = vectors[int(vectors.__len__()*0.8):]
        print('TrainSet size: {}\tTestSet size: {}'.format(trainSet.__len__(), testSet.__len__()))
        if avgEmbeddings:
            with open('../out/trainData.txt', 'w', encoding='utf-8') as outFile:
                for vector in trainSet:
                    outFile.write(str(vector).replace('[\'', '').replace(']\'', '').replace('\'', '')+'\n')
            with open('../out/testData.txt', 'w', encoding='utf-8') as outFile:
                for vector in testSet:
                    outFile.write(str(vector).replace('[\'', '').replace(']\'', '').replace('\'', '')+'\n')
        else:
            with open('../out/trainData.txt', 'w', encoding='utf-8') as outFile:
                for vector in trainSet:
                    outFile.write(str(vector) + '\n')
            with open('../out/testData.txt', 'w', encoding='utf-8') as outFile:
                for vector in testSet:
                    outFile.write(str(vector) + '\n')


def genPoliticsSubset(avgEmbeddings):
    preprocessQuotes('quote_db.csv', polSubset=True, avgEmbeddings=avgEmbeddings)
    splitTrainingTestData(avgEmbeddings)


def genFullDataset(avgEmbeddings):
    preprocessQuotes('quote_db.csv', polSubset=False, avgEmbeddings=avgEmbeddings)
    splitTrainingTestData(avgEmbeddings)


genFullDataset(avgEmbeddings=False)

#generateModelSubset()