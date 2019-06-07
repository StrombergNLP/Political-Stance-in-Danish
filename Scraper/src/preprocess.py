import nltk
import pandas as pd


# Generates feature vectors for all quotes in a given quote dataset passed as 'rawQuotes'. Includes either just the
# 'national policy' subset or the full dataset, generates features for use in generation of average word embedding
# vectors or full word embedding matrices, and includes either the politician one-hot embedding feature vector, the
# party one-hot embedding feature vector, both or none of them
def preprocessQuotes(rawQuotes, polSubset, avgEmbeddings, incPol, incParty):
    # Load word embedding library
    global w2vmodel
    w2vmodel = {}
    loadModel()

    # Open quote dataset and drop quotes identified as false positives
    inFile = pd.read_csv('../Resources/parsedQuotes/' + rawQuotes, sep=';', encoding='UTF-16', header=0, quoting=1)
    inFile = inFile.dropna()

    # Drop quotes not in the 'national policy' sub-topic if indicated
    if polSubset:
        inFile = inFile[inFile['subTopic'] == 'p']

    with open('../out/featurevecs.txt', 'w', encoding='utf-8') as outFile:
        with open('../resources/fanMapping.txt', 'r', encoding='utf-8') as fanMapFile:
            fanVec = fanMapFile.readlines()
            fanMap = {x.split(';')[0]: x.split(';')[1].replace('\n', '') for x in fanVec}
            for index, row in inFile.iterrows():
                # Tokenize words
                quoteTokens = nltk.word_tokenize(row['quote'])
                # Generate average quote embeddings and add to outfile if indicated
                quoteID = row['quoteID']
                if avgEmbeddings:
                    embeddedQuote = quote2AvgVec(quoteTokens)
                    finalVec = embeddedQuote + genFeatureVec(row, incPol, incParty) + [int(fanMap[row['fan']])] + [quoteID]
                    outFile.write(str(finalVec) + '\n')
                # Generate word vector quote embeddings and add to outfile
                else:
                    embeddedQuote = quote2vec(quoteTokens)
                    # Generate one-hot feature vector, and fill in index(es) corresponding to relevant party and/or
                    # politician
                    if incParty or incPol:
                        featureVec = [0.0] * 300
                        polPartyVec = genFeatureVec(row, incPol, incParty)
                        for i in range(len(polPartyVec)):
                            featureVec[i] = polPartyVec[i]
                        finalVec = [featureVec] + embeddedQuote + [[int(fanMap[row['fan']])]] + [[quoteID]]
                    else:
                        finalVec = embeddedQuote + [[int(fanMap[row['fan']])]] + [[quoteID]]
                    outFile.write(str(finalVec) + '\n')


# Load word embeddings into memory, uses an embedding library on the form 'word;embedding'
def loadModel():
    global w2vmodel
    with open('../resources/wordembeddings/filteredModel.csv', encoding='utf-8') as inFile:
        for i, line in enumerate(inFile):
            line = line.split(';')
            w2vmodel.update({line[0]: line[1]})


# Generates an average vector over word embedding vectors, to get a single quote vector
def quote2AvgVec(quoteTokens):
    global w2vmodel
    # Embeddings of size 300
    emb = [0.0] * 300
    # Count number of words in quote
    n = 0
    for word in quoteTokens:
        if word in w2vmodel:
            n = n + 1
            wordEmb = w2vmodel[word].split(', ')
            vector = [float(i) for i in wordEmb]
            emb = [x + y for x, y in zip(emb, vector)]
    return [x / n for x in emb]


# Generate a matrix containing word embeddings
def quote2vec(quoteTokens):
    global w2vmodel
    quoteEmb = []
    for word in quoteTokens:
        if word in w2vmodel:
            wordEmb = w2vmodel[word].split(', ')
            vector = [float(i) for i in wordEmb]
            quoteEmb.append(vector)
    return quoteEmb


# Generate context-based feature vector, including one-hot vector representing the quoted politician, one-hot vector
# representing quoted politician's party, both vectors concatenated or an empty vector, based on user input
def genFeatureVec(quoteVec, incPol, incParty):
    polVec = []
    partyVec = []
    # Extract one-hot feature for politician of quote, using mapping of politician to index
    if incPol:
        polMap = pd.read_csv('../Resources/mapping/politicianMapping.csv', sep=';', encoding='UTF-16', quoting=1,
                             header=None, index_col=0)
        polVec = [0] * 63
        polVec[int((polMap.loc[quoteVec.politician, 1]))] = 1

    # Extract one-hot feature for party affiliation of politician of quote, using mapping of party to index
    if incParty:
        partyMap = pd.read_csv('../Resources/mapping/partyMapping.csv', sep=';', encoding='UTF-16', quoting=1,
                               header=None, index_col=0)
        partyVec = [0] * 9
        partyVec[int((partyMap.loc[quoteVec.party, 1]))] = 1

    return polVec + partyVec


# Extracts a subset of the word embedding library based on the vocabulary within the quote dataset, and saves that
# subset
def generateModelSubset():
    # Read quote dataset
    inFile = pd.read_csv('../Resources/parsedQuotes/quote_db.csv', sep=';', encoding='UTF-16', header=0, quoting=1)

    # Drop quotes identified as false positives
    inFile = inFile.dropna()
    words = set()
    wordEmbeddings = list()
    wordsWOembeddings = list()

    # Tokenize and lower-case all words
    for index, row in inFile.iterrows():
        quoteTokens = nltk.word_tokenize(row['quote'])
        for word in quoteTokens:
            words.add(word.lower())

    # Open the full word embedding library
    with open('../Resources/wordembeddings/daft.vec', 'r', encoding='utf-8') as modelFile:
        for i, line in enumerate(modelFile):
            # Separate words from their embeddings
            line = line.split(' ')
            word = line[0]
            vector = str(line[1:-1]).replace('[', '').replace(']', '').replace('\'', '')
            if word in words:
                wordEmbeddings.append([word, str(vector)])
            else:
                wordsWOembeddings.append(word)

    # Save subset of word embeddings, and print the number of used and unused word embeddings from the total word
    # embedding file
    pd.DataFrame(wordEmbeddings).to_csv('../resources/wordembeddings/filteredModel.csv', encoding='utf-8', index=None,
                                        sep=';', header=['word', 'vector'])
    print(wordEmbeddings.__len__(), 'words with embedding found\n',
          wordsWOembeddings.__len__(), 'words without embedding found')

# Generate a sub-file containing 20 % of all quotes in dataset as test set, and 80 % of all quotes in dataset as
# training set
def splitTrainingTestData(avgEmbeddings):
    # Open file containing all quote embeddings
    with open('../out/featurevecs.txt', 'r', encoding='utf-8') as inFile:
        # Generate vectors from file
        vectors = [x.replace('\n', '').split(', ') for x in inFile.readlines()]
        # Splitting feature vectors into training and test set, with 80 % in training and 20 % in test
        trainSet = vectors[:int(vectors.__len__() * 0.8)]
        testSet = vectors[int(vectors.__len__() * 0.8):]
        print('TrainSet size: {}\tTestSet size: {}'.format(trainSet.__len__(), testSet.__len__()))
        # Save training and test files, file format depending on whether quotes are to be generated as an average of
        # all word embeddings in a quote, or a matrix of word embeddings
        if avgEmbeddings:
            with open('../out/trainData.txt', 'w', encoding='utf-8') as outFile:
                for vector in trainSet:
                    outFile.write(str(vector).replace('[\'', '').replace(']\'', '').replace('\'', '') + '\n')
            with open('../out/testData.txt', 'w', encoding='utf-8') as outFile:
                for vector in testSet:
                    outFile.write(str(vector).replace('[\'', '').replace(']\'', '').replace('\'', '') + '\n')
        else:
            with open('../out/trainData.txt', 'w', encoding='utf-8') as outFile:
                for vector in trainSet:
                    outFile.write(str(vector) + '\n')
            with open('../out/testData.txt', 'w', encoding='utf-8') as outFile:
                for vector in testSet:
                    outFile.write(str(vector) + '\n')


# Preprocesses the politics subset, generating word embeddings for all quotes, and splits embeddings into test and
# training data. incPol indicates whether the politician context-based feature is to be included, incParty indicates
# whether the party context-based feature is to be included
def genPoliticsSubset(avgEmbeddings, incPol, incParty):
    preprocessQuotes('quote_db.csv', incPol=incPol, incParty=incParty, polSubset=True, avgEmbeddings=avgEmbeddings)
    splitTrainingTestData(avgEmbeddings)


# Preprocesses the full quote dataset, generating word embeddings for all quotes, and splits embeddings into test and
# training data. incPol indicates whether the politician context-based feature is to be included, incParty indicates
# whether the party context-based feature is to be included
def genFullDataset(avgEmbeddings, incPol, incParty):
    preprocessQuotes('quote_db.csv', incPol=incPol, incParty=incParty, polSubset=False, avgEmbeddings=avgEmbeddings)
    splitTrainingTestData(avgEmbeddings)


genFullDataset(True, True, True)