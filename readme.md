# Thesis project
Thesis project by Rasmus Lehmann, written at the IT University of Copenhagen, 2019

The repository is organized as follows:

##Models/
Models/ contains all code and resources required for running the stance detection models found in Models/src. This
sub-folder contains the following model files:
- ConditionalLSTM.py
- QuoteLSTM.py
- sklearnClassifiers.py

Models/out is where benchmark files will be generated, when benchmark methods are called in one of the three model files listed above.

Models/resources contains two sub-folders with data files. The model ConditionalLSTM.py makes use of data files from Models/resources/quote2vec and QuoteLSTM.py and sklearnClassifiers.py make use of data files from Models/resources/avgQuote2Vec.

Both of the data folders contain additional sub-folders, one containing data for the full dataset generated in relation to this project, named /fullDataset, one containing data for a subset of the full dataset, named /nationalPolicy.

Within Models/resource/avgQuote2Vec/fullDataset is found additional versions of the full dataset, applied for testing. The subfolder /nocontext contains no context-based features whatsoever, /noPartyVec contains only context-based features for the quoted politician and /noPolVec contains only context-based features for the quoted politician's party affiliation.

All data folders contain two documents; trainData.txt used for training the models listed above and testData.txt used for testing the models after they are trained.

##Scraper/
Scraper/ contains all code and resources required for running quote extraction and automated data cleaning on data collected from the Infomedia media archive from the media outlet Ritzau. New data can be downloaded and placed in the folder Scraper\resources\ritzau, from where it can be added to the dataset using the scripts in Scraper\src.

!OBS! All article files are removed from the public version of this repo due to copyright

Scraper/out/ contains the cleaned article and quote datasets in cleadArticleDB.csv and cleadQuoteDB.csv. article_db.csv and quote_db.csv have the same information, but formatted to function best with the scripts in Scraper/src. featurevecs.txt contains the quote dataset converted into feature vectors using the prepreocess.py file in Scraper/src. featurevect.txt is used for generating the test and train subsets using preprocess.py.

Scraper/resources/ contains fanMapping.txt, which maps features in letter format to number format. nonArticleFlags.txt contains text pieces indicating that a given text is not part of an article. quoteRelatedFillerWords.txt contains a list of filler words used to identify quotes in ritzauPdfScraper.py found in Scraper/src/.

The subfolder Scaper/resources/mapping/ contains two mapping documents, mapping politicians' names to an index and political parties to an index.

Scraper/empty_db/ contains empty dataset files, used in ritzauPdfScraper.py located at Scraper/src/.

Scraper/resources/mapping/ contains the raw quotes scraped from pdf files using ritzauPdfScraper.py.

Scraper/resources/ritzau/ contains the article files from which quotes are scraped, subdivided by topic, political party and finally politician.

Scraper/resources/wordEmbeddings/ contains the file daft.vec, which holds a library fastText word embeddings, and filteredModel.csv contains a subset the same library, holding only the words present in a vocabulary generated from the words present in the quotes in the quote dataset.

Scraper/src/ holds the three script files used for preprocessing and quote extraction:
- datasetHelper.py
- preprocess.py
- ritzauPdfScraper.py

## Setup
Scripts for this project were built using Python 3.7. Running the scripts in this repository requires the following Python libraries:
- Pandas
- PDFMiner
- NLTK
- Numpy
- Sklearn
- Torch

## Running Model scripts
Model scripts are run from the Models\src\ sub-folder.

# ConditionalLSTM.py
The model is used by running one of the following functions:
- runFullBenchmark
- runSpecificBenchmark

runFullBenchmark generates a model for each hyperparameter combination defined in the script, trains it, and tests its performance, outputting a benchmark file in Models\out\. This model is relatively heavy, and it is therefore recommended to use a small hyperparameter space, or simply run one hyperparameter combination at a time using the runSpecificBenchmark function.

runSpecificBenchmark runs a benchmark for a given hyperparameter combination, and outputs the results in an out-file of the user's choice. The model takes the following arguments, in order:
- path: The path in Models\resources\quote2vec from which data should be used, the paths 'fullDataPath' and 'polSubsetPath' are defined in the script, and can be used as arguments
- LSTMLayers: The number of desired layers in the generated LSTM model
- LSTMDims: The number of desired nodes in each of the layers of the generated LSTM model
- ReLULayers: The number of desired linear layers with ReLU activation following the LSTM layers of the generated model
- ReLUDims: The number of desired nodes in each of the generated ReLU layers
- L2: The desired strength of L2 regularization
- fullRun: Whether or not the full hyperparameter space is currently being searched. If the model is called manually, this should always be 'False'
- outFile: An open out-file in csv format, to which results will be written

# QuoteLSTM.py
The model is used by running one of the following functions:
- runFullBenchmark
- runSpecificBenchmark

runFullBenchmark generates a model for each hyperparameter combination defined in the script, trains it, and tests its performance, outputting a benchmark file in Models\out\. This model is relatively heavy, and it is therefore recommended to use a small hyperparameter space, or simply run one hyperparameter combination at a time using the runSpecificBenchmark function. The model takes a single boolean argument, determining whether the model should be bi-directional.

runSpecificBenchmark runs a benchmark for a given hyperparameter combination, and outputs the results in an out-file of the user's choice. The model takes the following arguments, in order:
- path: The path in Models\resources\avgQuote2vec from which data should be used, the paths 'fullDataPath' and 'polSubsetPath' are defined in the script, and can be used as arguments
- LSTMLayers: The number of desired layers in the generated LSTM model
- LSTMDims: The number of desired nodes in each of the layers of the generated LSTM model
- ReLULayers: The number of desired linear layers with ReLU activation following the LSTM layers of the generated model
- ReLUDims: The number of desired nodes in each of the generated ReLU layers
- L2: The desired strength of L2 regularization
- fullRun: Whether or not the full hyperparameter space is currently being searched. If the model is called manually, this should always be 'False'
- outFile: An open out-file in csv format, to which results will be written
- biDirectional: Whether the generated model should be bi-directional

# sklearnClassifiers.py
The model is used by running the 'run' function, which takes two parameters:
- path: The path in Models\resources\avgQuote2vec from which data should be used, the paths 'fullDataPath' and 'polSubsetPath' are defined in the script, and can be used as arguments
- classifierType: Either 'randomForest' if a random forest classifier is desired, or 'GNB' if a Gaussian Naïve Bayes classifier is desired. Any other input will result in an error message.

## Running Scraper scripts
Scraper scripts are run from the Scraper\src\ sub-folder.

# datasetHelper.py
The script is used by running one of the two following functions:
- cleanFalsePositiveArticles
- removeArticlesWithoutQuotes

cleanFalsePositiveArticles removes quotes from articles in the article_db.csv file that have been identified as false positives from the quote_db.csv file and saves them to the cleanedQuoteDB.csv, all three files located in the Scraper\out\ sub-folder.

removeArticlesWithoutQuotes removes articles from the article database for which there are no quotes present in the quote database, the two files quote_db.csv, article_db.csv and cleanedArticleDB.csv all found in the Scraper\out\ sub-folder.

# preprocess.py
The script is generally used by running one of the three following functions:
- genPoliticsSubset
- genFullDataset
- generateModelSubset
Six other functions exist within the script, primarily for internal use, but can be of use in generating custom sub-sets for use in testing.

genFullDataset creates feature vectors for the use in the models located at Models\src\, using the full quote dataset located at Scraper\resources\parsedQuotes\, after which the feature vectors are split into a test and a training set. The function takes three parameters:
- avgEmbeddings: Whether embeddings are to be created as an average embedding across a full quote, or as a matrix of word embeddings
- incPol: Whether the politician one-hot vector context-based feature should be included
- incParty: Whether the party one-hot vector context-based feature should be included

genPoliticsSubset does the same thing as genFullDataset, but only includes quotes from the subset 'National policy'
- avgEmbeddings: Whether embeddings are to be created as an average embedding across a full quote, or as a matrix of word embeddings
- incPol: Whether the politician one-hot vector context-based feature should be included
- incParty: Whether the party one-hot vector context-based feature should be included

generateModelSubset extracts a subset of the word embedding library based on the vocabulary within the quote dataset, and saves that subset

# ritzauPdfScraper.py
The script is used by running one of the two following functions:
- parsePDF
- parseIntegration

parsePDF parses a PDF at a given file location, and extracts quotes and raw article text, adding them to the quote_db.csv and article_db.csv in Scraper\out\. The method removes non-article text using nonArticleFlags from Scraper\resources\, and removes filler words for quotes using quoteRelatedFillerWords.txt at the same location. The method takes five parameters:
- fileLocation: The file location of the PDF file from which quotes are to be scraped.
- politician: The politician the script is supposed to look for quotes for
- party: The party the passed politician is a part of
- topic: The topic of the quotes
- useEmptyDB: Whether or not an empty dataset file should be used. If so, they are read from Scraper\resources\empty_db\. If not, new quotes and articles are added to the already existing article_db.csv abd quote_db.csv files in Scraper\resources\

parseIntegration runs the parsePDF function for each of the files in the Scraper\resources\ritzau\intagration\ sub-folder
