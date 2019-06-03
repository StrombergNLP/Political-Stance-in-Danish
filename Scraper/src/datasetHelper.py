import pandas as pd


# Removes quotes from quote dataset where articles have been identified as false positives
def cleanFalsePositiveArticles():
    # Read article and quote datasets
    quoteDB = pd.read_csv('../out/quote_db.csv', sep=';', encoding='UTF-16', header=0, quoting=1)
    articleDB = pd.read_csv('../out/article_db.csv', sep=';', encoding='UTF-16', header=0, quoting=1)

    # Drop columns and merge datasets
    articleDB.drop('topic', axis=1, inplace=True)
    collectedDB = pd.merge(quoteDB, articleDB, how='inner', on='articleID', left_index=True)

    # Drop quotes where the article has been identified as false positive
    collectedDB = collectedDB[collectedDB.falsePositive != 1]

    # Drop columns native to the article dataset
    collectedDB.drop(['articleTitle', 'articleText', 'falsePositive', 'mediaOutlet'], axis=1, inplace=True)

    # Save updated dataset as quote dataset
    collectedDB.to_csv('../out/cleanedQuoteDB.csv', sep=';', encoding='UTF-16', index=False, quoting=1)


# Removes articles from article dataset where no quotes are present in the quote dataset for the article
def removeArticlesWithoutQuotes():
    # Read article and quote datasets
    quoteDB = pd.read_csv('../out/quote_db.csv', sep=';', encoding='UTF-16', header=0, quoting=1)
    articleDB = pd.read_csv('../out/article_db.csv', sep=';', encoding='UTF-16', header=0, quoting=1)

    # Remove articles where article ID is not present in quote dataset
    articleDB = articleDB[articleDB['articleID'].isin(quoteDB['articleID'])]

    # Save updated article dataset
    articleDB.to_csv('../out/cleanedArticleDB.csv', sep=';', encoding='UTF-16', quoting=1, index=0)
