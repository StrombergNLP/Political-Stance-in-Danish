import pandas as pd


def cleanFalsePositiveArticles():
    quoteDB = pd.read_csv('../out/filled_integration_db/quote_db.csv', sep=';', encoding='UTF-16', header=0, quoting=1)
    articleDB = pd.read_csv('../out/filled_integration_db/article_db.csv', sep=';', encoding='UTF-16', header=0, quoting=1)
    articleDB.drop('topic', axis=1, inplace=True)
    collectedDB = pd.merge(quoteDB, articleDB, how='inner', on='articleID', left_index=True)
    print(collectedDB)
    collectedDB = collectedDB[collectedDB.falsePositive != 1]
    print(collectedDB)
    collectedDB.drop(['articleTitle', 'articleText', 'falsePositive', 'mediaOutlet'], axis=1, inplace=True)
    collectedDB.to_csv('../out/cleanedQuoteDB.csv', sep=';', encoding='UTF-16', index=False, quoting=1)


def removeArticlesWithoutQuotes(subfolder):
    quoteDB = pd.read_csv('../out/' + subfolder + 'quote_db.csv', sep=';', encoding='UTF-16', header=0, quoting=1)
    articleDB = pd.read_csv('../out/' + subfolder + 'article_db.csv', sep=';', encoding='UTF-16', header=0, quoting=1)
    articleDB = articleDB[articleDB['articleID'].isin(quoteDB['articleID'])]
    print(articleDB)
    articleDB.to_csv('../out/' + subfolder + 'clean_article_db.csv', sep=';', encoding='UTF-16', quoting=1, index=0)


#removeArticlesWithoutQuotes('')

cleanFalsePositiveArticles()
