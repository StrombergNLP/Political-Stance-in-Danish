import io
import math
import os
import re
from datetime import datetime

import pandas as pd
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage


# Mainly from https://media.readthedocs.org/pdf/pdfminer-docs/latest/pdfminer-docs.pdf
def parsePDF(fileLocation, politician, party, topic, useEmptyDB):
    data = ''
    quoteRelatedFillers = open('../Resources/quoteRelatedFillerWords.txt', 'r', encoding='utf-8').readline().split(',')
    nonArticleFlags = open('../Resources/nonArticleFlags.txt', 'r', encoding='utf-8').readline().split(',')
    quoteFillers = {'-', '»', '«'}
    wrongQuoteFlags = set()
    correctQuoteFlags = set()
    upcomingCorrectQuoteFlags = set()
    politicianLastName = politician.split(' ')[-1]
    for filler in quoteRelatedFillers:
        quoteFillers.update([', ' + filler + '.*' + politician + '.*', ', ' + filler + ' hun.*', ', ' + filler +
                             ' han.*', ', ' + filler + '.*' + politicianLastName + '.*'])
        # Statement made by someone other than the given politician
        wrongQuoteFlags.update([', ' + filler + ' (?!.*' + politician + '|.*hun|.*han|.*' + politicianLastName + ').*'])
        correctQuoteFlags.update([', ' + filler + ' .*' + politician + '.*',
                                  ', ' + filler + ' .*' + politicianLastName + '.*'])
        upcomingCorrectQuoteFlags.update([filler + '[ |,].*' + politician, politician + '[ |,].*' + filler,
                                          filler + '[ |,].*' + politicianLastName, politicianLastName + '[ |,].*' + filler])

    # Open a PDF file.
    fp = open(fileLocation, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data = retstr.getvalue()

    # Removing null bytes, generated by "ft" and "ff"
    data = data.replace('\0', '')
    newArticle = True
    quotes = []
    articleTitle, articleText, date = '', '', ''
    quoteCount, articleCount, articleID = 0, 0, 0

    if useEmptyDB:
        quoteDB = pd.read_csv('../out/empty_db/quote_db.csv', sep=';', encoding='UTF-8', header=0)
        articleDB = pd.read_csv('../out/empty_db/article_db.csv', sep=';', encoding='UTF-8', header=0)

    else:
        quoteDB = pd.read_csv('../out/quote_db.csv', sep=';', encoding='UTF-8', header=0)
        articleDB = pd.read_csv('../out/article_db.csv', sep=';', encoding='UTF-8', header=0)

    maxArticleID = articleDB['articleID'].max()
    articleID = maxArticleID + 1 if not math.isnan(maxArticleID) else 1
    maxQuoteID = quoteDB['quoteID'].max()
    quoteID = maxQuoteID + 1 if not math.isnan(maxQuoteID) else 1

    quoteDicts = []
    articleDicts = []

    correctUpcomingQuote = False
    quotesInALine = False

    for paragraph in data.split('\n\n'):
        paragraph = paragraph.replace('\n', ' ')

        # Save article publication dates
        if 'Id:' in paragraph:
            date = re.compile('\\w* \\d{2}, \\d{4}').search(paragraph).group()
            strpDate = datetime.strptime(date, '%B %d, %Y')
            date = strpDate.strftime('%m/%d/%Y')

        if any(flag in paragraph for flag in nonArticleFlags) or re.search('\\d+\\W\\d+\\W\\d{4}', paragraph) \
                or re.search('^\\d+/\\d+$', paragraph) or paragraph == 'København':
            continue

        # Identify and extract quotes
        test = False
        if paragraph.startswith('- ') or paragraph.startswith('»'):
            if paragraph.startswith('- Det vil jeg meget gerne vende tilbage til onsdag,'):
                test = True
            paragraph = paragraph.replace('«', '')

            # Ignore quote if not from politician in question
            wrongQuote = False
            correctQuote = False

            # Eksempel: Kirsten Normann, integration, Støjberg quote der ville inkluderes
            for wrongQuoteFlag in wrongQuoteFlags:
                if re.search(wrongQuoteFlag, paragraph):
                    wrongQuote = True
                    quotesInALine = False

            # Remove 'fillers' around quotes, such as ', siger Martin Henriksen' and strip whitespace
            if not wrongQuote:
                for correctQuoteFlag in correctQuoteFlags:
                    if re.search(correctQuoteFlag, paragraph):
                        correctQuote = True

                if correctQuote or correctUpcomingQuote or quotesInALine:
                    if test: print(correctQuote, correctUpcomingQuote, quotesInALine)
                    quotesInALine = True
                    for quoteFiller in quoteFillers:
                        paragraph = re.sub(quoteFiller, '', paragraph)
                    paragraph = paragraph.strip()
                    quotes.append(paragraph)
        # Catch multiple quotes in a row, in question-answer chain
        elif not paragraph.startswith('Spørgsmål: '):
            quotesInALine = False

        correctUpcomingQuote = False

        for upcomingCorrectQuoteFlag in upcomingCorrectQuoteFlags:
            if re.search(upcomingCorrectQuoteFlag, paragraph):
                if paragraph.startswith('Mette Frederiksen havde tirsdag ikke umiddelbart'): print(upcomingCorrectQuoteFlag, paragraph)
                correctUpcomingQuote = True

        if re.search(politician + '.*:$', paragraph.strip()):
            correctUpcomingQuote = True

        if newArticle and not re.search('\\d+\\W\\d+\\W\\d{4}', paragraph):
            articleTitle = paragraph
            newArticle = False
            continue

        # Construct article string from paragraphs, excluding non-article paragraphs generated during PDF extraction
        articleText += paragraph

        # End of article
        if paragraph.startswith('The client may distribute'):
            # Save quotes with info in quote database
            for quote in quotes:
                quoteDicts.append({'quoteID': quoteID, 'quote': quote, 'politician': politician,
                                   'date': date, 'party': party, 'articleID': articleID.__str__(), 'topic': topic,
                                   'fan': '', 'articleText': articleText})
                quoteCount = quoteCount + 1
                quoteID = quoteID + 1

            # Save article with articleID in article database
            articleDicts.append({'articleID': articleID, 'topic': topic, 'articleTitle': articleTitle, 'articleText': articleText,
                                 'mediaOutlet': 'ritzau'})

            newArticle = True
            quotes.clear()
            articleText = ''

            articleID = articleID + 1
            articleCount = articleCount + 1

    # Append newly parsed quotes and articles to exisiting database
    quoteDB = quoteDB.append(pd.DataFrame(quoteDicts), sort=False)
    articleDB = articleDB.append(pd.DataFrame(articleDicts), sort=False)

    # Remove quote and article duplicates
    quoteDB.drop_duplicates(subset=['quote', 'politician'], inplace=True)
    articleDB.drop_duplicates(subset=['articleText'], inplace=True)

    print('Quotes for politician:', quoteCount, '\nArticles for politician:', articleCount, '\nTotal quotes:',
          len(quoteDB.index), '\nTotal articles:', len(articleDB.index))

    # Save updated databases
    quoteDB.to_csv('../out/quote_db.csv', sep=';', encoding='UTF-8', index=False, quoting=1)
    articleDB.to_csv('../out/article_db.csv', sep=';', encoding='UTF-8', index=False, quoting=1)


def parseIntegration():
    for root, subdir, files in os.walk('../resources/ritzau/integration/'):
        party = root.split('../resources/ritzau/integration/')[1]
        if not party == '':
            for file in files:
                politician = file.split('_')[0]
                print('\'' + root + '/' + file + '\'', politician, party)
                parsePDF('' + root + '/' + file + '', politician, party, 'integration', False)


def parseMartinHenriksen():
    parsePDF('../resources/ritzau/integration/Dansk Folkeparti/Martin Henriksen_2018.pdf', 'Martin Henriksen',
             'Dansk Folkeparti', 'immigration', False)
    # parsePDF('../resources/ritzau/pdfParserTest.pdf', 'Martin Henriksen', 'Dansk Folkeparti')


def testParsing():
    parsePDF('../resources/ritzau/pdfParserTest.pdf', 'Martin Henriksen', 'Dansk Folkeparti', 'immigration', True)


def testForDuplication():
    parsePDF('../resources/ritzau/integration/Radikale Venstre/Morten Østergaard_2018.pdf', 'Morten Østergaard',
             'Radikale Venstre', 'immigration', True)
    #parsePDF('../resources/ritzau/integration/Socialistisk Folkeparti/Kirsten Normann_2018.pdf', 'Kirsten Normann',
    #         'Socialistisk Folkeparti', 'immigration', False)


# testForDuplication()

# testParsing()

# parseMartinHenriksen()

parseIntegration()