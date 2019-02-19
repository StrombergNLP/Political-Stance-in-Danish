from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io
import re
import csv
from datetime import datetime
import os
import pandas as pd

# Mainly from https://media.readthedocs.org/pdf/pdfminer-docs/latest/pdfminer-docs.pdf
def parsePDF(fileName, politician):
    data = ''
    quoteFillers = {', siger '+politician, ', mener '+politician, '- ', 'siger hun', 'siger han'}
    nonArticleFlags = {'All material stored', 'The client', 'Internal redistribution', 'Media Archive',
                       'https://apps.infomedia.dk/mediearkiv', '/ritzau/', 'Id: '}

    # Open a PDF file.
    fp = open('../resources/'+fileName, 'rb')
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

    newArticle = True
    quotes = []
    articleTitle, article, date = '', '', ''
    quoteCount, articleCount, articleID = 0, 0, 0

    with pd.read_csv('../out/quote_db.csv', sep=';', encoding='UTF-8',
                     names=['quote', 'politician', 'article title', 'date', 'articleID']) as qdb, \
            pd.read_csv('../out/article_db.csv', sep=';', encoding='UTF-8', names=['articleID', 'articleText']) as adb:

    #with open('../out/quote_db.csv', 'a', encoding='UTF-8', newline='') as qdb, \
    #        open('../out/article_db.csv', 'a', encoding='UTF-8', newline='') as adb:
#
    #    if not os.stat('../out/article_db.csv').st_size > 1:
    #        reader = csv.reader(adb)
    #        articleID = max(int(column[0]) for column in reader)

        #quoteWriter = csv.writer(qdb, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        #quoteWriter.writerow(['quote', 'politician', 'article title', 'date', 'articleID'])
#
        #articleWriter = csv.writer(adb, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        #articleWriter.writerow(['articleID', 'articleText'])
        #qdf = pd.DataFrame.from_csv([qdb])
        #adf = pd.DataFrame.from_csv([adb])
        #articleID = adf['articleID'].max() + 1

        for paragraph in data.split('\n\n'):
            paragraph = paragraph.replace('\n', ' ')

            if newArticle is True and not paragraph.startswith('Media Archive') \
                    and not re.search('\d/\d{2}/\d{4}', paragraph):
                articleTitle = paragraph
                newArticle = False

            # Save article publication dates
            elif 'Id:' in paragraph:
                date = re.compile('\w* \d{2}, \d{4}').search(paragraph).group()
                strpDate = datetime.strptime(date, '%B %d, %Y')
                date = strpDate.strftime('%m/%d/%Y')

            # Construct article string from paragraphs, excluding non-article paragraphs generated during PDF extraction
            elif not any(flag in paragraph for flag in nonArticleFlags) and not re.search('\d/\d{2}/\d{4}', paragraph) \
                    and not re.search('^\d+/\d+$', paragraph) and not paragraph == 'København':
                article += paragraph

            # Identify and extract quotes
            if paragraph.startswith('- '):
                # Remove 'fillers' around quotes, such as ', siger Martin Henriksen' and strip whitespace
                for quoteFiller in quoteFillers:
                    paragraph = paragraph.replace(quoteFiller, '')
                    paragraph = paragraph.strip()
                quotes.append(paragraph)

            # End of article
            if paragraph.startswith('The client may distribute'):
                # Save quotes with info in quote database
                for quote in quotes:
                    quoteWriter.writerow([quote, politician, articleTitle, date, articleID.__str__()])
                    #qdf.append([quote, politician, articleTitle, date, articleID.__str__()])
                    quoteCount = quoteCount+1

                # Save article with articleID in article database
                print(article)
                articleWriter.writerow([articleID, article])
                #adf.append([articleID, article])

                newArticle = True
                quotes.clear()
                article = ''

                articleID = articleID + 1
                articleCount = articleCount + 1

        print(quoteCount.__str__())
        #adf.to_csv('../out/article_db.csv')
        #qdf.to_csv('../out/quote_db.csv')

def parseMartinHenriksen():
    # parsePDF('im_mh_int_2018.pdf', 'Martin Henriksen')
    parsePDF('pdfParserTest.pdf', 'Martin Henriksen')

parseMartinHenriksen()



