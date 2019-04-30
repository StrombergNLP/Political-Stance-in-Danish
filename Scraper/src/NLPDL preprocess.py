import gensim
import os
import json
import re
import sys
import getopt
import numpy as np
from nltk import word_tokenize

trainfile = "../data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json"
devfile = "../data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json"
testfile = "../data/semeval2017-task8-test-labels/subtaska.json"
datafolder = "../data/semeval2017-task8-dataset/rumoureval-data/"
testdatafolder = "../data/semeval2017-task8-test-data/"
devdatafolder = "../data/semeval2017-task8-dataset/rumoureval-data/germanwings-crash"
outputfolder = "../data/preprocess-output/"

# Negation words
negation_words = []
with open("../data/lexicon/negation-words.txt") as file:
    negation_words = [l.strip() for l in file.readlines()]  # note: all lowered

# Swear words
swear_words = []
with open("../data/lexicon/swear-words.txt") as file:
    swear_words = [l.strip().lower() for l in file.readlines()]


# Tweet class
class Tweet:
    def __init__(self, tweet_id, text, parent_id, url_count, has_image):
        self.tweet_id = tweet_id
        self.text = text
        self.parent_id = parent_id
        # 4 Attachments - url and image
        self.url_count = url_count
        self.has_image = has_image

    # 7 Tweet role
    def isSource(self):
        return self.parent_id is None

    def __repr__(self):
        if self.parent_id is None:
            self.parent_id = -1
        return "%d\t%s\t%d\t%s\t%s" % (self.tweet_id, self.text, self.parent_id, self.url_count, self.has_image)


def dfs(structure, node, path=None):
    """Traverse structure in dfs style and return all branches"""
    paths = []
    if path is None:
        path = []
    path.append(node)
    if structure[node] != []:
        for child in structure[node]:
            paths.extend(dfs(structure[node], child, path[:]))
    else:
        paths.append(path)
    return paths


def readConversation(folder):
    """Read the contents of a conversation folder
    and return (source, replies, structure)"""

    # Read source
    src_tweet_path = os.path.join(folder, "source-tweet")
    src_tweet = os.listdir(src_tweet_path)[0]  # There is only one
    source = readTweet(os.path.join(src_tweet_path, src_tweet))

    # #Read replies
    replies_path = os.path.join(folder, "replies")
    replies = dict()  # tweets
    if (os.path.exists(replies_path)):
        replies_files = os.listdir(replies_path)
        for file in replies_files:
            t = readTweet(os.path.join(replies_path, file))
            replies[t.tweet_id] = t

    # #Read structure
    structure_path = os.path.join(folder, "structure.json")
    with open(structure_path) as file:
        structure = json.load(file)  # parses to a dictionary
    src_id = list(structure.keys())[0]
    structure = dfs(structure, src_id)

    return (source, replies, structure)


def loadAndProcessData(folder, output_file, labels=None, test=False):
    threadcount = 0
    branchcount = 0
    tweetcount = 0
    nolabel = "UNK"
    label = ""
    conversations = sorted(os.listdir(folder))
    print("Processing conversations from", folder)

    with open(output_file, 'a') as out:
        # out.write("%s\n\n" % featureVectorHeader())
        for conversation in conversations:
            unique_ids = dict()
            threadcount += 1
            conversation_path = os.path.join(folder, conversation)
            (source, replies, structure) = readConversation(conversation_path)
            src_vec = tweetToVec(source, 0, [], None, replies, True)  # compute source once
            if (labels is not None):
                label = labels.get(str(source.tweet_id), nolabel)
                if (label == nolabel):
                    print("No labels for:", conversation)
            if (len(replies.items()) == 0):  # no replies exist
                out.write("{0}\t{1}\t{2}\n\n".format(source.tweet_id, label, src_vec))
                tweetcount += 1
                branchcount += 1
                continue
            if test:
                out.write("{0}\t{1}\t{2}\n".format(source.tweet_id, label, src_vec))
                tweetcount += 1
            for branch in structure:  # iterate branches in conversation
                branchcount += 1
                reply_vecs = []
                for i, tweet_id in enumerate(branch):  # go down each branch
                    if (i == 0):  # don't recompute source
                        continue
                    for rep_id, reply in replies.items():  # find the reply tweet
                        if (int(tweet_id) == rep_id):
                            vec = tweetToVec(reply, i, branch, source, replies)
                            if (vec == []):
                                print("Empty", rep_id)
                            else:
                                reply_vecs.append((tweet_id, vec))
                                if tweet_id not in unique_ids:
                                    unique_ids[tweet_id] = 0
                if (len(reply_vecs) > 0):
                    if not test:  # if test, dont repeat source tweet
                        out.write("{0}\t{1}\t{2}\n".format(source.tweet_id, label, src_vec))
                    for (rep_id, reply_vec) in reply_vecs:
                        if test:
                            unique_ids[rep_id] += 1
                            if unique_ids[rep_id] > 1:  # Skip repeating reply tweets
                                continue
                        if (labels is not None):
                            label = labels.get(rep_id, nolabel)
                            if (label == nolabel):
                                print("No labels for:", conversation)
                        out.write("{0}\t{1}\t{2}\n".format(rep_id, label, reply_vec))
                    out.write("\n")
            tweetcount += len(unique_ids)
    return (threadcount, branchcount, tweetcount)


def loadTestData(labels_file, datafolder, output_file):
    labels = loadLabels(labels_file)
    output_file = os.path.join(outputfolder, output_file)
    if os.path.exists(output_file):
        print("Remove", output_file)
        os.remove(output_file)
    (threadcount, branchcount, tweetcount) = loadAndProcessData(datafolder, output_file, labels, test=True)
    print("threads:{0} branches:{1} tweets:{2}".format(threadcount, branchcount, tweetcount))


def loadTrainingData(output_file):
    labels = loadLabels(trainfile)
    topics = sorted(os.listdir(datafolder))
    allthreadcount = 0
    allbranchcount = 0
    alltweetcount = 0
    output_file = os.path.join(outputfolder, output_file)
    if os.path.exists(output_file):
        print("Remove", output_file)
        os.remove(output_file)
    for topic in topics:
        if (topic == "germanwings-crash"):  # This is the development data
            continue
        topic_path = os.path.join(datafolder, topic)
        (threadcount, branchcount, tweetcount) = loadAndProcessData(topic_path, output_file, labels)
        allthreadcount += threadcount
        allbranchcount += branchcount
        alltweetcount += tweetcount
    print("threads:{0} branches:{1} tweets:{2}".format(allthreadcount, allbranchcount, alltweetcount))


def labeltoid(x):
    return {
        "support": 0,
        "deny": 1,
        "query": 2,
        "comment": 3
    }[x]


def loadLabels(path, shuffle=False):
    labels = dict()
    with open(path) as f:
        if not shuffle:
            for k, v in json.load(f).items():
                labels[k] = labeltoid(v)
        else:
            i = 0
            for k, _ in json.load(f).items():
                labels[k] = i
                i += 1
                if i == 4:
                    i = 0
    return labels


# reads all tweets in some directory
def readTweets(path):
    tweets = []
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".json") and not filename.startswith(
                    "structure") and not "dev" in filename and not "train" in filename:
                tweets.append(readTweet(os.path.join(dirpath, filename)))

    return tweets


# reads a tweet json file given a path
def readTweet(path, cleanTweetText=True):
    with open(path, "r") as file:
        tweet = json.load(file)
        hasImage = False
        if "media" in tweet["entities"]:
            hasImage = len(tweet["entities"]["media"]) > 0
        tweettext = tweet["text"]
        if cleanTweetText:
            if "media" in tweet["entities"]:
                for url in tweet["entities"]["media"]:
                    tweettext = tweettext.replace(url["url"], "pic")
            if "urls" in tweet["entities"]:
                for url in tweet["entities"]["urls"]:
                    tweettext = tweettext.replace(url["url"], "url")
        return Tweet(tweet["id"],
                     tweettext,
                     tweet["in_reply_to_status_id"],
                     len(tweet["entities"]["urls"]),
                     hasImage)


def preprocessTweet(tweet):
    # Remove non-alphabetic characters
    # regex = re.compile(r'([^\s\w]|_)+') #not (whitespace and character) or underscore
    # First parameter is the replacement, second parameter is input string
    tweet_alpha = re.sub("[^a-zA-Z]", " ", tweet.text)  # replace with space

    # Convert all words to lower case and tokenize
    return word_tokenize(tweet_alpha.lower())


# Feature extraction

# word2vec on Google News Dataset
# Note: The following requires at least 4 GB memory and possible also 64-architecture
model = lambda: None
load = False
EMB_N = 300


def loadVocabModel():
    print("Loading word2vec model...")
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '../data/word-embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    print("Done")


def avgWord2Vec(tweet_tokens):
    global model
    vec = np.zeros(EMB_N)  # word embedding
    # A tweet can max be 140 words (characters)
    # make up for varying lengths with zero-padding
    n = len(tweet_tokens)
    for w_i in range(n):
        word = tweet_tokens[w_i]
        if (word in model):
            vec += model[word]
    # Average word embeddings
    return vec / n


# Tweet lexicon
def countSpecialWords(tweet_tokens, lexicon):
    count = 0
    for t in tweet_tokens:  # assume all tokens are alphabetic words
        if t in lexicon:
            # for w in lexicon: #lookup special words
            #     if w == t:
            count += 1
    return count


# Punctuation
def Punctuation(text):
    # Period (.)
    period = '.' in text
    # Explamation mark (!)
    e_mark = '!' in text
    # Question mark(?)
    q_mark = '?' in text
    # Ratio of capital letters
    cap_count = sum(1 for c in text if c.isupper())
    cap_ratio = cap_count / len(text)
    return [int(period), int(e_mark), int(q_mark), float(cap_ratio)]


# Word2Vec cosine similarity wrt source, preceding, and thread tweet
def w2vCosineSimilarity(tweet_tokens, other):
    # Lookup words in w2c vocab
    tweet_words = []
    for token in tweet_tokens:
        if token in model.wv.vocab:  # check that the token exists
            tweet_words.append(token)
    other_words = []
    for token in other:
        if token in model.wv.vocab:
            other_words.append(token)

    if len(tweet_words) > 0 and len(other_words) > 0:  # make sure there is actually something to compare
        # cosine similarity between two sets of words
        return model.n_similarity(other_words, tweet_words)
    else:
        return 0.  # no similarity if one set contains 0 words


def ContentLength(text):
    # word and char count
    return [len(text.split()), len(text)]


def featureVectorHeader():
    header = "Tweet_ID\tlabel\t#., #!, #?, cap_ratio, #negation, #swear, URL, IMG, src, words, chars"
    if load:
        header += ", simToSrc, simToPrev, simToConv, wembs"
    return header


# Tweet to vector concatenation
def tweetToVec(tweet, i, branch, source, replies, isSource=False):
    # return [0] * 11 #empty feature vector for testing
    # Punctuation features [#., #!, #?, #cap_ratio]
    vec = Punctuation(tweet.text)

    # Tokenized, lower cased letters and only alphabetic chars
    tokens = preprocessTweet(tweet)

    # Special word counts
    vec.append(countSpecialWords(tokens, negation_words))
    vec.append(countSpecialWords(tokens, swear_words))

    # Presence of a URL
    vec.append(int(tweet.url_count > 0))
    # Presence of images
    vec.append(int(tweet.has_image))
    # whether the tweet is a source tweet or not
    vec.append(int(tweet.isSource()))
    # content length [word count, char count]
    vec.extend(ContentLength(tweet.text))

    if (load):
        allConvTokens = []
        for reply in replies.values():
            allConvTokens.extend(preprocessTweet(reply))

        simToSrc = 0
        simToPrev = 0
        simToConv = w2vCosineSimilarity(tokens, allConvTokens)

        if (not isSource):
            srcTweetTokens = preprocessTweet(source)
            prevTweetTokens = []
            if (i - 1) == 0:
                prevTweetTokens = srcTweetTokens
            else:
                for rep_id, reply in replies.items():  # find the reply tweet
                    if (int(branch[i - 1]) == rep_id):
                        prevTweetTokens = preprocessTweet(reply)
            simToPrev = w2vCosineSimilarity(tokens, prevTweetTokens)
            simToSrc = w2vCosineSimilarity(tokens, srcTweetTokens)

        vec.extend([simToSrc, simToPrev, simToConv])
        vec.extend(avgWord2Vec(tokens))

    return vec


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, "lht", ["loadw2v=", "help="])
    except getopt.GetoptError:
        print("see: preprocess.py -help")
        sys.exit(2)
    for opt, _ in opts:
        if opt in ("-l", "-loadw2v"):
            global load
            load = True
            loadVocabModel()
        elif opt in ("-h", "-help"):
            print("run 'preprocess.py -l' or 'preprocess.py -loadw2v' to load w2v model")
            sys.exit()
        elif opt in ("-t"):

            topics = sorted(os.listdir(datafolder))
            for topic in topics:
                tweets = readTweets(os.path.join(datafolder, topic))
                for i in range(10):
                    print(tweets[i])
                return

    # Run functions below
    loadTrainingData("train.txt")
    loadTestData(testfile, testdatafolder, "test.txt")
    loadTestData(devfile, devdatafolder, "dev.txt")


if __name__ == "__main__":
    main(sys.argv[1:])