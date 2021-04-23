from __future__ import unicode_literals, print_function


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, pprint, string

string.punctuation
limit = 0
import subprocess
import unicodedata, json, csv, random, time, sys
import datetime, pickle, copy
import warnings

# To perform LDA, we'll use gensim.
warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")

from pandas import DataFrame, concat
from joblib import Parallel, delayed
from string import punctuation
from collections import Counter
from pathlib import Path

from tqdm import tqdm  # loading bar
from tqdm.notebook import tqdm

tqdm.pandas(desc="progress-bar")
from tqdm import tqdm, tqdm_notebook
from tqdm import tqdm_notebook as tqdm
from tqdm.auto import tqdm

import nltk

nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk, pos_tag_sents

nltk.download("averaged_perceptron_tagger")
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("stopwords")
nltk.download("wordnet")

from itertools import groupby
from collections import OrderedDict
from nltk.chunk import tree2conlltags

from functools import reduce

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from IPython.display import display

# Enable logging for gensim - optional
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR
)


pd.options.display.max_columns = 20


def news_articles():
    df = pd.read_csv(
        "articles.csv",
        usecols=["date", "title", "author", "content"],
        engine="python",
        error_bad_lines=False,
    )
    df.drop_duplicates("content")  # Remove duplicate from "content" columns
    df = df[~df["content"].isnull()]  # Remove rows with empty cotent
    df.dropna(inplace=True)
    return df


df = news_articles()
# print(df.head(3))  # sanity check


def initial_process(df):
    data = df.copy()
    data = data[~data["content"].isnull()]  # Remove rows with empty cotent

    # Select contents of length between 140 and 2000 characters.
    data = data[(data.content.map(len) > 140) & (data.content.map(len) <= 2000)]
    data.reset_index(inplace=True, drop=True)

    # collecting 10000 random sample for computational ease
    data = data.sample(10000, random_state=2021)
    data.reset_index(inplace=True, drop=True)

    # aligning the columns in order of requirement
    data = data[["date", "title", "author", "content"]]
    # renaming date -> date_of_news
    data.rename(columns={"date": "date_of_news"}, inplace=True)
    return data


data = initial_process(df)
# print(data.tail(3))  # sanity check

# PREPROCESSING
def preProcess(text):
    #  Remove non-ASCII characters
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
        .lower()
    )  # lower case lowers the sparsity of the data

    # remove numbers
    text = re.sub(r"\d+", "", str(text))

    # removing salutaions (if any)
    text = re.sub("Mr\.", "Mr", str(text))
    text = re.sub("Mrs\.", "Mrs", str(text))

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"won\'t", "will not", text)
    text = text.replace("(ap)", "")
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"(\w+)\'s", "\g<1> is", text)
    text = re.sub(r"(\w+)\'ve", "\g<1> have", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)

    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # removing any reference to outside text
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    text = text.strip()
    return text


# preprocessing texts
data["processedContent"] = data["content"].apply(preProcess)
# print(data.head(3))

num_of_rare_words = 25
RARE_WORDS = set(
    [w for (w, wc) in Counter().most_common()[: -num_of_rare_words - 1 : -1]]
)


def remove_rare_words(text):
    return " ".join([word for word in str(text).split() if word not in RARE_WORDS])


data["processedContent"] = data["processedContent"].apply(
    lambda text: remove_rare_words(text)
)
# print(data.head(3))

# REMOVE FREQUENT WORDS
FREQ_WORDS = set([w for (w, wc) in Counter().most_common(25)])


def remove_freq_words(text):
    return " ".join([word for word in str(text).split() if word not in FREQ_WORDS])


data["processedContent"] = data["processedContent"].apply(
    lambda text: remove_freq_words(text)
)
# print(data.head(3))

# STOPWORDS REMOVAL
with open("stopwords.json") as json_file:
    addStopwords = json.load(json_file)
    # print(addStopwords); print(len(addStopwords['en'])

add_stopwords = set(addStopwords["en"])
stop_words = set(stopwords.words("english"))

# add words that aren't in the NLTK stopwords list
STOPWORDS = stop_words.union(add_stopwords)
STOPWORDS = list(STOPWORDS)
# print(STOPWORDS)
# print()
# print(len(STOPWORDS))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


data["processedContent"] = data["processedContent"].apply(
    lambda text: remove_stopwords(text)
)
# print(data.head(2))

# LEMATIZE
wnl = WordNetLemmatizer()


def lemmatize_words(text):
    return " ".join([wnl.lemmatize(word) for word in text.split()])


data["processedContent"] = data["processedContent"].apply(
    lambda text: lemmatize_words(text)
)
# print(data.head(2))

# REMOVE PUNCTUATION
PUNCT_REMOVE = string.punctuation


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", PUNCT_REMOVE))


data["processedContent"] = data["processedContent"].apply(
    lambda text: remove_punctuation(text)
)
# print(data.head(3))

# tokenizer to 'processedContent' column through all rows and store in 'tokens' column.
data["tokens"] = data["processedContent"].apply(word_tokenize)

#  POS tagging of the sentence and ne_chunk() to recognize each named entity in sentences
data["ner"] = data["processedContent"].apply(
    lambda x: nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(x)), binary=True)
)
data.tail(3)

# CSV to JSON
xDf = data[["date_of_news", "title", "author", "ner"]].copy()
xDf.to_csv("newDf.csv")


def csv_to_json(in_file_csv, out_file_json):
    json_array = []

    # read csv file
    with open(in_file_csv, "r", encoding="utf-8") as csv_file:
        # load csv file data using csv library's dictionary reader
        csv_reader = csv.DictReader(csv_file)

        # convert each csv row into python dict
        for row in csv_reader:
            # add the abobe python dict to json array
            json_array.append(row)

    # convert python jsonArray to JSON String and write to file
    with open(out_file_json, "w") as json_file:
        json_string = json.dumps(json_array, indent=4)
        json_file.write(json_string)


in_file_csv = r"newDf.csv"
out_file_json = r"newDf.json"

csv_to_json(in_file_csv, out_file_json)

# TOPIC MODEL
yDf = data[["date_of_news", "title", "author", "tokens"]].copy()

bigram = gensim.models.Phrases(yDf["tokens"], min_count=5, threshold=100)
bigramModel = gensim.models.phrases.Phraser(bigram)

yDf["bigramTokens"] = yDf["tokens"].apply(lambda tokens: bigramModel[tokens])

# Creating Dictionary
id2word = corpora.Dictionary(yDf["bigramTokens"])

# Creating Corpus
texts = yDf["bigramTokens"].tolist()
dictionary = Dictionary(texts)

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Define LDA model in function that takes the number of topics as a parameter.
def LDAmodel(num_topics, passes=1):
    return gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        leave=False,
        id2word=id2word,
        alpha="auto",
        eta="auto",
        num_topics=num_topics,  # the number of topics is equal to num_topics
        random_state=2021,
        eval_every=1,
        chunksize=2000,
        passes=passes,
        per_word_topics=True,
    )


def coherence_computation(model):
    coherence = CoherenceModel(
        model=model, texts=texts, dictionary=id2word, coherence="c_v"
    )
    return coherence.get_coherence()


def display_topics(model):
    topics = model.show_topics(
        num_topics=model.num_topics, formatted=False, num_words=10
    )
    topics = map(lambda c: map(lambda cc: cc[0], c[1]), topics)
    DATA = DataFrame(topics)
    DATA.index = ["topic_{0}".format(i) for i in range(model.num_topics)]
    DATA.columns = ["keyword_{0}".format(i) for i in range(1, 10 + 1)]
    return DATA


def explore_models(DATA, range=range(5, 25)):
    id2word = corpora.Dictionary(DATA["bigramTokens"])
    texts = DATA["bigramTokens"].tolist()
    corpus = [id2word.doc2bow(text) for text in texts]

    coherence_values = []
    model_list = []

    for num_topics in tqdm_notebook(range, leave=False):
        lda = LDAmodel(num_topics, passes=5)
        model_list.append(lda)
        coherence = coherence_computation(lda)
        coherence_values.append(coherence)

    fig = plt.figure(figsize=(10, 5))
    plt.title("Optimal number of topics")
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence Score")
    plt.grid(True)
    plt.plot(range, coherence_values)

    return coherence_values, model_list


coherence_values, model_list = explore_models(yDf, range=range(5, 40, 5))

# Print the coherence scores
limit = 40
start = 5
step = 6
x = range(start, limit, step)

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

bestModel = LDAmodel(num_topics=10, passes=5)
# print(display_topics(model = bestModel))

# Compute Coherence Score using c_v
coherence_model_lda = CoherenceModel(
    model=bestModel, texts=texts, dictionary=dictionary, coherence="c_v"
)
coherence_lda = coherence_model_lda.get_coherence()
print("\nc_v Coherence Score: ", coherence_lda)

# Compute Coherence Score using UMass
coherence_model_lda = CoherenceModel(
    model=bestModel, texts=texts, dictionary=dictionary, coherence="u_mass"
)
coherence_lda = coherence_model_lda.get_coherence()
print("\nUMass Coherence Score: ", coherence_lda)
