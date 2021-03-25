from sklearn.datasets import fetch_20newsgroups
from gensim.models import word2vec
from bs4 import BeautifulSoup
import re
import nltk
import time
import ssl

start = time.time()
print(start)

ssl._create_default_https_context = ssl._create_unverified_context
news = fetch_20newsgroups(subset='all')
print(len(news))

X,y = news.data, news.target

def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('totokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
    return sentences

sentences = []
for x in X:
    sentences.extend(news_to_sentences(x))

num_features = 30
min_word_count = 20
num_workers = 2
context = 5
downsampling = 1e-3

model = word2vec.Word2Vec(sentences, workers= num_workers, size= num_features, min_count=min_word_count, window=context, sample=downsampling)

model.init_sims(replace=True)

# model.save('./output/')
