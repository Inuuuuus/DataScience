from sklearn.datasets import fetch_20newsgroups
from gensim.models import word2vec
from bs4 import BeautifulSoup
import re
import nltk
import time

start = time.time()
print(start)

news = fetch_20newsgroups(subset='all')
print(len(news))