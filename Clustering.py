# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 22:27:50 2018

@author: Faishal
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

#input to dataframe
outfile = "D:\\File\\"
dataset = pd.DataFrame.from_csv(outfile + 'ds_asg_data.csv')

#delete row NA
dataset = dataset.dropna()
dataset.isnull().sum()

#List of topic
topic = dataset.groupby('article_topic').size()

#Split text and target
sentence = dataset['article_content'] #text
y = dataset['article_topic'] #target

#preprocessing by regex
sentence = sentence.str.lower()
sentence = sentence.str.replace(r"[^a-zA-Z0-9]+"," ")
sentence = sentence.str.replace(r"([^\w])"," ") 
sentence = sentence.str.replace(r"\b\d+\b", " ")
sentence = sentence.str.replace(r"\s+|\r|\n", " ")
sentence = sentence.str.replace(r"^\s+|\s$", "")

#stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#stopword
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# stemming and stopword process
X = []
index = 1

for item in sentence:
    print('data nomor: {}'.format(index))
    item = stemmer.stem(item)
    item = stopword.remove(item)

    X.append(item)
    index = index + 1
    
X = pd.Series(X)

#Data Source
text = X

#if not using stemming and stopword
text = sentence #text

# Using TFIDF vectorizer to convert words to Vector Space
tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                   use_idf=True,
                                   min_df=0.24, max_df=0.85,
                                   ngram_range=(1, 2))

# Fit the vectorizer to text data
tfidf_matrix = tfidf_vectorizer.fit_transform(text)
terms = tfidf_vectorizer.get_feature_names()
# print(terms)

# Kmeans++
km = KMeans(n_clusters=29, init='k-means++', max_iter=300, n_init=1, verbose=0, random_state=3425)
km.fit(tfidf_matrix)
labels = km.labels_
clusters = labels.tolist()

# Calculating the distance measure derived from cosine similarity
distance = 1 - cosine_similarity(tfidf_matrix)

# Dimensionality reduction using Multidimensional scaling (MDS)
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(distance)
xs, ys = pos[:, 0], pos[:, 1]

# Saving cluster visualization after mutidimensional scaling
for x, y, in zip(xs, ys):
    plt.scatter(x, y)

    # Creating dataframe containing reduced dimensions, identified labels and text data for plotting KMeans output
result = pd.DataFrame(dict(label=clusters, data=text, x=xs, y=ys))
topic.to_csv(os.path.join(outfile, 'kmeans_clustered_DFN.csv'), sep=';')

#List of cluster
listcluster = result.groupby('label').size()
