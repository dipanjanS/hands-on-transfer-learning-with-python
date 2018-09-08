
# coding: utf-8

# # Feature Engineering
# Textual Data
# Important Imports

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Prepare a Sample Corpus

corpus = ['pack my box with five dozen liquor jugs.',
 'pack my box',
 'the quick brown fox jumps over the lazy dog.',
 'the brown fox is quick and the blue dog is lazy',
 'pack my box with five dozen liquor jugs and biscuits',
 'the dog is lazy but the brown fox is quick']

labels = ['picnic', 'picnic', 'animals', 'animals', 'picnic', 'animals']
corpus = np.array(corpus)
corpus_df = pd.DataFrame({'document': corpus, 
                          'category': labels})
corpus_df = corpus_df[['document', 'category']]
corpus_df


# Bag of Words

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(corpus_df.document)
cv_matrix = cv_matrix.toarray()
cv_matrix


# TF-IDF 

tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(corpus_df.document)
tv_matrix = tv_matrix.toarray()

vocab = tv.get_feature_names()
pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)


# N-Gram Vectorizer

bv = CountVectorizer(ngram_range=(2,2))
bv_matrix = bv.fit_transform(corpus_df.document)
bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names()
pd.DataFrame(bv_matrix, columns=vocab)

