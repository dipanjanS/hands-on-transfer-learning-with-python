# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:45:59 2018

@author: tghosh
"""
import config 
import numpy as np
import os

class GloVe:    
    
    def __init__(self, embd_dim=50):
        if embd_dim not in [50, 100, 200, 300]:
            raise ValueError('embedding dim should be one of [50, 100, 200, 300]')
        self.EMBEDDING_DIM = embd_dim
        self.embedding_matrix = None
        
    def _load(self):
        print('Reading {} dim GloVe vectors'.format(self.EMBEDDING_DIM))
        self.embeddings_index = {}
        with open(os.path.join(config.GLOVE_DIR, 'glove.6B.'+str(self.EMBEDDING_DIM)+'d.txt'),encoding="utf8") as fin:
            for line in fin:
                try:
                    values = line.split()
                    coefs = np.asarray(values[1:], dtype='float32')
                    word = values[0]
                    self.embeddings_index[word] = coefs
                except:
                    print(line)

        print('Found %s word vectors.' % len(self.embeddings_index))
        
    def _init_embedding_matrix(self, word_index_dict, oov_words_file='OOV-Words.txt'):
        self.embedding_matrix = np.zeros((len(word_index_dict)+2 , self.EMBEDDING_DIM)) # +1 for the 0 word index from paddings.
        not_found_words=0
        missing_word_index = []
        
        with open(oov_words_file, 'w') as f: 
            for word, i in word_index_dict.items():
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    self.embedding_matrix[i] = embedding_vector
                else:
                    not_found_words+=1
                    f.write(word + ','+str(i)+'\n')
                    missing_word_index.append(i)

            #oov by average vector:
            self.embedding_matrix[1] = np.mean(self.embedding_matrix, axis=0)
            for indx in missing_word_index:
                self.embedding_matrix[indx] = np.random.rand(self.EMBEDDING_DIM)+ self.embedding_matrix[1]
        print("words not found in embeddings: {}".format(not_found_words))
        
        
    def get_embedding(self, word_index_dict):
        if self.embedding_matrix is None:
            self._load()
            self._init_embedding_matrix(word_index_dict)
        return self.embedding_matrix
    
    def update_embeddings(self, word_index_dict, other_embedding, other_word_index):
        num_updated = 0
        for word, i in other_word_index.items():
            if word_index_dict.get(word) is not None:
                embedding_vector = other_embedding[i]
                this_vocab_word_indx = word_index_dict.get(word)
                #print("BEFORE", self.embedding_matrix[this_vocab_word_indx])
                self.embedding_matrix[this_vocab_word_indx] = embedding_vector                
                #print("AFTER", self.embedding_matrix[this_vocab_word_indx])
                num_updated+=1
        
        print('{} words are updated out of {}'.format(num_updated, len(word_index_dict)))

class Word2Vec(GloVe):
    def __init__(self, embd_dim=50):
        super().__init__(embd_dim=embd_dim)
        
    def _load(self):
        print('Reading {} dim Gensim Word2Vec vectors'.format(self.EMBEDDING_DIM))
        self.embeddings_index = {}
        with open(os.path.join(config.WORD2VEC_DIR, 'word2vec_'+str(self.EMBEDDING_DIM)+'_imdb.txt'),encoding="utf8") as fin:
            for line in fin:
                try:
                    values = line.split()
                    coefs = np.asarray(values[1:], dtype='float32')
                    word = values[0]
                    self.embeddings_index[word] = coefs
                except:
                    print(line)

        print('Found %s word vectors.' % len(self.embeddings_index))
#test
#glove=Word2Vec(50)
#initial_embeddings = glove.get_embedding({'good':2, 'movie':3})    
