# -*- coding: utf-8 -*-
"""
Created on Thu May  3 08:12:41 2018

@author: tghosh
"""

import config
from preprocessing.utils import Preprocess, remove_empty_docs
from dataloader.embeddings import Word2Vec
from model.cnn_document_model import DocumentModel, TrainingParameters
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split




train_params = TrainingParameters('imdb_relu_activation', 
                                  model_file_path = config.MODEL_DIR+ '/imdb/model_04.hdf5',
                                  model_hyper_parameters = config.MODEL_DIR+ '/imdb/model_04.json',
                                  model_train_parameters = config.MODEL_DIR+ '/imdb/model_04_meta.json',
                                  num_epochs=20,
                                  batch_size=128)

train_params.save()

#train_df = Loader.load_imdb_data(directory = 'train')
train_df = pd.read_csv(config.IMDB_DATA_CSV + '/movie_reviews_train.csv', encoding='ISO-8859-1')
print(train_df.shape)

#test_df = Loader.load_imdb_data(directory = 'test')
test_df = pd.read_csv(config.IMDB_DATA_CSV + '/movie_reviews_test.csv', encoding='ISO-8859-1')
print(test_df.shape)

full_df = test_df.append(train_df)
train_df, test_df = train_test_split(full_df,  test_size = 0.3, random_state = 42)
print(test_df.shape, train_df.shape)

corpus = train_df['review'].tolist()
target = train_df['sentiment'].tolist()
corpus, target = remove_empty_docs(corpus, target)
print(len(corpus))

Preprocess.NUM_SENTENCES = 20
print(Preprocess.MAX_SEQUENCE_LENGTH)
preprocessor = Preprocess(corpus=corpus)
print(Preprocess.MAX_SEQUENCE_LENGTH)
corpus_to_seq = preprocessor.fit()

test_corpus = test_df['review'].tolist()
test_target = test_df['sentiment'].tolist()
test_corpus, test_target = remove_empty_docs(test_corpus, test_target)
print(len(test_corpus))

test_corpus_to_seq = preprocessor.transform(test_corpus)

x_train = np.array(corpus_to_seq)
x_test  = np.array(test_corpus_to_seq)

y_train = np.array(target)
y_test = np.array(test_target)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


glove=Word2Vec(50)
initial_embeddings = glove.get_embedding(preprocessor.word_index)


imdb_model = DocumentModel(vocab_size=preprocessor.get_vocab_size(),
                                    word_index = preprocessor.word_index,
                                    num_sentences=Preprocess.NUM_SENTENCES,                                    
                                    embedding_weights=initial_embeddings,
                                    embedding_regularizer_l2 = 0.0,
                                    word_filters = 30,
                                    sent_filters = 16,
                                    doc_k_maxpool = 5,
                                    conv_activation = 'relu',
                                    train_embedding = False,
                                    learn_word_conv = True,
                                    learn_sent_conv = True,
                                    hidden_dims=64,
                                    num_hidden_layers=1,
                                    sent_dropout = 0.2,                                        
                                    input_dropout=0.1, 
                                    hidden_gaussian_noise_sd=0.5,
                                    final_layer_kernel_regularizer=0.04)

imdb_model._save_model(train_params.model_hyper_parameters)

from keras.optimizers import Adam
adam = Adam(lr=0.002)
              
imdb_model.get_classification_model()\
    .compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=["accuracy"])

checkpointer = ModelCheckpoint(filepath=train_params.model_file_path,
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)

early_stop = EarlyStopping(patience=2)
imdb_model.get_classification_model().fit(x_train, y_train, batch_size=train_params.batch_size,
               epochs=train_params.num_epochs,
               verbose=2,
               #validation_split=0.01,
               validation_data=[x_test, y_test],
               callbacks=[checkpointer])

imdb_model.load_model_weights(train_params.model_file_path)
imdb_model.get_classification_model().evaluate( x_test, y_test, batch_size=train_params.batch_size*10, verbose=2)



#RESULTS
#imdb/model_03.hdf5 : 0.8757, #hidden units =16
#imdb/model_04.hdf5 : 0.8663, #hidden units =64