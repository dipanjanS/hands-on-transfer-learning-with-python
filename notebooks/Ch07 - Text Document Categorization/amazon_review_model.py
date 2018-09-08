# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:34:14 2018

@author: tghosh
"""

import config
from dataloader.loader import Loader
from preprocessing.utils import Preprocess, remove_empty_docs
from dataloader.embeddings import GloVe
from model.cnn_document_model import DocumentModel, TrainingParameters
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


train_df = Loader.load_amazon_reviews('train')
print(train_df.shape)

test_df = Loader.load_amazon_reviews('test')
print(test_df.shape)

dataset = train_df.sample(n=200000, random_state=42)
dataset.sentiment.value_counts()


corpus = dataset['review'].values
target = dataset['sentiment'].values
print(corpus.shape, target.shape)

corpus, target = remove_empty_docs(corpus, target)
print(len(corpus))

preprocessor = Preprocess(corpus=corpus)
corpus_to_seq = preprocessor.fit()

holdout_corpus = test_df['review'].values
holdout_target = test_df['sentiment'].values
print(holdout_corpus.shape, holdout_target.shape)

holdout_corpus, holdout_target = remove_empty_docs(holdout_corpus, holdout_target)
print(len(holdout_corpus))
holdout_corpus_to_seq = preprocessor.transform(holdout_corpus)

glove=GloVe(50)
initial_embeddings = glove.get_embedding(preprocessor.word_index)

amazon_review_model = DocumentModel(vocab_size=preprocessor.get_vocab_size(),
                                    word_index = preprocessor.word_index,
                                    num_sentences=Preprocess.NUM_SENTENCES,                                    
                                    embedding_weights=initial_embeddings,
                                    conv_activation = 'tanh',
                                    hidden_dims=64,                                        
                                    input_dropout=0.40, 
                                    hidden_gaussian_noise_sd=0.5)

train_params = TrainingParameters('model_with_tanh_activation', 
                                  model_file_path = config.MODEL_DIR+ '/amazonreviews/model_06.hdf5',
                                  model_hyper_parameters = config.MODEL_DIR+ '/amazonreviews/model_06.json',
                                  model_train_parameters = config.MODEL_DIR+ '/amazonreviews/model_06_meta.json',
                                  num_epochs=35)

train_params.save()

amazon_review_model._model.compile(loss="binary_crossentropy", 
                            optimizer=train_params.optimizer,
                            metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath=train_params.model_file_path,
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)

early_stop = EarlyStopping(patience=2)

x_train = np.array(corpus_to_seq)
y_train  = np.array(target)

x_test = np.array(holdout_corpus_to_seq)
y_test = np.array(holdout_target)

print(x_train.shape, y_train.shape)

amazon_review_model.get_classification_model().fit(x_train, y_train, 
                      batch_size=train_params.batch_size, 
                      epochs=train_params.num_epochs,
                      verbose=2,
                      validation_split=train_params.validation_split,
                      callbacks=[checkpointer])

amazon_review_model.get_classification_model().evaluate( x_test, y_test, train_params.batch_size*10, verbose=2)

amazon_review_model._save_model(train_params.model_hyper_parameters)



''' Which embeddings changes most '''

learned_embeddings = amazon_review_model.get_classification_model().get_layer('imdb_embedding').get_weights()[0]

embd_change = {}
for word, i in preprocessor.word_index.items():    
    embd_change[word] = np.linalg.norm(initial_embeddings[i]-learned_embeddings[i])
embd_change = sorted(embd_change.items(), key=lambda x: x[1], reverse=True)
embd_change[0:20]

