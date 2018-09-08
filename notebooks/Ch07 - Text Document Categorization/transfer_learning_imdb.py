# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:33:45 2018

@author: tghosh
"""

import config
from dataloader.loader import Loader
from preprocessing.utils import Preprocess, remove_empty_docs
from dataloader.embeddings import GloVe
from model.cnn_document_model import DocumentModel, TrainingParameters
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC



train_params = TrainingParameters('imdb_transfer_tanh_activation', 
                                  model_file_path = config.MODEL_DIR+ '/imdb/transfer_model_10.hdf5',
                                  model_hyper_parameters = config.MODEL_DIR+ '/imdb/transfer_model_10.json',
                                  model_train_parameters = config.MODEL_DIR+ '/imdb/transfer_model_10_meta.json',
                                  num_epochs=30,
                                  batch_size=128)


#train_df = Loader.load_imdb_data(directory = 'train')
train_df = pd.read_csv(config.IMDB_DATA_CSV + '/movie_reviews_train.csv', encoding='ISO-8859-1')
print(train_df.shape)
# build TFIDF features on train reviews
tv = TfidfVectorizer(use_idf=True, min_df=0.00005, max_df=1.0, ngram_range=(1, 1), stop_words = 'english', 
                      sublinear_tf=True)
tv_features = tv.fit_transform(train_df['review'].tolist())


#test_df = Loader.load_imdb_data(directory = 'test')
test_df = pd.read_csv(config.IMDB_DATA_CSV + '/movie_reviews_test.csv', encoding='ISO-8859-1')
print(test_df.shape)

corpus = train_df['review'].tolist()
target = train_df['sentiment'].tolist()
corpus, target = remove_empty_docs(corpus, target)
print(len(corpus))

preprocessor = Preprocess(corpus=corpus)
corpus_to_seq = preprocessor.fit()

#Take only 5% of data for training
train_df = train_df.sample(frac=0.05, random_state = train_params.seed)
corpus = train_df['review'].tolist()
target = train_df['sentiment'].tolist()
corpus_to_seq = preprocessor.transform(corpus)

test_corpus = test_df['review'].tolist()
test_target = test_df['sentiment'].tolist()
test_corpus, test_target = remove_empty_docs(test_corpus, test_target)
print(len(test_corpus))

test_corpus_to_seq = preprocessor.transform(test_corpus)

x_train = np.array(corpus_to_seq)
x_test  = np.array(test_corpus_to_seq)

y_train = np.array(target)
y_test = np.array(test_target)

print(x_train.shape, y_train.shape)

glove=GloVe(50)
initial_embeddings = glove.get_embedding(preprocessor.word_index)

amazon_review_model = DocumentModel.load_model("C:/Users/tghosh/Work/Data Science/Transfer Learning/Chapter-7/models/amazonreviews/model_06.json")
amazon_review_model.load_model_weights("C:/Users/tghosh/Work/Data Science/Transfer Learning/Chapter-7/models/amazonreviews/model_06.hdf5")
learned_embeddings = amazon_review_model.get_classification_model().get_layer('imdb_embedding').get_weights()[0]

glove.update_embeddings(preprocessor.word_index , np.array(learned_embeddings), amazon_review_model.word_index)

initial_embeddings = glove.get_embedding(preprocessor.word_index)

imdb_model = DocumentModel(vocab_size=preprocessor.get_vocab_size(),
                                    word_index = preprocessor.word_index,
                                    num_sentences=Preprocess.NUM_SENTENCES,                                    
                                    embedding_weights=initial_embeddings,
                                    embedding_regularizer_l2 = 0.0,
                                    conv_activation = 'tanh',
                                    train_embedding = True,
                                    learn_word_conv = False,
                                    learn_sent_conv = False,
                                    hidden_dims=64,                                        
                                    input_dropout=0.1, 
                                    hidden_layer_kernel_regularizer=0.01,
                                    final_layer_kernel_regularizer=0.01)


#transfer word & sentence conv filters
for l_name in ['word_conv','sentence_conv','hidden_0', 'final']:
    imdb_model.get_classification_model()\
              .get_layer(l_name).set_weights(weights=amazon_review_model
                                .get_classification_model()
                                .get_layer(l_name).get_weights())

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
               verbose=2,validation_split=0.01,
               callbacks=[checkpointer])

#imdb_model.load_model_weights(train_params.model_file_path)
imdb_model.get_classification_model().evaluate( x_test, y_test, batch_size=train_params.batch_size*10, verbose=2)



#imdb_model._save_model(train_params.model_hyper_parameters)
#train_params.save()



#learned_embeddings = imdb_model.get_classification_model().get_layer('imdb_embedding').get_weights()[0]
#embd_change = {}
#for word, i in preprocessor.word_index.items():    
#    embd_change[word] = np.linalg.norm(initial_embeddings[i]-learned_embeddings[i])
#embd_change = sorted(embd_change.items(), key=lambda x: x[1], reverse=True)
#embd_change[0:100]


#print(len(tv.get_feature_names()))
#tv_train_features = tv.transform(corpus)
#tv_test_features = tv.transform(test_corpus)
#
#clf = SVC(C=1,kernel='linear', random_state=1, gamma=0.01)
#svm=clf.fit(tv_train_features, target)
#preds_test = svm.predict(tv_test_features)
#
#from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
#print(classification_report(y_test, preds_test))
#print(confusion_matrix(y_test, preds_test))