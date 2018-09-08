# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:57:16 2018

@author: tghosh
"""

from keras.layers import Input, Dense, Embedding, Conv1D, Dropout, Concatenate, Lambda, GaussianNoise
from keras.layers.core import Reshape, Flatten, Permute
from keras.models import Model
from keras import regularizers
from model.custom_layer import KMaxPooling
import json

class TrainingParameters:
    
    def __init__(self,  model_name ,
                        model_file_path, 
                        model_hyper_parameters, 
                        model_train_parameters, 
                        seed = 55,                 
                        test_data_proportion = 0.3,
                        batch_size = 64,
                        num_epochs = 20,
                        validation_split = 0.05,
                        optimizer = 'rmsprop',
                        learning_rate = 0.001):
        self.model_name = model_name
        self.model_file_path = model_file_path
        self.model_hyper_parameters = model_hyper_parameters
        self.model_train_parameters = model_train_parameters
        self.seed = seed
        self.test_data_proportion = test_data_proportion
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        self.optimizer = optimizer
        self.learning_rate = learning_rate
    
    
    def save(self):
        with open(self.model_train_parameters, "w", encoding= "utf-8") as file:
            json.dump(self.__dict__, file)



class DocumentModel:
    
    def __init__(self,  vocab_size, 
                        word_index,
                        embedding_dim=50,
                        embedding_weights = None,
                        embedding_regularizer_l2 = 0.0,
                        train_embedding=True,
                        sentence_len=30,
                        num_sentences=10,                        
                        word_kernel_size = 5, 
                        word_filters=30, 
                        sent_kernel_size=5,
                        sent_filters = 16,
                        sent_k_maxpool =3 ,
                        input_dropout = 0,
                        doc_k_maxpool = 4,
                        sent_dropout = 0,
                        hidden_dims = 64,
                        conv_activation = 'relu',
                        hidden_activation = 'relu',
                        hidden_dropout = 0,
                        num_hidden_layers = 1,
                        hidden_gaussian_noise_sd = 0.5,
                        hidden_layer_kernel_regularizer = 0.0,
                        final_layer_kernel_regularizer = 0.0, 
                        num_units_final_layer = 1,
                        learn_word_conv=True,
                        learn_sent_conv=True):
        
        
        self.vocab_size = vocab_size 
        self.word_index = word_index
        self.embedding_dim = embedding_dim
        self.embedding_weights = embedding_weights 
        self.train_embedding = train_embedding
        self.embedding_regularizer_l2 = embedding_regularizer_l2
        self.sentence_len = sentence_len
        self.num_sentences = num_sentences                   
        self.word_kernel_size =  word_kernel_size
        self.word_filters = word_filters
        self.sent_kernel_size = sent_kernel_size
        self.sent_filters = sent_filters
        self.sent_k_maxpool = sent_k_maxpool
        self.input_dropout = input_dropout
        self.doc_k_maxpool = doc_k_maxpool
        self.sent_dropout = sent_dropout
        self.hidden_dims = hidden_dims
        self.conv_activation = conv_activation
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.num_hidden_layers = num_hidden_layers
        self.hidden_gaussian_noise_sd = hidden_gaussian_noise_sd
        self.final_layer_kernel_regularizer = final_layer_kernel_regularizer  
        self.hidden_layer_kernel_regularizer = hidden_layer_kernel_regularizer                  
        self.learn_word_conv = learn_word_conv
        self.learn_sent_conv = learn_sent_conv
        self.num_units_final_layer=num_units_final_layer
        if vocab_size != len(word_index):
            print("Vocab Size = {}  and the index of vocabulary words passed has {} words".format(vocab_size,len(word_index)))
            
        self._build_model()
        self.weights_file = None
        
    def _build_model(self):
        max_seq_length = self.sentence_len*self.num_sentences
        #Embedding Layer
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                input_length=max_seq_length,
                                trainable=self.train_embedding,
                                embeddings_regularizer = regularizers.l2(self.embedding_regularizer_l2),
                                name='imdb_embedding')
        
        if self.embedding_weights is not None:
            embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                weights=[self.embedding_weights],
                                input_length=max_seq_length,
                                trainable=self.train_embedding,
                                embeddings_regularizer = regularizers.l2(self.embedding_regularizer_l2),
                                name='imdb_embedding')
        
        #input layer : sequence of word indices for each sentence
        sequence_input = Input(shape=(max_seq_length,), dtype='int32')
        z = embedding_layer(sequence_input)
        if self.input_dropout>0:
            z = Dropout(self.input_dropout)(z) 
    
        conv_blocks = []
        i=0
        #same convolution filters to be used for all sentences.
        word_conv_model = Conv1D(filters=self.word_filters,
                                 kernel_size=self.word_kernel_size,
                                 padding="valid",
                                 activation=self.conv_activation, 
                                 trainable = self.learn_word_conv,
                                 name = "word_conv",
                                 strides=1)
    
        for sent in range(self.num_sentences):
            #get once sentence from the input
            sentence =  Lambda(lambda x : x[:,sent*self.sentence_len: (sent+1)*self.sentence_len, :])(z) 
            conv = word_conv_model(sentence)            
            conv = KMaxPooling(k=self.sent_k_maxpool)(conv)
            #transpose pooled values per sentence
            conv = Reshape([self.word_filters*self.sent_k_maxpool,1])(conv)
            conv_blocks.append(conv)
        
        #append all sentence convolution feature maps and make sentence embeddings
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    
        #transform to (steps, input_dim)
        z = Permute([2,1], name='sentence_embeddings')(z)
        
        if self.sent_dropout>0:
            z = Dropout(self.sent_dropout)(z) 
    
        sent_conv = Conv1D(filters=self.sent_filters,
                              kernel_size=self.sent_kernel_size,
                              padding="valid",
                              activation=self.conv_activation,
                              trainable = self.learn_sent_conv,
                              name = 'sentence_conv',
                              strides=1)(z)
    
        z = KMaxPooling(k=self.doc_k_maxpool)(sent_conv)
        z = Flatten(name='document_embedding')(z)
        
        if self.hidden_gaussian_noise_sd:
            z = GaussianNoise(self.hidden_gaussian_noise_sd)(z)
        elif self.hidden_dropout:
            z = Dropout(self.hidden_dropout)(z) 
        
        for i in range(self.num_hidden_layers):
            layer_name = 'hidden_{}'.format(i)
            z = Dense(self.hidden_dims, activation=self.hidden_activation, name=layer_name, 
                      kernel_regularizer=regularizers.l2(self.hidden_layer_kernel_regularizer))(z)
        
        output_activation = 'sigmoid'
        if self.num_units_final_layer>1:
            output_activation = 'softmax'
            
        model_output = Dense(self.num_units_final_layer, activation=output_activation,
                             kernel_regularizer=regularizers.l2(self.final_layer_kernel_regularizer),
                             name='final')(z)
    
        self._model = Model(sequence_input, model_output)

    
    def get_document_model(self):
        return Model(inputs=self._model.input,
                     outputs=self._model.get_layer('document_embedding').output)
    
    def get_sentence_model(self):
        return Model(inputs=self._model.input,
                     outputs=self._model.get_layer('sentence_embeddings').output)
    
    def get_classification_model(self):
        return self._model

    def _save_model(self,file_name):
        model_params = {}      
        for key in self.__dict__.keys():
            if key not in ['_model','embedding_weights']:
                model_params[key] = self.__dict__[key]             
        
        with open(file_name, "w", encoding= "utf-8") as hp_file:
            json.dump(model_params, hp_file)
        
    def load_model(file_name):    
        with open(file_name, "r", encoding= "utf-8") as hp_file:
            model_params = json.load(hp_file)
            doc_model = DocumentModel( **model_params)   
            print(model_params)
        return doc_model   
    
    def load_model_weights(self, model_weights_filename):
        self._model.load_weights(model_weights_filename, by_name=True)
        
##Test
#doc_model = DocumentModel(vocab_size=10000, word_index={'he':2})
#doc_model._save_model('test.hyper')
#doc_model = DocumentModel.load_model('test.hyper')
#doc_model.get_sentence_model().summary()

