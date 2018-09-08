# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:44:05 2018

@author: tghosh
"""
import config
from preprocessing import utils
import re
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import reuters 
from sklearn.preprocessing import LabelEncoder

class Loader:
    
    amzn_reviews_kaggle_regx = re.compile(r'__label__(?P<label>([1|2]))(\s+)(?P<summary>(.*)):(?P<review>(.*))')    
    
    def load_20newsgroup_data(categories = None, subset='all'):
        data = fetch_20newsgroups(subset=subset,
                              shuffle=True,
                              remove=('headers', 'footers', 'quotes'),
                             categories = categories)
        return data

    
    def load_imdb_data(directory = 'train', datafile = None):
        '''
        Parse IMDB review data sets from Dataset from http://ai.stanford.edu/~amaas/data/sentiment/
        and save to csv.
        '''    
        labels = {'pos': 1, 'neg': 0}
        df = pd.DataFrame()
        
        for sentiment in ('pos', 'neg'):
            path =r'{}/{}/{}'.format(config.IMDB_DATA, directory, sentiment)
            for review_file in os.listdir(path):
                with open(os.path.join(path, review_file), 'r', encoding= 'utf-8') as input_file:
                    review = input_file.read()
                df = df.append([[utils.strip_html_tags(review), labels[sentiment]]], 
                                             ignore_index=True)
        
        df.columns = ['review', 'sentiment']
        indices = df.index.tolist()
        np.random.shuffle(indices)
        indices = np.array(indices)
        df = df.reindex(index=indices)
        if datafile is not None:
            df.to_csv(os.path.join(config.IMDB_DATA_CSV, datafile), index=False)
        return df
    
    def load_imdb_unsup():
        df = pd.DataFrame()
        
        path =r'{}/{}/{}'.format(config.IMDB_DATA, 'train', 'unsup')
        for review_file in os.listdir(path):
            with open(os.path.join(path, review_file), 'r', encoding= 'utf-8') as input_file:
                review = input_file.read()
                df = df.append([[utils.strip_html_tags(review)]], ignore_index=True)
        
        df.columns = ['review']
        indices = df.index.tolist()
        np.random.shuffle(indices)
        indices = np.array(indices)
        df = df.reindex(index=indices)
        return df
    
    def load_amazon_reviews(test_or_train='train'):
        '''
        Loads data from to a dataframe. Data is in fastText format.
        https://www.kaggle.com/bittlingmayer/amazonreviews
        '''
        data = []
        fastText_filename = os.path.join(config.AMAZON_TEST_DATA ,'test.ft.txt')
        if test_or_train is 'train':
            fastText_filename = os.path.join(config.AMAZON_TRAIN_DATA ,'train.ft.txt')
           
        with open(fastText_filename, encoding="utf8") as fin:
            for line in fin:
                m = Loader.amzn_reviews_kaggle_regx.search(line)
                data.append({
                    'review':'{} . {}'.format(m.group('summary'),m.group('review')),
                    'sentiment': int(m.group('label'))-1 #convert 1,2 to 0, 1        
                })
        return pd.DataFrame(data)
    
    def load_reuters(test_or_train='train'):
        # List of categories
        data = []
        categories = reuters.categories()
        encoder = LabelEncoder()
        encoder.fit(categories)
        
        print(str(len(categories)) + " categories")
        
        for category in categories:
            category_docs = reuters.fileids(category)
            for document_id in category_docs:   
                if document_id.startswith(test_or_train):  
                    print(document_id)
                    data.append({
                        'document':reuters.raw(document_id),
                        'label': encoder.transform([category])[0]
                    })
        
        return pd.DataFrame(data)
    
    def load_processed_20newsgrp_data(test_or_train='train'):
        return pd.read_csv(config.PROCESSED_20_NEWS_GRP + '/20ng-'+test_or_train+'-all-terms.txt', 
                       header=None, sep='\t', 
                       names=['label', 'text'])
#Test
#Loader.parse_imdb_data(directory='test', datafile='movie_reviews_test.csv')    
#testdf=Loader.load_amazon_reviews('test')
#print(testdf.shape)


