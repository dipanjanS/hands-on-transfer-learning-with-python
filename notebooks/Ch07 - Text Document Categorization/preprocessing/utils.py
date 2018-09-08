# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:07:15 2018

@author: tghosh
"""
from bs4 import BeautifulSoup
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
import re
import sys


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text



def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip() or len(doc)>30:
            filtered_corpus.append(doc)
            filtered_labels.append(label)
    return filtered_corpus, filtered_labels

class Preprocess:
    NUM_SENTENCES = 10
    SENTENCE_LEN = 30
    MIN_WD_COUNT = 5
    MAX_SEQUENCE_LENGTH = SENTENCE_LEN * NUM_SENTENCES
    
    def __init__(self, corpus):
        Preprocess.MAX_SEQUENCE_LENGTH = Preprocess.SENTENCE_LEN * Preprocess.NUM_SENTENCES
        self.corpus = corpus
                
    def _build_vocab(self):
        word_index ={}
        for doc in self.corpus:
            for sentence in sent_tokenize(doc):
                tokens = wordpunct_tokenize(sentence)
                tokens = [token.lower().strip()  for token in tokens]
                tokens = [token for token in tokens  if re.match('^[a-z]+$',token) is not None ]
                for token in tokens:
                    word_index[token] = word_index.get(token, 0)+1
                    
        filtered_word_index={}
        # i= 0 for empty, 1 for OOV
        i = 2
        for word, count in word_index.items():
            if count >= Preprocess.MIN_WD_COUNT :
                filtered_word_index[word] = i
                i +=1
        print('Found %s unique tokens.' % len(filtered_word_index))
        return filtered_word_index
    
    def _text2wordindex_seq(self, word_index, corpus):
        '''
        Splits each doc into sentences and then converts the sentence into a sequence of word indices.
        Also, padds short sentences with zeros and short documents with zero sentences.
        '''
        data = []
        doc_count = 0 
        for doc in corpus:
            doc2wordseq = []
            sent_num =0
            doc_count+=1
            if doc_count%1000 == 0 :
                percent_processed = doc_count*100/len(corpus)
                sys.stdout.write("\r%f%% documents processed." % percent_processed)
                sys.stdout.flush()
            for sentence in sent_tokenize(doc):
                
                words = wordpunct_tokenize(sentence)
                words = [token.lower().strip()  for token in words]
                word_id_seq = [word_index[word] for word in words  if word_index.get(word) is not None]
                #word_id_seq = tokenizer.texts_to_sequences([sentence])
                padded_word_id_seq = pad_sequences([word_id_seq], maxlen=Preprocess.SENTENCE_LEN,
                                                   padding='post',
                                                   truncating='post')

                if sent_num < Preprocess.NUM_SENTENCES:
                    doc2wordseq = doc2wordseq + list(padded_word_id_seq[0])        
                else:
                    break
                sent_num +=1
            #incase #sentences in doc is lass than NUM_SENTENCES do post padding
            doc2wordseq = pad_sequences([doc2wordseq], maxlen=Preprocess.MAX_SEQUENCE_LENGTH,
                                                   padding='post',
                                                   truncating='post')
            data.append(doc2wordseq[0])
        sys.stdout.write("\rAll documents processed." )
        return data
    
    def fit(self):
        word_index = self._build_vocab()
        self.word_index = word_index
        self.processed = self._text2wordindex_seq(word_index, self.corpus)
        return self.processed
    
    def transform(self, corpus):
        return self._text2wordindex_seq(self.word_index, corpus)
    
    def get_vocab_size(self):
        if self.word_index:
            return len(self.word_index)+2
        else:
            raise ValueError('fit must be called first to build vocab')
            
    def get_vocab(self):
        if self.word_index:
            return self.word_index
        else:
            raise ValueError('fit must be called first to build vocab')
