#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:37:17 2017

@author: adam
"""
import gensim
import numpy as np
from keras.models import load_model
import sys

keras_model = load_model('keras_model/lstm_sentiment-08-0.85.hdf5')
w2v_model = gensim.models.Word2Vec.load('word2vec_model/w2v_sentiment')


def words_2_vec(words,length):
    vec = np.zeros((length,100))
    for i in range(len(words)):
        if words[i] in w2v_model.wv.vocab.keys() and i < length:
            vec[i,:]=w2v_model.wv[words[i]]
    return vec


def sentiment_classification(words):
    words = words.split()
    target = ['negative','positive']
    
    vec = words_2_vec(words,100)
    vec = vec.reshape(1,100,100)
    
    result = target[np.argmax(keras_model.predict(vec))]
    print result
    return result


if __name__=='__main__':
    words = sys.argv[1]
    sentiment_classification(words)    