#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/4
import numpy as np
import nltk
from keras.preprocessing import sequence

MAX_SENTENCE_LENGTH = 40


def input_sentence(INPUT_SENTENCE, model, word2index):
    print INPUT_SENTENCE
    XX = np.empty(1, dtype=list)
    i = 0
    # 句子分词
    words = nltk.word_tokenize(INPUT_SENTENCE.lower())
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq

    XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
    labels = [int(round(x[0])) for x in model.predict(XX)]
    label2word = {1: '积极', 0: '消极'}
    return label2word[labels[i]]
