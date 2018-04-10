#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/4
import collections

import nltk
import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence

from code.train import word_to_index
from utils.constant import *


# 手动输入句子分析
def input_sentence(INPUT_SENTENCE):
    model = load_model(path+'/data/train_model.h5')
    return predict(INPUT_SENTENCE, model)


def predict(text, model):
    word2index, vocab_size, num_recs, word_freqs = word_to_index(0, 0, collections.Counter())
    # 使用numpy生成一个list的数组
    input_lsit = np.empty(1, dtype=list)
    # 使用nltk句子分词
    words = nltk.word_tokenize(text.lower())
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    input_lsit[0] = seq
    input_lsit = sequence.pad_sequences(input_lsit, maxlen=MAX_SENTENCE_LENGTH)
    labels = [int(round(x[0])) for x in model.predict(input_lsit)]
    return labels[0]


if __name__ == '__main__':
    word = 'I love you'
    result = input_sentence(word)
    label2word = {1: '积极', 0: '消极'}
    print label2word[result]
