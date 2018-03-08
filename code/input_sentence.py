#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/4
import numpy as np
import nltk
from nltk.corpus import stopwords
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import collections

from code.train import word_to_index

MAX_SENTENCE_LENGTH = 40
MAX_NB_WORDS = 50000  # 处理的最大单词数量


# 手动输入句子分析
def input_sentence(INPUT_SENTENCE):
    model = load_model('../data/train_model.h5')
    return predict(INPUT_SENTENCE,model)


def predict(text,model):
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
    label2word = {1: '积极', 0: '消极'}
    return label2word[labels[0]]


if __name__ == '__main__':
    word = ''
    result = input_sentence(word)
    print result
