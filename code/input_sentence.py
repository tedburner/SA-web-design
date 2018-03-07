#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/4
import numpy as np
import nltk
from keras.models import load_model
from keras.preprocessing import sequence

MAX_SENTENCE_LENGTH = 40


# 手动输入句子分析
def input_sentence(INPUT_SENTENCE):
    model = load_model('/Users/jianglingjun/Document/PycharmProjects/SA-web-design/code/train_model.h5')
    print INPUT_SENTENCE
    # # 使用numpy生成一个list的数组
    # input_lsit = np.empty(1, dtype=list)
    # i = 0
    # # 使用nltk句子分词
    # words = nltk.word_tokenize(INPUT_SENTENCE.lower())
    # seq = []
    # for word in words:
    #     if word in word2index:
    #         seq.append(word2index[word])
    #     else:
    #         seq.append(word2index['UNK'])
    # input_lsit[i] = sequence
    #
    # input_lsit = sequence.pad_sequences(input_lsit, maxlen=MAX_SENTENCE_LENGTH)
    label = model.predict(INPUT_SENTENCE)
    label2word = {1: '积极', 0: '消极'}
    return label2word[label]
if __name__ == '__main__':
    input_sentence('I love you')
