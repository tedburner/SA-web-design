#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/2

from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np

# EDA
from utils.lstm_train import train_lstm


def train():
    maxlen = 0  # 句子最大长度
    word_freqs = collections.Counter()  # 词频
    num_recs = 0  # 样本数
    nltk.download('punkt')
    # 文本转为索引数字模式
    with open('/Users/jianglingjun/Document/PycharmProjects/SA-web-design/code/train_data.txt', 'r+') as f:
        for line in f:
            label, sentence = line.strip().split("\t")
            words = nltk.word_tokenize(sentence.lower())
            if len(words) > maxlen:
                maxlen = len(words)
            for word in words:
                word_freqs[word] += 1
            num_recs += 1
    print('max_len ', maxlen)
    print('nb_words ', len(word_freqs))

    # 准备数据
    MAX_SENTENCE_LENGTH = 40
    MAX_FEATURES = 2000
    vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
    word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v: k for k, v in word2index.items()}
    X = np.empty(num_recs, dtype=list)
    y = np.zeros(num_recs)
    i = 0

    with open('/Users/jianglingjun/Document/PycharmProjects/SA-web-design/code/train_data.txt', 'r+') as f:
        for line in f:
            label, sentence = line.strip().split("\t")
            words = nltk.word_tokenize(sentence.lower())
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs
            y[i] = int(label)
            i += 1
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

    # 数据划分
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    # 训练模型
    model = train_lstm(vocab_size, Xtrain, ytrain, Xtest, ytest)

    return model, word2index
