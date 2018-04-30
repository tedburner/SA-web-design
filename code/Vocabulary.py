#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/4/10
import os
import random
import codecs
import re

import numpy as np
from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2

from utils.constant import path, max_words


def load_data(positives, negatives):
    pos_dir = path + 'data/test/pos'
    neg_dir = path + 'data/test/neg'
    return load_dir(pos_dir, 1, positives) + load_dir(neg_dir, 0, negatives)


def load_dir(dirname, label, size):
    data = []
    files = os.listdir(dirname)
    random.shuffle(files)
    for fname in files[:size]:
        for line in open(os.path.join(dirname, fname)):
            data.append((line, label))
    return data


def size():
    vocab = build()
    return len(vocab)


def tokenize(text):
    return [x.strip() for x in re.split('(\W+)', text) if x.strip()]


def vectorize(text):
    vocab = build()
    text = text.lower()
    words = filter(lambda x: x in vocab, tokenize(text))
    return [vocab[w] for w in words]


def build():
    vocab = dict()
    with codecs.open(path + "data/imdb.vocab", 'r', 'UTF-8') as trainfile:
        words = [x.strip().rstrip('\n') for x in trainfile.readlines()]
        vocab = dict((c, i + 1) for i, c in enumerate(words))
    return vocab


def build_model():
    model = Sequential()
    model.add(Embedding(size(), 128))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, W_regularizer=l2(0.01)))
    model.add(Activation('sigmoid'))

    model.load_weights(path + "data/imdb_lstm.w", by_name=True)
    json_string = model.to_json()
    model = model_from_json(json_string)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def pad(X):
    return pad_sequences(X, maxlen=max_words)


def classify(X):
    inp = [vectorize(X)]
    inp = np.array(pad(inp))
    model = build_model()
    y = model.predict(inp)[0][0]
    return (round(y), y)


if __name__ == '__main__':
    pos = "This movie, quite literally, does not have one redeeming feature. The characters are one-dimensional, cliched, incredibly misogynistic and stupid. The script looks as if it was cobbled together from 100 other movies, the acting is horrible, and some of the 'gross-out' humour made me feel nauseous.Shame on you, Gregory Poirier, for thinking ANY of this would be funny or interesting!The worst movie I've seen in several years."
    neg = "Send them to the freezer. This is the solution two butchers find after they discover the popularity of selling human flesh. An incredible story with humor and possible allegories that make it much more than a horror film. The complex characters defy superficial classification and make the story intriguing and worthwhile - if you can stand it. Definitely a dark film but also a bit redemptive."
    print classify(pos)
    print classify(neg)
