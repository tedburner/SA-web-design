#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/21
import keras
import numpy as np
from keras import Sequential
from keras.datasets import imdb
from keras.layers import Embedding, Flatten, Dense, LSTM, Dropout
from keras.preprocessing import sequence
from utils.constant import *

(X_train, y_train), (X_test, y_test) = imdb.load_data()

avg_len = list(map(len, X_train))
print(np.mean(avg_len))

m = max(list(map(len, X_train)), list(map(len, X_test)))

maxword = 400

X_train = sequence.pad_sequences(X_train, maxlen=maxword)
X_test = sequence.pad_sequences(X_test, maxlen=maxword)
vocab_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1

# 建模
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=maxword))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=100)
scores = model.evaluate(X_test, y_test)
print(scores)
model.save(path + 'data/model.h5')

