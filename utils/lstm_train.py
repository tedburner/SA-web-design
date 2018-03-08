#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/2

# 定义网络结构
from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation

# 参数设置
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40


def make_model(vocab_size, X_train, Y_train, X_test, Y_test):
    print '创建模型...'
    model = Sequential()
    # 嵌入层 输入数据的维度就是max_features，即我们选取的词频列表中排名前max_features的词频，输入维度为embedding_dims = 128
    model.add(Embedding(vocab_size,
                        EMBEDDING_SIZE,
                        input_length=MAX_SENTENCE_LENGTH))

    model.add(LSTM(HIDDEN_LAYER_SIZE,
                   dropout=0.2,
                   recurrent_dropout=0.2))

    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    print '编译模型...'
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print "训练..."
    model.fit(X_train,
              Y_train,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(X_test, Y_test))

    print "评估..."
    score, acc = model.evaluate(X_test,
                                Y_test,
                                batch_size=BATCH_SIZE)
    print 'Test score:', score
    print 'Test accuracy:', acc
    return model
