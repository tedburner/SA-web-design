#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/2
# @descrip 定义网络结构

from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation

from utils.constant import *


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
    # 接下来就加一个 Dense全联接层，抹平就是为了可以把这一个一个点全连接成一个层.
    model.add(Dense(1))
    # 接着再加一个激活函数
    model.add(Activation('sigmoid'))
    # 加载之前训练的结果
    # model.load_weights(path + "/data/train_model.h5")

    print '编译模型...'
    # 二分类问题binary_crossentropy；多分类问题categorical_crossentropy
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
