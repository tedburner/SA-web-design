#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/21

from keras.models import load_model
from keras.preprocessing.text import Tokenizer

from code.train import *

if __name__ == '__main__':
    texts = ['I like read.', 'Well done!']
    model = load_model(path + '/data/model.h5')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1
    encoded_docs = tokenizer.texts_to_sequences(texts)
    result = sequence.pad_sequences(encoded_docs, maxlen=MAX_SENTENCE_LENGTH)
    print result
    print model.predict(result)
    labels = [int(round(x[0])) for x in model.predict(result)]
    label2word = {1: '积极', 0: '消极'}
    for i in range(len(texts)):
        print('{}   {}'.format(label2word[labels[i]], texts[i]))
