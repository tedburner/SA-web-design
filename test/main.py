#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/21

from keras.models import load_model
from keras.preprocessing.text import Tokenizer

from code.train import *

if __name__ == '__main__':
    input = 'I like read.'
    texts = [input]
    input_lsit = np.empty(1, dtype=list)
    model = load_model(path + '/data/model.h5')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    input_lsit = tokenizer.text_to_sequences(texts)
    result = sequence.pad_sequences(input_lsit, maxlen=MAX_SENTENCE_LENGTH)
    labels = [int(round(x[0])) for x in model.predict(result)]
    label2word = {1: '积极', 0: '消极'}
    print label2word[labels[0]]