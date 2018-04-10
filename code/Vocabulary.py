#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/4/10
import os
import random

import nltk

from utils.constant import path


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
