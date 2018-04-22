#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/3/21
# @descrip 系统中的常量
import collections

path = "/Users/jianglingjun/Document/PycharmProjects/SA-web-design/"

MAX_NB_WORDS = 112531  # 处理的最大单词数量
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 400
max_words = 500
# 七牛云上的图片
image_urls = ['https://p1gixpzpk.bkt.clouddn.com/cry.jpeg',
              'https://p1gixpzpk.bkt.clouddn.com/laugh.jpeg']

maxlen = 0  # 句子最大长度
num_recs = 0  # 样本数
word_freqs = collections.Counter()  # 词频
