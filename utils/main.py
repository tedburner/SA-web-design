#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author : lingjun.jlj
# @create : 2018/4/10
import random

data = []
data.append(("1121", 0))
random.shuffle(data)
for x, y in data:
    print x, y
