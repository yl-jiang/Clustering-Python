#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/26 17:18
# @Author  : jyl
# @File    : demo.py
import numpy as np
import os

data_path = r'C:\Users\jyl07\Desktop\Aggregation.txt'
raw_data = np.loadtxt(data_path, delimiter='	', usecols=[0, 1])
print(raw_data)




