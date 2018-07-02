#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/8 16:13
# @Author  : jyl
# @File    : PLA.py
import numpy as np


data_path = r'D:\Data\ML\PLA_data.txt'
data = np.loadtxt(data_path, usecols=(0, 1, 2, 3, 4), dtype=np.float32)
w0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
for itr_num in range(1, 2001):
    error_pre = 0
    np.random.seed(itr_num)
    tem = np.random.permutation(data)
    signs = tem[:, -1]
    for index, row in enumerate(tem):
        pre = np.sum(w0[:4] * row[0:4]) + w0[-1]
        if np.sign(pre) != signs[index]:
            error_pre += 1
            w0[:4] = w0[0:4] + signs[index] * row[0:4]
            w0[-1] = -pre + signs[index] * 0.07
    accuracy = np.sum(np.equal(np.sign(np.sum(w0[0:4] * tem[:, 0:4], axis=1) + w0[-1]), signs).astype(int)) / 400
    print('%d iteration update times: %d, accuracy: %6.3f' % (itr_num, error_pre, accuracy))
    if error_pre == 0:
        print(w0)
        break





