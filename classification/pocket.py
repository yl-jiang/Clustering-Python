#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/8 20:11
# @Author  : jyl
# @File    : pocket.py
import numpy as np

train_data = r'D:\Data\ML\pocket\pocket_train.txt'
test_data = r'D:\Data\ML\pocket\pocket_test.txt'
data_train = np.loadtxt(train_data, usecols=(0, 1, 2, 3, 4), dtype=np.float32)
data_test = np.loadtxt(test_data, usecols=(0, 1, 2, 3), dtype=np.float32)
signs_test = np.loadtxt(test_data, usecols=4, dtype=np.float32)
w0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
pocket = None
start_accuracy = 0.0
for itr_num in range(1, 2001):
    error = 0
    np.random.shuffle(data_train)
    signs_train = data_train[:, 4]
    for index, row in enumerate(data_train):
        pre = np.sum(w0[:4] * row[:4]) + w0[-1]
        if np.sign(pre) != signs_train[index]:
            error += 1
            w0[:4] = w0[0:4] + signs_train[index] * row[:4]
            w0[-1] = -pre + signs_train[index] * 0.0017
    accuracy_1 = np.sum(np.equal(np.sign(np.sum(w0[0:4] * data_train[:, 0:4], axis=1) + w0[-1]), signs_train).astype(int)) / data_train.shape[0]

    if start_accuracy < accuracy_1:
        print('train accuracy: %6.3f' % accuracy_1)
        pocket = w0
        start_accuracy = accuracy_1
        accuracy_2 = np.sum(np.equal(np.sign(np.sum(pocket[0:4] * data_test[:, 0:4], axis=1) + pocket[-1]), signs_test).astype(int)) / data_test.shape[0]
        print('test accuracy: %6.3f' % accuracy_2)
        print('***' * 100)


