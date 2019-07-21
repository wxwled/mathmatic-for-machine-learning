# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:19:02 2019

@author: wxw
"""

import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-1*x))
σ = sigmoid
a0 = np.random.rand(4,1)
W1 = np.random.rand(3,4)
b1 = np.random.rand(3,1)
W2 = np.random.rand(2,3)
b2 = np.random.rand(2,1)
print('a0 = \n',a0)
print('W1 = \n',W1)
print('b1 = \n',b1)
print('W2 = \n',W2)
print('b2 = \n',b2)
#calcuate a1=σ(W1*a0+b1)'
def a1(a0):
    return σ(W1.dot(a0)+b1)
def a2(a1):
    return σ(W2.dot(a1)+b2)
print('a1 = \n',a1(a0))
print('a2 = \n',a2(a1(a0)))
