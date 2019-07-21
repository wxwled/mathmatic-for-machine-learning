# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 20:56:46 2019

@author: wxw
"""

import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-1*x))
σ = np.tanh
#a0 = np.mat([[0.3,0.4,0.1]]).T
#W1 = np.mat([[-2,4,-1],
#               [6,0,3]])
#b1 = np.mat([[0.1,-2.5]]).T
#print('a0 = \n',a0)
#print('W1 = \n',W1)
#print('b1 = \n',b1)

#calcuate a1=σ(W1*a0+b1)'
def a1(a0,W1,b1):
    return σ(W1*a0+b1)

#print('a1 = \n',a1(a0,W1,b1))

W = np.mat([[-0.945,-0.266,-0.912],[2.055,1.218,0.229]])
b = np.mat([0.613,1.642]).T
X = np.mat([0.1,0.5,0.6]).T
Y = np.mat([0.25,0.75]).T
def cost(a1,Y):
    return np.linalg.norm(a1-Y)**2

Y_pred = a1(X,W,b)
print('cost = \n',cost(Y_pred,Y))

X = np.mat([0.7,0.6,0.2]).T
Y = np.mat([0.9,0.6]).T
Y_pred = a1(X,W,b)
print('cost = \n',cost(Y_pred,Y))