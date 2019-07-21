# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 18:32:03 2019

@author: wxw
"""

import numpy as np
#sigma = np.tanh
sigma = lambda z : 1/(1+np.exp(-z))
d_sigma = lambda z : np.cosh(z/2)**(-2)/4#速度最快
def a1(w1, b1, a0):
    z = w1 * a0 + b1
    return sigma(z)

def C(w1, b1, x, y):
    return (a1(w1,b1,x)-y)**2

def dCda(w1, b1, x, y):
    return 2*(a1(w1,b1,x)-y)

def dadz(w1, b1, x):
    z = w1 * x + b1
    return 1/np.cosh(z)**2

def dzdw(x):
    return x

def dCdw(w1, b1, x, y):
    return dCda(w1, b1, x, y) * dadz(w1, b1, x) * dzdw(x)

def dCdb(w1, b1, x, y):
    return dCda(w1, b1, x, y) * dadz(w1, b1, x)

w1 = -5
b1 = 4
sample=[(1,0),(0,1),(0.5,0.5)]
#print('dCdw = \n', dCdw (w1, b1, 1, 0))
#print('dCdb = \n', dCdb (w1, b1, 0, 1))

def gradient_decend(w1, b1, lr, gw, gb):
    w1 = w1 - lr * gw
    b1 = b1 - lr * gb
    return w1,b1

lr=0.1
for i in range(100000):
    (x,y) = sample[np.random.randint(3)]
    gw = dCdw(w1, b1, x, y)
    gb = dCdb(w1, b1, x, y)
    w1,b1 = gradient_decend(w1, b1, lr, gw, gb)
print(w1,b1)
print(C(w1, b1, x, y))
print(a1(w1, b1, 1))
print(a1(w1, b1, 0))
print(a1(w1, b1, 0.5))


