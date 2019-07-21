# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 20:32:30 2019

@author: wxw
"""

import numpy as np

σ = np.tanh
w1 = -2.5
b1 = 2.5
X=[0,1]
Y=[1,0]
def a1(a0):
    return σ(w1*a0+b1)
def Not(a0):
    return not bool(a0)
def mse(X,Y,a1):
    s=0
    for i in range(len(X)):
       s += (Y[i]-a1(X[i]))**2 
    return s/len(X)
print('a1(0) = ',a1(0))
print('a1(1) = ',a1(1))
print('mse = ',mse(X,Y,a1))
print('mse(Not) = ',mse(X,Y,Not))
print('rmse = ',mse(X,Y,a1)**0.5)

def cost(X,Y,σ,w1,b1):
    s=0
    for i in range(len(X)):
       s += (Y[i]-σ(w1*X[i]+b1))**2 
    return s/len(X) 

#画出损失的图像，根据能看出使得损失最小的参数取值是w=-b,b>2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
w1 = np.arange(-5, 5, 0.1)
b1 = np.arange(-5, 5, 0.1)
W1, B1 = np.meshgrid(w1, b1)  # 网格的创建，生成二维数组
plt.xlabel('w1')
plt.ylabel('b1')
ax.plot_surface(W1, B1, cost(X,Y,σ,W1,B1), rstride=1, cstride=1, cmap='rainbow')