# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:19:42 2019

@author: wxw
"""
import numpy as np
import matplotlib.pyplot as plt
import time

sigma = lambda z : 1/(1+np.exp(-z))
d_sigma = lambda z : np.cosh(z/2)**(-2)/4#速度最快
#d_sigma2 = lambda z : sigma(z)*sigma(-z)#速度最慢

#速度不快
#def d_sigma3(z):
#    t = sigma(z)
#    return t*(1-t)

#print(sigma(1),d_sigma(1),d_sigma2(1))

#速度测试
#time1=time.time()
#for i in range(10000):
#    d_sigma(1)
#time2=time.time()
#print(time2-time1)

def reset_network (n1=6,n2=7,random=np.random):
    global W1,W2,W3,b1,b2,b3
    W1 = random.randn(n1,1)/2
    W2 = random.randn(n2,n1)/2
    W3 = random.randn(1,n2)/2
    b1 = random.randn(n1,1)/2
    b2 = random.randn(n2,1)/2
    b3 = random.randn(1,1)/2

#test
#reset_network()
#print(W1,W2,W3,b1,b2,b3)
    
def network_function(a0):
    z1 = W1 @ a0 + b1
    a1 = sigma(z1)
    z2 = W2 @ a1 + b2
    a2 = sigma(z2)
    z3 = W3 @ a2 + b3
    a3 = sigma(z3)
    return a0, z1, a1, z2, a2, z3, a3

def cost(x, y):
    return np.linalg.norm(network_function(x)[-1]-y)**2 / x.size

# GRADED FUNCTION

def J_W3(x,y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = J @ a2.T / x.size
    return J

def J_b3(x,y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J

def J_W2(x,y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = J @ a1.T / x.size
    return J

def J_b2(x,y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J  

def J_W1(x,y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    J = J * d_sigma(z1)
    J = J @ a0.T / x.size
    return J

def J_b1(x,y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    J = J * d_sigma(z1)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J  

def update_network(x,y,lr):
    global W1,W2,W3,b1,b2,b3
#    print(J_W1(x,y))
#    print(J_W2(x,y))
#    print(J_W3(x,y))
#    print(J_b1(x,y))
#    print(J_b2(x,y))
#    print(J_b3(x,y))
    W1 = W1 - lr * J_W1(x,y)
    W2 = W2 - lr * J_W2(x,y)
    W3 = W3 - lr * J_W3(x,y)
    b1 = b1 - lr * J_b1(x,y)
    b2 = b2 - lr * J_b2(x,y)
    b3 = b3 - lr * J_b3(x,y)

x = np.array([[0.1,0,-0.1]])
y = np.array([[0,0.1,0.2]])
reset_network()
#print(network_function(x))
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x, y)
plt.show()

lr=0.1
for i in range(10000):
    k=np.random.randint(3)
    update_network(x.T[k].reshape(1,1),y.T[k].reshape(1,1),lr)
    if i%5000 ==0:
        print(cost(x,y))

y_pred=network_function(x)[-1]
print(cost(x, y))
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x,y_pred)
plt.show()







