# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:34:06 2019

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np

def f(x,miu,sig):
    return np.exp(-(x-miu)**2/(2*sig**2))/np.sqrt(2*np.pi)/sig

def dfdmiu(x,miu,sig):
    return f(x,miu,sig)*(x-miu)/sig**2

def dfdsig(x,miu,sig):
    return f(x,miu,sig)*(((x-miu)/sig)**2-1)/sig

#steepest descent 最速下降法
def steepest_step(x,y,miu,sig,aggression):
    J = np.array([-2*(y-f(x,miu,sig))@dfdmiu(x,miu,sig),-2*(y-f(x,miu,sig))@dfdsig(x,miu,sig)])
    step = -J*aggression
    return step


#随机生成男女身高数据
height_data=np.hstack(((np.random.randn(5000)*8+175),(np.random.randn(4000)*8+165)))

#画图
plt.figure(figsize=(7,9))
ax=plt.subplot(211)

#直方图数据
y,x,_=ax.hist(height_data,bins=100,histtype="stepfilled",density=True,alpha=0.8)
p=np.array([])
for i in range(len(x)-1):
    p=np.append(p,(x[i]+x[i+1])/2)
x=p

#选取初值
miu=145
sig=6
p=np.array([miu])
q=np.array([sig])
for i in range(1000):
    dmiu,dsig = steepest_step(x,y,miu,sig,200)
    miu += dmiu
    sig += dsig
    p = np.append(p,[miu],axis=0)
    q = np.append(q,[sig],axis=0)
ax.plot(x,f(x,miu,sig))
ax.set(xlabel='x',ylabel='y',title='gauss distribution')
ax.grid()

ax2=plt.subplot(212)
u = np.linspace(140,200,200)
v = np.linspace(5,50,200)
U,V=np.meshgrid(u,v)
s=0
for i in range(len(x)):
   s+= (f(x[i],U,V)-y[i])**2
costs=s
ax2.contour(U,V,costs,100,colors='black',alpha=0.4)
ax2.scatter(p,q,c=p,alpha=0.8,s=8)
plt.show()
print('miu=%.2f sig=%.2f'%(miu,sig))



