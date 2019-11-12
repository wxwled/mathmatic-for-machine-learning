# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:53:25 2019

@author: Administrator
"""

import numpy as np

def random_distribution(num=2):
    random_distribution = np.random.rand(num)
    return random_distribution/random_distribution.sum()

#cross entropy 交叉熵 用基于Q分布编码基于P分布的信息所需的比特数
#特别的 当P==Q时，交叉熵取最小值为H(P)
def entropy(P,Q):
    if np.linalg.norm(P-Q) == 0:
        if np.linalg.norm(P) == 1:
            return 0
    return -1*np.dot(P,np.log2(Q))

#相对熵 用基于Q分布编码基于P分布的信息相比P分布自身所需的额外比特数
def KL_Div(P,Q):
    return entropy(P,Q)-entropy(P,P)

def print_diff(num=2):
    P=np.zeros(num)
    P[0]=1
    Q=random_distribution(num)
    print(Q)
    print(KL_Div(P,Q))
    