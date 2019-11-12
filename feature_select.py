# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 09:36:31 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston


dataset = load_boston()
#print(dataset.data[0])
#print(dataset.target[0])
#print(dataset.feature_names)

train_x = pd.DataFrame(dataset.data,columns=dataset.feature_names)
train_y = pd.DataFrame(pd.qcut(dataset.target,5,labels=False),columns=['price'])


#1 信息增益(互信息)H(Y)-H(Y|X) 离散化数据可用
#最大信息系数
from minepy import MINE
def mic(x,y):
    m = MINE()
    m.compute_score(x,y)
    return m.mic()

def mic_score(features,label):
    mic_dict={}
    for feature in list(features.columns):
        mic_dict[feature] = mic(features[feature],label['price'])
    mic_df=pd.Series(mic_dict)
    mic_df=pd.DataFrame(mic_df,columns=['mic'])
    return mic_df


#2 相关系数以及特征本身的方差
def corr_and_var_score(features,label):
    train = pd.concat([features,label],axis=1)
    corr_df= train.corr()['price']
    var_df = train.var()
    statistic = pd.concat([corr_df,var_df],axis=1)
    statistic.columns = ['corr','var']
    return statistic.iloc[:-1,:]

#3 卡方检验 非参数检验
def chi2_score(features,label):
    """
    features: DataFrame  
    label: DataFrame 
    """
    #法1
#    from sklearn.feature_selection import chi2
#    from sklearn.feature_selection import SelectKBest
#    model=SelectKBest(chi2, k=2)
#    model.fit_transform(train_x, train_y)
#    return model.scores_
    #法2
    from sklearn import preprocessing
    #将整数标签转换成二值矩阵（独热码/哑编码），也是筛选矩阵
    #通过点积进行分类求和，得到实际特征标签总值矩阵
    Y = preprocessing.LabelBinarizer().fit_transform(train_y)
    observed=np.dot(Y.T,features)
    
    #理论特征标签总值矩阵分布与标签分布保持一致即秩1矩阵，可以通过矩阵乘法生成
    label_distribution=np.array([Y.mean(axis=0)])#axis=0表示跨行运算
    features_sum=np.array([train_x.sum()])
    #假设特征与标签相互独立，那么特征标签总值矩阵如下
    expected = np.dot(label_distribution.T,features_sum)
    
    #计算chi2=sum((observed-expected)^2/expected)#实际与理论的相对差距
    chi2=pd.DataFrame((np.square(expected-observed)/expected).sum(axis=0))
    chi2.columns=['chi2']
    chi2.index=features.columns
    return chi2
    
print(pd.concat([mic_score(train_x,train_y),chi2_score(train_x,train_y),corr_and_var_score(train_x,train_y)],axis=1))   
    
    
    