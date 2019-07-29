# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:11:57 2019

@author: Administrator
"""
from scipy import optimize
def f(x):
    return x**6/6-3*x**4-2*x**3/3+27*x**2/2+18*x-30

#def d_f(x):
#    return x**5-12*x**3-2*x**2+27*x+18
#
#def update_x(x):
#    return x-f(x)/d_f(x)
##root1=1.063,root2=-3.76
##choose ramdom origin
#x=update_x(1)
#print(x)
##repeat N-R_M
#x=update_x(x)
#print(x)

#using optimize
#bad point
#x0=-2.9
#print(optimize.newton(f,x0))
#error point
 