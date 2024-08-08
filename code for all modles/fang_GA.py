import numpy as np
import pandas as pd
import os
from scipy import optimize
from scipy.optimize import LinearConstraint
from geneticalgorithm import geneticalgorithm as ga
import math

def position_check(x,A,hi=None):
    lo=0
    if hi is None:
        hi = len(A)
    while lo < hi:
        mid = (lo+hi)//2
        if x < A[mid]:
            hi = mid
        else:
            lo = mid+1
    y=lo-1
    if x==A[y]:
        return y, y
    else:
        return y, y+1

def combine(e_f,e_b):
    ## e_f,e_b are narray with size of (8,)
    # k=[1-(e_f[0]*e_b[0]+e_f[1]*e_b[0]+e_f[2]*e_b[0]+e_f[3]*e_b[0]+e_f[4]*e_b[0]+
    #       e_f[0]*e_b[1]+e_f[1]*e_b[1]+e_f[2]*e_b[1]+e_f[3]*e_b[1]+e_f[4]*e_b[1]+
    #       e_f[0]*e_b[2]+e_f[1]*e_b[2]+e_f[2]*e_b[2]+e_f[3]*e_b[2]+e_f[4]*e_b[2]+
    #       e_f[0]*e_b[3]+e_f[1]*e_b[3]+e_f[2]*e_b[3]+e_f[3]*e_b[3]+e_f[4]*e_b[3]+
    #       e_f[0]*e_b[4]+e_f[1]*e_b[4]+e_f[2]*e_b[4]+e_f[3]*e_b[4]+e_f[4]*e_b[4])]^(-1)
    k=(1-(e_f[1]*e_b[0]+e_f[2]*e_b[0]+e_f[3]*e_b[0]+e_f[4]*e_b[0]+
          e_f[0]*e_b[1]+e_f[2]*e_b[1]+e_f[3]*e_b[1]+e_f[4]*e_b[1]+
          e_f[0]*e_b[2]+e_f[1]*e_b[2]+e_f[3]*e_b[2]+e_f[4]*e_b[2]+
          e_f[0]*e_b[3]+e_f[1]*e_b[3]+e_f[2]*e_b[3]+e_f[4]*e_b[3]+
          e_f[0]*e_b[4]+e_f[1]*e_b[4]+e_f[2]*e_b[4]+e_f[3]*e_b[4]))**(-1)
    result=np.zeros(len(e_f))
    for i in range(5):
        result[i]=k*(e_f[i]*e_b[i]+e_f[i]*e_b[7]+e_f[7]*e_b[i])  ##7 is len(e_f)-1
    result[5]=k*(e_f[5]*e_b[5]+e_f[5]*e_b[6]+e_f[6]*e_b[5])
    result[6]=k*(e_f[6]*e_b[6])
    result[7]=result[5]+result[6]
    return result

def object_f(e1,e2):
    std=0
    e4=np.zeros(e1.shape[0])
    for i in range(e1.shape[0]):
        e3=abs(e1[i]-e2[i])
        temp=e3[0]*(0.25*e3[1]+0.5*e3[2]+0.75*e3[3]+e3[4])\
             +e3[1] * (0.25 * e3[2] + 0.5 * e3[3] + 0.75 * e3[4])\
             +e3[2]* (0.25 * e3[3] + 0.5 * e3[4])\
             +e3[3]* (0.25 * e3[4] )
        e4[i]=temp
    mean=sum(e4)/e1.shape[0]
    for i in range(e1.shape[0]):
        temp=(e4[i]-mean)**2
        std=std+temp
    std=math.sqrt(std/(e1.shape[0]-1))
    result=mean+2*std
    print(mean, std, result)
    return result


def differnce(W_j):
    print(W_j)
    W_j=W_j/100
    A = [0, 0.25, 0.5, 0.75, 1]
    e_ij=df.iloc[:, 0:len(W_j)]  #dataframe
    e_ij_1=np.zeros((df.shape[0],e_ij.shape[1],len(A)))   #narray  e_ij->e_ij_1 DF->narray
    for i in range(df.shape[0]):
        for j in range (len(W_j)):
            lo,up=position_check(e_ij.iloc[i,j],A,hi=None)
            if lo==up:
                e_ij_1[i][j][lo]=1
            else:
                e_ij_1[i][j][lo]=(A[up]-e_ij.iloc[i,j])/(A[up]-A[lo])
                e_ij_1[i][j][up] = (e_ij.iloc[i, j]-A[lo]) / (A[up] - A[lo])


    e_ij_2=np.zeros((df.shape[0],e_ij.shape[1],len(A)+3))    #narray  e_ij_1->e_ij_2 narray->narray
    for i in range(df.shape[0]):
        for j in range(len(W_j)):
            if df['reliability'][i] == 1:
                w_ij = W_j[j] / (1 + W_j[j] - 0.99)
            else:
                w_ij = W_j[j] / (1 + W_j[j] - df['reliability'][i])  # df['reliability'][i] is reliability of each row
            for k in range(len(A)):
                e_ij_2[i][j][k]=w_ij*e_ij_1[i][j][k]
            e_ij_2[i][j][len(A)]= w_ij*(1-np.sum(e_ij_1[i][j]))
            e_ij_2[i][j][len(A)+1] = 1-w_ij
            e_ij_2[i][j][len(A) + 2]=e_ij_2[i][j][len(A)]+e_ij_2[i][j][len(A)+1]

    ## combine process
    e_ij_3=np.zeros((df.shape[0],len(A)+3)) #combined result

    for i in range(df.shape[0]):
        e_ij_I=e_ij_2[i][0]
        for j in range(1,7):
            temp=combine(e_ij_I,e_ij_2[i][j])
            e_ij_I=temp
        e_ij_3[i]=e_ij_I

    e_i=np.zeros((df.shape[0],len(A)))  ##e xing zhi

    for i in range(df.shape[0]):
        for j in range (5):
            e_i[i][j]=e_ij_3[i][j]/(1-e_ij_3[i][6])

    e_i_average=np.zeros((df.shape[0],len(A)))  ##e xing zhi average

    for i in range(df.shape[0]):
        lo,up=position_check(df.iloc[i,8],A,hi=None)
        if lo==up:
            e_i_average[i][lo]=1
        else:
            e_i_average[i][lo]=(A[up]-df.iloc[i,8])/(A[up]-A[lo])
            e_i_average[i][up] = (df.iloc[i,8]-A[lo]) / (A[up] - A[lo])

    object_v=object_f(e_i,e_i_average)
    if sum(W_j)!=1:
        object_v=object_v+1
    print(object_v)
    return object_v

def cons1(args):
    return sum(args)-1

#W_j=[0.1,0.1,0.1,0.2,0.1,0.2,0.2]  ##input weight vector
W_j=[10,10,10,20,10,20,20]
#if len(W_j) != 0:
  #raise Exception("Sorry, no numbers below zero")

basepath='C:\\Work\\project\\FangRan'
global df
df= pd.read_csv(os.path.join(basepath,'evidence - Copy.csv'), encoding='utf-8')
#result=differnce(W_j,df)
#bnds = ((0, 100), (0, 100),(0, 100),(0, 100),(0, 100),(0, 100),(0, 100))
#bnds = ((0, 1), (0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1))
#bnds=np.array([[0, 1],[0, 1],[0,1],[0, 1],[0, 1],[0, 1],[0, 1]])
bnds=np.array([[0, 100],[0, 100],[0,100],[0, 100],[0, 100],[0, 100],[0, 100]])

algorithm_param = {'max_num_iteration': 10,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=differnce,dimension=7,variable_type='int',variable_boundaries=bnds,
         algorithm_parameters=algorithm_param)
model.run()




