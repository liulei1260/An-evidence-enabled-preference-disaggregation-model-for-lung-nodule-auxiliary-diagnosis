import numpy as np
import pandas as pd
import os
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
    print(mean,std,result)
    return result

W_j=[0.072,0.499,0.072,0.072,0.072,0.142,0.072]  ##input weight vector
#W_j=[0.11, 0.14,  0.07,  0.07, 0.54,  0.07,  0]
#if len(W_j) != 0:
  #raise Exception("Sorry, no numbers below zero")
A=[0,0.25,0.5,0.75,1]
basepath='C:\\Work\\project\\FangRan'
df = pd.read_csv(os.path.join(basepath,'data_final.csv'), encoding='utf-8')
# x = np.zeros((10,10,6)) y=[0,1,2,3,4,5] y=np.array(y) x[0][0]=y
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

e_i_df=pd.DataFrame(e_i, columns = ['ei_1','ei_2','ei_3','ei_4','ei_5'])

object_v=object_f(e_i,e_i_average)
#print(object_v)

D_star=np.zeros(df.shape[0])
for i in range(df.shape[0]):
    D_star[i]=np.argmax(e_i[i])+1

V_D=np.zeros(df.shape[0])
for i in range(df.shape[0]):
    V_D[i]=(D_star[i]-1)/4

Ar_i=np.zeros(df.shape[0])
for i in range(df.shape[0]):
    Ar_i[i]=1-abs(V_D[i]-df.iloc[i,8])

e_i_df['D_star']=D_star.tolist()
e_i_df['Ar_i']=Ar_i.tolist()


