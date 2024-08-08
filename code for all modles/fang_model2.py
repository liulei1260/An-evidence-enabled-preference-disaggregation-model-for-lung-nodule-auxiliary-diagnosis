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
        e4[i]=abs(e1[i]-e2[i])
    mean=sum(e4)/e1.shape[0]
    for i in range(e1.shape[0]):
        temp=(e4[i]-mean)**2
        std=std+temp
    std=math.sqrt(std/(e1.shape[0]-1))
    result=mean+2*std
    return result

W_j=[0.48028,0.0,0.42867,0.0713,0.00002,0.00002,0.01971]  ##input weight vector
#W_j=[0.11, 0.14,  0.07,  0.07, 0.54,  0.07,  0]
#if len(W_j) != 0:
  #raise Exception("Sorry, no numbers below zero")
A=[0,0.25,0.5,0.75,1]
basepath='C:\\Work\\project\\FangRan'
df = pd.read_csv(os.path.join(basepath,'data_final.csv'), encoding='utf-8')
# x = np.zeros((10,10,6)) y=[0,1,2,3,4,5] y=np.array(y) x[0][0]=y
e_ij=df.iloc[:, 0:len(W_j)]  #dataframe

v_ei=np.zeros(df.shape[0])
for i in range(df.shape[0]):
    temp=0
    for j in range (len(W_j)):
        w_ij=W_j[j]*df['reliability'][i]
        temp=temp+w_ij*e_ij.iloc[i][j]
    v_ei[i]=temp

object_v=object_f(v_ei,df['reliability'].values)

#e_i_df=pd.DataFrame(v_ei, columns = ['v_ei'])

#print(object_v)

D_star=np.zeros(df.shape[0])
for i in range(df.shape[0]):
    xxx=[]
    xxx=[abs(x - v_ei[i]) for x in A]
    D_star[i]=np.argmin(xxx)+1


Ar_i=np.zeros(df.shape[0])
for i in range(df.shape[0]):
    Ar_i[i]=1-abs(v_ei[i]-df.iloc[i,8])

#e_i_df['D_star']=D_star.tolist()
#e_i_df['Ar_i']=Ar_i.tolist()
e_i_df=pd.DataFrame({'v_ei': v_ei, 'D_star': D_star,'Ar_i': Ar_i})