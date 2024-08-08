
import numpy as np
import pandas as pd
import os
from scipy import optimize
from scipy.optimize import LinearConstraint
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import math
import statistics
matplotlib.use('Agg')

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
    #print(mean, std, result)
    return result


def differnce(W_j):
    #print(W_j)
    #print(statistics.stdev(W_j))
    #W_j=W_j/100
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
    #print(object_v)
    # temp = np.array(W_j)
    # temp = np.append(temp,object_v )
    # global indexx
    # temp = np.append(temp, indexx)
    # indexx=indexx+1
    # global solution
    # solution = np.append(solution, [temp], axis=0)
    return object_v

def cons1(args):
    return sum(args)-1
    #return sum(args)-100

def callbackF(Xi,status):
    temp=np.array(Xi)
    temp=np.append(temp,differnce(Xi))
    global indexx
    temp=np.append(temp,indexx)
    indexx = indexx + 1
    global solution
    solution = np.append(solution, [temp], axis=0)

def object_f2(e1,e2):
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


def differnce2(W_j):
    #print(W_j)
    #print(statistics.stdev(W_j))
    #W_j=W_j/100
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

    object_v=object_f2(e_i,e_i_average)
    #print(object_v)
    # temp = np.array(W_j)
    # temp = np.append(temp,object_v )
    # global indexx
    # temp = np.append(temp, indexx)
    # indexx=indexx+1
    # global solution
    # solution = np.append(solution, [temp], axis=0)
    return object_v

W_j=[0.1,0.1,0.1,0.2,0.1,0.2,0.2]  ##input weight vector
#W_j=[10,10,10,20,10,20,20]
#if len(W_j) != 0:
  #raise Exception("Sorry, no numbers below zero")
solution = np.empty((0,9), int)
indexx=0

global df
basepath='C:\\Work\\project\\FangRan'
df = pd.read_csv(os.path.join(basepath,'evidence - Copy1.csv'), encoding='utf-8')

cons_trant=1    #contral of the constraint of bound
if cons_trant==0:
    bnds = ((0, 1), (0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1))
else:
    low_bound = 1 / (2 * len(W_j))
    bnds = ((low_bound, 0.5), (low_bound, 0.5), (low_bound, 0.5), (low_bound, 0.5), (low_bound, 0.5), (low_bound, 0.5), (low_bound, 0.5))

#result=differnce(W_j,df)
#bnds = ((0, 100), (0, 100),(0, 100),(0, 100),(0, 100),(0, 100),(0, 100))
#bnds = ((0, 1), (0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1))
#bnds = ((1/14, 0.5), (1/14, 0.5),(1/14, 0.5),(1/14, 0.5),(1/14, 0.5),(1/14, 0.5),(1/14, 0.5))
#bnds = ((0.1, 0.9), (0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9))
# cons=({'type': 'eq',
#        'fun': cons1},
#         {'type':'eq','fun': lambda x : max([x[i]-int(x[i]) for i in range(len(x))])})
cons=({'type': 'eq',
       'fun': cons1})
#optimize_para=optimize.minimize(differnce,W_j,args=(df,),method='SLSQP',bounds=bnds,constraints=cons,options={'maxiter':20,'disp': False})

# cons_cobyla=({'type': 'eq',
#        'fun': cons1},
#         {'type':'ineq','fun': lambda x : (x[i] for i in range(len(x)))},
#              {'type':'ineq','fun': lambda x : (1-x[i] for i in range(len(x)))})
A = np.array([[1, 1,1,1,1,1,1]])
b = np.array([1])
lincon = LinearConstraint(A, b, b)
optimize_para=optimize.minimize(differnce,W_j,method='trust-constr',bounds=bnds,constraints=lincon,callback=callbackF, options={'maxiter':100,'disp': False})
print('The solution is:')
print(optimize_para.x)
print(optimize_para.fun)
differnce2(optimize_para.x)

pca = PCA(n_components=2)
df_so=solution[:,:7]
X_train_pca = pca.fit_transform(df_so)
df_so=np.concatenate((df_so,solution[:,7].T[:, None]),axis=1)
columns_name=list(df.columns)[:7]
columns_name=np.append(columns_name, 'fitness')
df_so=pd.DataFrame(df_so, columns = columns_name)

#X, Y = np.meshgrid(X_train_pca[:,0],X_train_pca[:,1])
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(X_train_pca[:,0], X_train_pca[:,1], solution[:,7], linewidth=0.2, antialiased=True )

# color_dimension = solution[:,7]# change to desired fourth dimension
# minn, maxx = color_dimension.min(), color_dimension.max()
# norm = matplotlib.colors.Normalize(minn, maxx)
# m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
# m.set_array([])
# fcolors = m.to_rgba(color_dimension)
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(X_train_pca[:,0], X_train_pca[:,1], solution[:,7], facecolors=fcolors, vmin=minn, vmax=maxx)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot3D(X_train_pca[:,0], X_train_pca[:,1], solution[:,7], 'red')
ax.scatter3D(X_train_pca[:,0], X_train_pca[:,1], solution[:,7], c=solution[:,7], cmap='Greens');
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('MDD')
ax.xaxis.label.set_color('blue')
ax.yaxis.label.set_color('blue')
ax.zaxis.label.set_color('blue')
Z = np.outer(solution[:,7].T, solution[:,7])
X, Y = np.meshgrid(X_train_pca[:,0], X_train_pca[:,1])
color_dimension = Z # change to desired fourth dimension
minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('MDD')
ax.xaxis.label.set_color('blue')
ax.yaxis.label.set_color('blue')
ax.zaxis.label.set_color('blue')

if cons_trant==0:
    plt.savefig('plot_fmin.png')
    df_so.to_csv('fmin_dfsolution.csv', index=False)
else:
    plt.savefig('plot_fmin_con.png')
    df_so.to_csv('fmin_dfsolution_con.csv', index=False)
