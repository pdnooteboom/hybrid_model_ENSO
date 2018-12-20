# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 12:46:25 2017

@author: Peter Nooteboom
"""
import  numpy as np
import networkx as nx
from numba import jit
import sys

def cross_corr(X, Y): #returns the correlation between sequence X and Y
#    meanX = 0.
#    meanY = 0.
#    meanXY = 0.
#    for i in range(len(X)):
        
    cc = (np.mean((X*Y))-np.mean(X)*np.mean(Y))/(np.std(X)*np.std(Y))
    return np.abs(cc)

def cross_corr0(X, Y, s): #returns the correlation between sequence X and Y
        
    if np.shape(Y) != np.shape(X):
        sys.exit("X and Y must have the same size.")
        
    # Create cross correlation array
    cc = np.empty((s[0], s[0]), dtype = 'float32')
        
    # Use numpy corrcoef to compute correlations and select upper right block of the correlation matrix
    cc[:, :] = np.ma.corrcoef(X, Y)[0:s[0], s[0]:2*s[0]]
        
    return cc    

@jit
def delta(W,totsize):
    lens = 100
    S = np.zeros(lens)
    Sdifmax = 0.# = np.zeros(totsize-1)
    weights = np.linspace(1,0,lens)#np.zeros(int(totsize*totsize/2.))
    
    G = nx.from_numpy_matrix(np.zeros((totsize,totsize)))
    
    for i in range(lens):
        
        x,y = np.where(W>weights[i])
        
        for j in range(len(x)):
            G.add_edge(x[j],y[j]) 
            W[x[j],y[j]]=0
            W[y[j],x[j]]=0
            
        S[i] = nx.number_of_nodes(max(nx.connected_component_subgraphs(G), key=len)) 
        if(S[i]-S[i-1]>Sdifmax):
            Sdifmax = S[i]-S[i-1]  
    
    return Sdifmax/float(totsize)

#@jit
def weightA0(T,totsize,llat,llon):
    W = np.zeros((totsize,totsize))
    #T = np.swapaxes(T,0,1)
    for tau in range(30):
        X = T[:,tau:]
        Y = T[:,0:T.shape[1]-tau]
        s = np.shape(X)
        w = cross_corr0(X,Y,s)#np.maximum(cross_corr0(X,Y,s),cross_corr0(Y,X,s))
        #print 'w',w
        W = np.maximum(W,w)
        
    return W
 
@jit
def weightA1(T,totsize,llat,llon):
    W = np.zeros((totsize,totsize))
    for n in range(totsize):    
        for k in range(totsize):
            maxw = 0.
            for tau in range(3):
                X = T[tau:,n]
                Y = T[:len(T)-tau,k]

                w = cross_corr(X,Y)
                #print 'w',w
                if(w>maxw):
                    
                    maxw = w
            if(maxw>W[k,n]):
                W[n,k] = maxw
                W[k,n] = maxw
    return W
    
@jit
def S(W,totsize):
    
    lens = 100
    S = np.zeros(lens)
    weights = np.linspace(1,0,lens)#np.zeros(int(totsize*totsize/2.))
    
    G = nx.from_numpy_matrix(np.zeros((totsize,totsize)))
    
    for i in range(lens):
        
        x,y = np.where(W>weights[i])
        
        for j in range(len(x)):
            G.add_edge(x[j],y[j]) 
            W[x[j],y[j]]=0
            W[y[j],x[j]]=0
            
        S[i] = nx.number_of_nodes(max(nx.connected_component_subgraphs(G), key=len)) 
    
    return S,weights
    
#@jit
def S2(A):
    # Returns the size of the largest component that is already thresholded
    #from adjacency matrix A
    
    G = nx.from_numpy_matrix(A)          
    S = nx.number_of_nodes(max(nx.connected_component_subgraphs(G), key=len)) 
    
    return S   
    
@jit
def adj_single(lensliding,N,D,sq,totsize,lag,lenseq,lagged1,THRESHOLD,start):
    A = np.empty((lensliding,N,totsize,totsize),dtype=bool)

    for l in range(lensliding):
        t = int(start+sq*l)
        for i in range(N):
            #data = np.zeros((totsize,totsize),dtype=bool)
            #if(lenseq+lensliding>sdata1):
            #    print 'Watch out! Too much sliding, time dimensions of the data not large enough'
            #else:       
            if(not lagged1):
                X = D[i,:, t+lag:lenseq+t]
                Y = D[i,:, t:lenseq+t-lag]
            else:
                X = D[i,:, t:lenseq+t-lag]
                Y = D[i,:, t+lag:lenseq+t]        
                
            s = np.shape(X)
            similarity_matrix = np.abs(cross_corr0(X, Y, s))
            for j in range(totsize):#nsize_SST,totsize-1):
                for k in range(j):#nsize_SST):
                    #link two nodes if the correlation exceeds the threshold value
                    #and if j is not k, to prevent a node to connect to itslef
                    if(similarity_matrix[j][k]>THRESHOLD and j!=k):
                        A[l][i][j][k]=True #data[j][k]=True
                        A[l][i][k][j]=True #data[k][j]=True
            #A[l][i] = data
    return A
    
@jit
def adj2(l,N,D,D2,sq,totsize,lag,lenseq,lagged1,THRESHOLD,start):
    A = np.empty((totsize,totsize),dtype=bool)#lensliding,N,
    #for l in range(lensliding):
    t = int(start+sq*l)
        #for i in range(N):     
    if(not lagged1):
        X = np.concatenate((D[0,:, t+lag:lenseq+t],D2[0,:, t:lenseq+t-lag]),axis=0)
        Y = np.concatenate((D[0,:, t+lag:lenseq+t],D2[0,:, t:lenseq+t-lag]),axis=0)
    else:
        X = np.concatenate((D[0,:, t:lenseq+t-lag],D2[0,:, t+lag:lenseq+t]),axis=0)
        Y = np.concatenate((D[0,:, t:lenseq+t-lag],D2[0,:, t+lag:lenseq+t]),axis=0)        
        
    s = np.shape(X)
    similarity_matrix = np.abs(cross_corr0(X, Y, s))
    for j in range(totsize):
        for k in range(j):
            #link two nodes if the correlation exceeds the threshold value
            #and if j is not k, to prevent a node to connect to itslef
            if(similarity_matrix[j][k]>THRESHOLD and j!=k):
                A[j][k]=True 
                A[k][j]=True 
    return A

@jit    
def cs(A,s,lensliding):
    cs = np.zeros(lensliding)
    for i in range(lensliding):
        G = nx.from_numpy_matrix(A[i,0])
        cn = list(nx.connected_components(G))
        for j in range(len(cn)):
            if(len(cn[j])==s):
                cs[i] += 1*s
    return cs

@jit(nopython=True) 
def c2_local(A,lensliding):
    totsize = A.shape[2]
    #returns an adjacency matrix with ones on the positions that are part 
    #of a cluster of size s, dependent on the time (or lensliding)
    cs_local = np.zeros((lensliding,totsize))
    for l in range(lensliding):
        for i in range(totsize):
            n=0
            for j in range(totsize):
                if(A[l,0,i,j]==1):
                    n += 1
                    k = j
            if(n==1):
                p = 0
                for m in range(totsize):
                    if(A[l,0,m,k]==1):
                        p += 1
            if(n==1 and p==1):
                cs_local[l,i] = 1.
    return cs_local

@jit(nopython=True) 
def degree(A,lensliding):
    totsize = A.shape[2]
    #returns an adjacency matrix with ones on the positions that are part 
    #of a cluster of size s, dependent on the time (or lensliding)
    degree = np.zeros((lensliding,totsize))
    for l in range(lensliding):
        for i in range(totsize):
            for j in range(totsize):
                if(A[l,0,i,j]==1):
                    degree[l,i] += 1 
    return degree
    
@jit    
def cs0(A,s,lensliding,i):
    cs = 0
    G = nx.from_numpy_matrix(A)
    cn = list(nx.connected_components(G))
    for j in range(len(cn)):
        if(len(cn[j])==s):
            cs += 1*s
    return cs
    
@jit(nopython=True)
def cross_cs(A,s,size1):
    #count which nodes have only one connection
    cs = 0 #np.zeros(lensliding)
#    for l in range(lensliding):
    for j in range(size1):
        n = 0
        for i in range(size1):
            n += A[j+size1,i]
        if(n==s):
            cs += 1
    return cs
    
@jit(nopython=True)
def cross_local_cs(A,s,totsize,firstnet):
    #count which nodes have only 's' connections with the other network
    size1 = totsize/2
    cs = np.zeros(size1)
#    for l in range(lensliding):
    if(not firstnet):
        for j in range(size1):
            n = 0
            for i in range(size1):
                n += A[j+size1,i]
            if(n==s):
                cs[j] += 1
    else:
        for j in range(size1):
            n = 0
            for i in range(size1):
                n += A[j,i+size1]
            if(n==s):
                cs[j] += 1        
    return cs
    
@jit    
def cross_cs2(s,size1,lensliding,D,D2,sq,lag,lenseq,lagged1,THRESHOLD):    
    c2 = np.zeros(lensliding)
    for l in range(lensliding):
        A = adj2(l,1,D,D2,sq,size1*2,lag,lenseq,lagged1,THRESHOLD,0)
        
        c2[l] = cross_cs(A,s,size1)/float(size1)
    return c2
  
@jit(nopython=True)
def cross_cs_2(A,s,size1):
    cs = 0 #np.zeros(lensliding)
#    for l in range(lensliding):
    for j in range(size1):
        n = 0
        points = []
        for i in range(size1):
            n += A[j+size1,i]
            points.append(i)
        if(n==s):
            cs += 1
    k=0
    for j in range(len(points)):
        k += A[j+size1,points[i]]  
    if(k!=s and k!=0):
        cs = cs - 1          

    return cs
              
@jit    
def cross_cs2_2(s,size1,lensliding,D,D2,sq,lag,lenseq,lagged1,THRESHOLD):    
    c2 = np.zeros(lensliding)
    for l in range(lensliding):
        A = adj2(l,1,D,D2,sq,size1*2,lag,lenseq,lagged1,THRESHOLD,0)
        
        c2[l] = cross_cs_2(A,s,size1)/float(size1)
    return c2
                   
@jit(nopython=True)
def cross_cs_h(A,s,size1):
    cs = 0 #np.zeros(lensliding)
#    for l in range(lensliding):
    for i in range(size1):
        n = 0
        for j in range(size1):
            n += A[j+size1,i]
        if(n==s):
            cs += 1
    return cs
    
@jit    
def cross_cs2_h(s,size1,lensliding,D,D2,sq,lag,lenseq,lagged1,THRESHOLD):    
    c2 = np.zeros(lensliding)
    for l in range(lensliding):
        A = adj2(l,1,D,D2,sq,size1*2,lag,lenseq,lagged1,THRESHOLD,0)
        
        c2[l] = cross_cs_h(A,s,size1)/float(size1)
    return c2
    
@jit    
def c2_2net(s,size1,lensliding,D,D2,sq,lag,lenseq,lagged1,THRESHOLD):    
    c2 = np.zeros(lensliding)
    for l in range(lensliding):
        A = adj2(l,1,D,D2,sq,size1*2,lag,lenseq,lagged1,THRESHOLD,0)
#        import matplotlib.pylab as plt
#        plt.imshow(A,interpolation='none')
#        plt.show()
        
        c2[l] = cs0(A,s,size1,l)/float(size1)
    return c2