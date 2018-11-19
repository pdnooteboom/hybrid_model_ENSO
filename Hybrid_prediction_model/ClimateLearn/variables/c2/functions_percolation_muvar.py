# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 12:46:25 2017
Functions used in cluster sizes calculations
@author: Peter Nooteboom
"""
import  numpy as np
import networkx as nx
from numba import jit
import sys

def cross_corr(X, Y): #returns the correlation between sequence X and Y        
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
        w = cross_corr0(X,Y,s)
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
    #Return adjacency matrix of network
    A = np.empty((lensliding,N,totsize,totsize),dtype=bool)

    for l in range(lensliding):
        t = int(start+sq*l)
        for i in range(N):   
            if(not lagged1):
                X = D[i,:, t+lag:lenseq+t]
                Y = D[i,:, t:lenseq+t-lag]
            else:
                X = D[i,:, t:lenseq+t-lag]
                Y = D[i,:, t+lag:lenseq+t]        
                
            s = np.shape(X)
            similarity_matrix = np.abs(cross_corr0(X, Y, s))
            for j in range(totsize):
                for k in range(j):
                    #link two nodes if the correlation exceeds the threshold value
                    #and if j is not k, to prevent a node to connect to itslef
                    if(similarity_matrix[j][k]>THRESHOLD and j!=k):
                        A[l][i][j][k]=True 
                        A[l][i][k][j]=True
    return A
    

@jit    
def cs(A,s,lensliding):
    #Returns amount of clusters of size s in matrix A
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
    