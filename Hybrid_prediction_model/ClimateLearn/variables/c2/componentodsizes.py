# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:06:42 2017

@author: Peter Nooteboom
"""
import scipy.io
import numpy as np
import scipy.signal
import functions_percolation_muvar as fw
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from netCDF4 import Dataset

pf8 = "C:/Users/User/Documents/Thesis/observations/weekly/u10_SST2.nc"  
pf9 = "C:/Users/User/Documents/Thesis/observations/weekly/u10_SST3.nc"

data_u_SST =  Dataset(pf8)
data_u_SST2 =  Dataset(pf9)

pf = "C:/Users/User/Documents/Thesis/observations/weekly/SSH-global-reanalysis-phys-001-017-1979-1983.nc"  
pf2 = "C:/Users/User/Documents/Thesis/observations/weekly/SSH-global-reanalysis-phys-001-017-1984-1988.nc"    
pf3 = "C:/Users/User/Documents/Thesis/observations/weekly/SSH-global-reanalysis-phys-001-017-1989-1993.nc"  
pf4 = "C:/Users/User/Documents/Thesis/observations/weekly/SSH-global-reanalysis-phys-001-017-1994-1998.nc"  
pf5 = "C:/Users/User/Documents/Thesis/observations/weekly/SSH-global-reanalysis-phys-001-017-1999-20032.nc"  
pf6 = "C:/Users/User/Documents/Thesis/observations/weekly/SSH-global-reanalysis-phys-001-017-2004-2008.nc"  
pf7 = "C:/Users/User/Documents/Thesis/observations/weekly/SSH-global-reanalysis-phys-001-017-2009-2013.nc"  

data = Dataset(pf) 
data2 = Dataset(pf2) 
data3 = Dataset(pf3) 
data4 = Dataset(pf4)
data5 = Dataset(pf5)
data6 = Dataset(pf6)
data7 = Dataset(pf7)

D = np.concatenate((data['sossheig'][:,1:161:6,::18],data2['sossheig'][:,1:162:6,::18],
                   data3['sossheig'][:,1:161:6,::18],data4['sossheig'][:,1:162:6,::18],
                    data5['sossheig'][:,1:161:6,::18],data6['sossheig'][:,1:162:6,::18],
                    data7['sossheig'][:,1:161:6,::18]),axis=0)
# sea surface height weekly:
D = D[::7,:,:]
 

lagged1 = False

lon = data_u_SST['longitude'][::18]
llon = len(lon)
lat = data_u_SST['latitude'][1:161:6]
llat = len(lat)

D = np.swapaxes(D,0,2)
D = np.swapaxes(D,0,1)

#lengths of both the h ans SST timeseries:
sdata1 =   D.shape[2]#

#The amount of nodes for the SST and h:
size1 = D.shape[0]*D.shape[1]
totsize=size1

D = np.reshape(D,(1,size1,sdata1))
D = scipy.signal.detrend(D)

lenseq = 52 #the length of every timeseries in months
sq = 4. #amount of months shifted 
nslides = 1

#Change sdata1 if you want to change these variables:
spart = int((sdata1-lenseq)/nslides)# #size of one part of the timeserie
lensliding = int(spart/sq)#int(sdata1/sq-lenseq)#3 #amount of times the window is shifted one month ahead.

time = np.linspace(1979+lenseq/52.,1979+(lensliding*sq+lenseq)/52.,lensliding*nslides)

del data_u_SST

nino34 = np.loadtxt('monthlynino34.txt')
nino2 = np.zeros(lensliding)
for i in range(lensliding):
    nino2[i] = nino34[int(i*sq+lenseq-5)/52,1+int((int(sq*i+lenseq-5)%52)/52.*12)]

#lengths of both the h ans SST timeseries:
THRESHOLD = 0.9#0.999
lag = 0
lagged1=True

print 'Threshold: ',THRESHOLD
print 'lenseq: ',lenseq

#Change sdata1 if you want to change these variables:
spart = int((sdata1-lenseq))# #size of one part of the timeserie
lensliding = int(spart/sq)#int(sdata1/sq-lenseq)#3 #amount of times the window is shifted one month ahead.
    

#%% Calculating C_s
#Calculate adjacency matrix (contains all correlations between nodes): 
A = fw.adj_single(lensliding,1,D,sq,totsize,lag,lenseq,lagged1,THRESHOLD,0)
print 0
#calculate the amount of clusters of size 2:
c2 = fw.cs(A,2,lensliding)
# divide by the total amount of nodes to obtain c2:
c2 = c2/float(totsize)

if(THRESHOLD==0.9):
    np.save('c2_h_th09_lenseq{}.pny'.format(lenseq),c2)


#c3 = fw.cs(A,3,lensliding)
#c3 = c3/float(totsize)
#c5 = fw.cs(A,5,lensliding)
#c5 = c5/float(totsize)
#c9 = fw.cs(A,9,lensliding)
#c9 = c9/float(totsize)

print 'Threshold',THRESHOLD

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=1.6)
par1 = host.twinx()

host.set_xlabel("Time (years)",fontsize=25)
host.set_ylabel("$c_s$",fontsize=25)
par1.set_ylabel("Nino34",fontsize=25)
#p2, = par1.plot(time,nino2,linewidth=2.5)
p1, = host.plot(time, c2,linewidth=2.5)
#p7, = par1.plot(time, delta,linewidth=2.5)
#p3, = host.plot(time, c3,linewidth=2.5)
#p4, = host.plot(time, c5,linewidth=2.5)
#p5, = host.plot(time, c9,linewidth=2.5)

#host.legend(handles=[p1,p2], loc=3)
#par1.set_ylim(0.5, 1.8)

plt.draw()
plt.show()

#%%plot a running mean of c2
def running_mean(l, N):
    sum = 0
    result = list( 0 for x in l)

    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)

    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N

    return result

rmc2 = running_mean(c2, 9)
    
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=1.6)
par1 = host.twinx()
host.set_xlabel("Time (years)",fontsize=25)
host.set_ylabel("$c_s$",fontsize=25)
par1.set_ylabel("Nino34",fontsize=25)
p2, = par1.plot(time,nino2,linewidth=2.5)
p1, = host.plot(time, rmc2,linewidth=2.5)
#host.legend(handles=[p1,p2], loc=3)
#par1.set_ylim(0.5, 1.8)
plt.draw()
plt.show()
