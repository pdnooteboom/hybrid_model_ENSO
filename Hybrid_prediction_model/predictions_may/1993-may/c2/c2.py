# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 07:44:40 2017

@author: Peter Nooteboom
"""

from datetime import timedelta
from netCDF4 import Dataset
import numpy as np
import pandas as pd 
import scipy.io
import scipy.signal
import functions_percolation_muvar as fw
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

pf0 = '/Users/nooteboom/Documents/MA/Reproduce_paper_results/data_pred_may/prediction_may/SSH_1993-may2017/'

#Load the SSH data from janari 1993 up to april 2017
pf = 'datasetSSH1.nc'
data = Dataset(pf0 + pf)
pf = 'datasetSSH2.nc'
data1 = Dataset(pf0 + pf)
pf = 'datasetSSH3.nc'
data2 = Dataset(pf0 + pf)
pf = 'datasetSSH4.nc'
data3 = Dataset(pf0 + pf)
pf = 'datasetSSH5.nc'
data4 = Dataset(pf0 + pf)
pf = 'datasetSSH6.nc'
data5 = Dataset(pf0 + pf)
pf = 'datasetSSH7.nc'
data6 = Dataset(pf0 + pf)
pf = 'datasetSSH8.nc'
data7 = Dataset(pf0 + pf)
#Concatenate the loaded data to D for the coarse grid only
D = np.concatenate((data['sla'][:,::5,::18],data1['sla'][:,::5,::18],data2['sla'][:,::5,::18],data3['sla'][:,::5,::18],data4['sla'][:,::5,::18],data5['sla'][:,::5,::18],data6['sla'][:,::5,::18],data7['sla'][:,::5,::18]),axis=0)
timeD = np.concatenate((data['time'][:],data1['time'][:],data2['time'][:],data3['time'][:],data4['time'][:],data5['time'][:],data6['time'][:],data7['time'][:]),axis=0)

#Load the more recent SSH data (up to may)
pfr = '/Users/nooteboom/Documents/MA/Reproduce_paper_results/data_pred_may/prediction_may/SSH_1993-may2017/SSH_data_1993-pres/'

data= Dataset(pfr + 'nrt_global_allsat_msla_h_20140408_20140414.nc/nrt_global_allsat_msla_h_20140408_20140414.nc')
dtime =  timedelta(days=6)

times = pd.date_range('2016-10-11', periods=232, freq='1440min') #until 30-05-2017

for i in range (len(times)):
    time = times[i].strftime("%Y%m%d")#.astype(int)
    time2 =(times[i]+dtime).strftime("%Y%m%d")#.astype(int)
    data = Dataset(pfr + 'nrt_global_allsat_msla_h_'+time+'_'+time2+'.nc/nrt_global_allsat_msla_h_'+time+'_'+time2+'.nc')

    #append D with the recent data    
    D = np.append(D,data['sla'][:,280:439:5,559:1119:18],axis=0)
    timeD = np.append(timeD,data['time'][:])    

#Get axis of D right and take weekly in stead of daily data
#D has shape (longitudes, latitudes, time)
D = np.swapaxes(D,0,2)
D = np.swapaxes(D,0,1)
D = D[:,:,::7]

#longitudes and latitudes of the coarse grid
lon = data['lon'][559:1119:18]
llon = len(lon)
lat = data['lat'][280:439:5]
llat = len(lat)

sdata1 =   D.shape[2]#length of time series

#The amount of nodes for the SST and h:
size1 = D.shape[0]*D.shape[1]
totsize=size1

#reshape/flatten D
D = np.reshape(D,(1,size1,sdata1))
D = scipy.signal.detrend(D) #detrend the time series
#%%
lenseq = 52 #the length of every timeseries in months
sq = 4. #amount of months shifted 

#Change sdata1 if you want to change these variables:
spart = int((sdata1-lenseq))#size of one part of the timeserie
lensliding = int(spart/sq) #amount of times the window is shifted one month ahead.

#lengths of both the h ans SST timeseries:
THRESHOLD = 0.9#threshold of correlation for link
lagged1=True
lag = 0

print 'Threshold: ',THRESHOLD
print 'lenseq: ',lenseq
#%% Calculating C_s
A = fw.adj_single(lensliding,1,D,sq,totsize,lag,lenseq,lagged1,THRESHOLD,0)

c2 = fw.cs(A,2,lensliding)
c2 = c2/float(totsize)

if(THRESHOLD==0.9):
    np.save('c2_h_th09_lenseq{}.pny'.format(lenseq),c2)


starttime = timeD[0]/365.+1950
time = np.zeros(len(c2))
for i in range(len(c2)):
    time[i] = starttime+1+4/52.*i 
np.save('timec2.npy',time)

print 'Threshold',THRESHOLD

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=1.6)
#par1 = host.twinx()
host.set_xlabel("Time (years)",fontsize=25)
host.set_ylabel("$c_s$",fontsize=25)
#p2, = par1.plot(time,nino2,linewidth=2.5)
p1, = host.plot(time, c2,linewidth=2.5)
plt.draw()
plt.show()

#%%
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
#par1 = host.twinx()
host.set_xlabel("Time (years)",fontsize=25)
host.set_ylabel("$c_s$",fontsize=25)
#par1.set_ylabel("Nino34",fontsize=25)
#p2, = par1.plot(time,nino2,linewidth=2.5)
p1, = host.plot(time, rmc2,linewidth=2.5)
#host.legend(handles=[p1,p2], loc=3)
#par1.set_ylim(0.5, 1.8)

plt.draw()
plt.show()
