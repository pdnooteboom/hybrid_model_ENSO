# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:45:44 2017

@author: Peter Nooteboom
"""

import numpy as np
import matplotlib.pylab as plt
import scipy.signal
from netCDF4 import Dataset
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from datetime import datetime, timedelta

import functions_WWB as fwb

root = '/Users/nooteboom/Documents/MA/running_predictions/prediction_december2018/data/'
root2 = '/Users/nooteboom/Documents/MA/running_predictions/prediction_december2018/data/'

pf1 = root + 'uwnd.mon.mean.nc'
pf2 = root + 'wspd.mon.mean.nc'

datau = Dataset(pf1)
dataw = Dataset(pf2)
#for u
lon = datau['lon'][(datau['lon'][:]>=140) & (datau['lon'][:]<=280)]
lat = datau['lat'][(datau['lat'][:]<=20) & (datau['lat'][:]>=-20)]
ulon, ulat = np.meshgrid(lon, lat)

Y1 = 540 #372# choose 540 for the year 1993, 373 for year 1979
Y2 = 10**8 # 792 #792 for year 2014, 10**8 for present

try:
    print 'first date of wind NCAR dataset: ', datetime(1800, 1,1) + timedelta(hours=datau['time'][Y1])
    print 'last date of wind NCAR dataset: ', datetime(1800, 1,1) + timedelta(hours=datau['time'][Y2])
except:
    print 'first date of wind NCAR dataset: ', datetime(1800, 1,1) + timedelta(hours=datau['time'][-1])
    print 'last date of wind NCAR dataset: ', datetime(1800, 1,1) + timedelta(hours=datau['time'][-1])    
u = datau['uwnd'][Y1:Y2,(datau['lat'][:]<=20) & (datau['lat'][:]>=-20),(datau['lon'][:]>=140) & (datau['lon'][:]<=280)]
#And for w the same as u
lon = datau['lon'][(datau['lon'][:]>=140) & (datau['lon'][:]<=280)]
lat = datau['lat'][(datau['lat'][:]<=20) & (datau['lat'][:]>=-20)]
#ulon, ulat = np.meshgrid(lon, lat)

w = dataw['wspd'][Y1:Y2,(datau['lat'][:]<=20) & (datau['lat'][:]>=-20),(datau['lon'][:]>=140) & (datau['lon'][:]<=280)]
 #resulting wind stress:
wind = w * u  

llat = len(lat)
llon = len(lon)
ltime = u.shape[0]    

#For the sst:

#This dataset stops for now at februari 2017
pf3 = root + 'HadISST_sst.nc'
pf4 = root + 'HadISST1_SST_update.nc'
datasst = Dataset(pf3)
datasst2 = Dataset(pf4)

Y1 = 1308# 1475#1308# Choose 1476 for the year 1993
Y2 = 10**8#1728#   #10**8 for present

sstlon = np.append(datasst['longitude'][datasst['longitude'][:]>=140], datasst['longitude'][datasst['longitude'][:]<=-80]) #datasst['longitude'][:]
sstlat = datasst['latitude'][np.logical_and(datasst['latitude'][:]<=20, datasst['latitude'][:]>=-20)];  #datasst['latitude'][:]; 
sstlon, sstlat = np.meshgrid(sstlon,sstlat)


sst = np.append(datasst['sst'][Y1:Y2,(datasst['latitude'][:]<=20) & (datasst['latitude'][:]>=-20),(datasst['longitude'][:]>=140)],datasst['sst'][Y1:Y2,(datasst['latitude'][:]<=20) & (datasst['latitude'][:]>=-20),(datasst['longitude'][:]<=-80)],axis=2)
sstlon[sstlon<0] += 360

points = np.concatenate((sstlon.flatten()[:,np.newaxis] , sstlat.flatten()[:,np.newaxis]), axis=1)
#sst = datasst['sst'][Y1:Y2,:,:]
try:
    print 'first date of HadISST dataset: ', datetime(1870, 1,1) + timedelta(days=int(datasst['time'][Y1]))
    print 'last date of HadISST dataset: ', datetime(1870, 1,1) + timedelta(days=int(datasst['time'][Y2]))
except:
    print 'first date of HadISST dataset: ', datetime(1870, 1,1) + timedelta(days=int(datasst['time'][-1]))
    print 'last date of HadISST dataset: ', datetime(1870, 1,1) + timedelta(days=int(datasst['time'][-1]))    
#Swap some axis
wind = np.swapaxes(wind,0,2)
wind = np.swapaxes(wind,0,1)
sst = np.swapaxes(sst,0,2)  
sst = np.swapaxes(sst,0,1)

#Define time
time = np.linspace(1979,2014,ltime) 
#Decrease resolution of SST:
SST = np.zeros(wind.shape)
for i in range(sst.shape[2]):
    if i%20==0: print i/np.float(sst.shape[2])

    gridz = griddata(points, sst[:,:,i].flatten(), (ulon, ulat), method='linear')
    SST[:,:,i] = gridz

time = np.linspace(1993,2018.9166666666666666,ltime) 
   
SST = fwb.sub_season_monthly(time,SST)
wind = fwb.sub_season_monthly(time,wind)
#%%
resi = np.zeros((llat,llon,ltime))

for lo in range(llon):
    for la in range(llat):
        X = wind[la,lo]
        Y = SST[la,lo]
        x = (X-np.nanmean(X))/np.std(X)
        y = (Y-np.nanmean(Y))/np.std(Y)
        if((np.isnan(y)).all()):
            resi[la,lo] = x
        else:
            p = np.polyfit(y,x,1)
            r = p[0]*y+p[1]
            resi[la,lo] = x - r

wu_re = np.reshape(resi,(llat*llon,ltime))

wu_re = scipy.signal.detrend(wu_re)
ncomp = 2

pca = PCA(n_components=ncomp)
pca.fit(wu_re)
V = pca.components_
for i in range(V.shape[0]):
    plt.plot(time,V[i])
    plt.show()   

print 'variance of the different components: ',pca.explained_variance_ratio_
print 'total variance: ', np.sum(pca.explained_variance_ratio_)

eof = pca.fit_transform(wu_re)
eof = np.reshape(eof,(llat,llon,ncomp))
eof[eof==0] = float('nan')
plt.contourf(lon,lat,eof[:,:,0])
plt.colorbar()
plt.show()
plt.contourf(lon,lat,eof[:,:,1])
plt.colorbar()
plt.show()

pc1 = V[0]
pc2 = V[1]

plt.plot(time,pc2)
plt.show()

np.save('secondPC_WWB_1993.npy',pc2)

#%% plot in plotly subplot with EOF and PC2
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

"""Login into plotly for plot"""


trace2 = go.Scatter(
    x=time,
    y=pc2,
    line = dict(
        width = 3,
        color = ('rgb(24,12,255)')),
    name='second PC'
)
    
trace1 = {
    'z': eof[:,:,1],
    'x': lon,
    'y': lat,
    'type': 'contour',
    'showscale':  True,    
    'contours' : dict(
            start=-16,
            end=16,
            size=4,

        ),
        'colorbar':dict(
                      thickness=32,
                      len=0.9,
                      #position=0.9,
                      tickfont= dict(size=21),
                        nticks = 5,
                      xanchor = 'right',
                      x = 0.96
                      ),

}
fig = tls.make_subplots(rows=1, cols=2, shared_yaxes=False, shared_xaxes=False)#,
                        #subplot_titles=('second PC','second EOF'))
#, shared_xaxes=True
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout']['xaxis2'].update(title='Time (year)',
                                    dtick=5,
                                    tickangle=45,
                                    showgrid=True,
                                    tickfont=dict(
                                    #color='#d62728'
                                    size=30),
                                    titlefont=dict(
                                        size=30,
                                    ),
                                    domain=[0.05,0.45])
fig['layout']['xaxis1'].update(title='$^{\\circ}\ E$',
                                    #dtick=5,
                                    #showgrid = True,
                                    #tickangle=45,
                                    tickfont=dict(
                                    #color='#d62728'
                                    size=30),
                                    titlefont=dict(
                                        size=30,
                                    ),
                                    domain=[0.53,0.9])

fig['layout'].update(showlegend=False)#,title='ENSO in black, zonal skewness in red')

fig['layout']['yaxis2'].update(        
                                title='PC',
                                zeroline=False,
                                titlefont=dict(
                                    size=30
                                ),
                                #position=0.61,
                                tickfont=dict(
                                    #color='#d62728'
                                    size=30
                                ),
                                #showgrid = False,

                                range=[-0.15,0.15],domain=[0.05,0.95])#                                    
                                    
fig['layout']['yaxis1'].update(
                                title='$^{\\circ}\ N$',
                                titlefont=dict(
                                    size=30
                                    ),
                                tickfont=dict(
                                    #color='#d62728'
                                    size=30
                                    ),domain=[0.05,0.95])#,title='Jan')#title='NINO34',

fig['layout'].update(height=700,width=1500,margin=go.Margin(t=40,b=80,l=30,r=10))
fig['layout']['titlefont'].update({'size':30}) 
                                               
                                              
plot_url = py.plot(fig, filename='PC')

