# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:45:44 2017
Calculation of the second proniciple component of the residual of the wind stress
@author: Peter Nooteboom
"""

import numpy as np
import matplotlib.pylab as plt
import scipy.signal
from netCDF4 import Dataset
from sklearn.decomposition import PCA

import functions_WWB as fwb


#Loading the horizontal wind component
pf1 = 'C:/Users/User/Documents/Thesis/observations/WWB/input/uwnd.mon.mean.nc'
pf2 = 'C:/Users/User/Documents/Thesis/observations/WWB/input/wspd.mon.mean.nc'
datau = Dataset(pf1)
dataw = Dataset(pf2)
#for u define longitude and latitude
lon = datau['lon'][(datau['lon'][:]>=140) & (datau['lon'][:]<=284)]
lat = datau['lat'][(datau['lat'][:]<=20) & (datau['lat'][:]>=-20)]

Y1 = 372 # choose 360 for 1978, 372 for 1979
Y2 = 792 # choose 672 for 2004, 792 for 2014
u = datau['uwnd'][Y1:Y2,(datau['lat'][:]<=20) & (datau['lat'][:]>=-20),(datau['lon'][:]>=140) & (datau['lon'][:]<=280)]
#And for w the same as u for longitude and latitude
lon = dataw['lon'][(dataw['lon'][:]>=140) & (dataw['lon'][:]<=280)]
lat = dataw['lat'][(dataw['lat'][:]<=20) & (dataw['lat'][:]>=-20)]

w = dataw['wspd'][Y1:Y2,(dataw['lat'][:]<=20) & (dataw['lat'][:]>=-20),(dataw['lon'][:]>=140) & (dataw['lon'][:]<=280)]
 #resulting wind stress:
wind = w * u  

#Define lengths
llat = len(lat)
llon = len(lon)
ltime = u.shape[0]
    
#Load SST
pf3 = 'C:/Users/User/Documents/Thesis/observations/WWB/input/HadISST_sst.nc'
datasst = Dataset(pf3)

Y1 = 1308# Choose 1296 for 1978, 1308 for 1979
Y2 = 1728# Choose 1608 for 2004, 1728 for 2014
sst = np.append(datasst['sst'][Y1:Y2,(datasst['latitude'][:]<=20) & (datasst['latitude'][:]>=-20),(datasst['longitude'][:]>=140)],datasst['sst'][Y1:Y2,(datasst['latitude'][:]<=20) & (datasst['latitude'][:]>=-20),(datasst['longitude'][:]<=-80)],axis=2)

#Swap some axis
wind = np.swapaxes(wind,0,2)
wind = np.swapaxes(wind,0,1)
sst = np.swapaxes(sst,0,2)    
sst = np.swapaxes(sst,0,1)  
#Define time
time = np.linspace(1979,2014,ltime) 
#Decrease resolution of SST
SST = fwb.lowerres_SST(sst,llat,llon)


# Subtract the seasonal cycle
SST = fwb.sub_season_monthly(time,SST)
wind = fwb.sub_season_monthly(time,wind)
        
#%%
#Subtract the linear effect of SST from the horizontal wind component
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
#SST = scipy.signal.detrend(SST)

#derend
wu_re = scipy.signal.detrend(wu_re)
#Amount of components considered for the PC calculation
ncomp = 2

#Perform the PC calculation 
pca = PCA(n_components=ncomp)
pca.fit(wu_re)
V = pca.components_
pc1 = V[0]
pc2 = V[1]
for i in range(V.shape[0]):
    plt.plot(time,V[i])
    #plt.xlim(time[0],len(time)-1)
    plt.show()   

print 'variance of the different components: ',pca.explained_variance_ratio_
print 'total variance: ', np.sum(pca.explained_variance_ratio_)

#Calculate the EOFs of the PCs
eof = pca.fit_transform(wu_re)
eof = np.reshape(eof,(llat,llon,ncomp))
eof[eof==0] = float('nan')

#Plot the first two PCs
plt.contourf(lon,lat,eof[:,:,0])
plt.colorbar()
plt.show()
plt.contourf(lon,lat,eof[:,:,1])
plt.colorbar()
plt.show()

plt.plot(time,pc2)
plt.show()

np.save('firstPC_WWB_weekly.npy',pc1)
np.save('secondPC_WWB_weekly.npy',pc2)


#%% Calculate and print the highest lagged correlation and its lag
def monthly_to_weekly(array):
    #This function returns from an array with monthly data, an array 
    #that contains the value of that month for the week you are in
    res = np.zeros(len(array)*52/12)
    for i in range(len(res)):
        weekofyear = i%52
        year = i/52
        monthofyear = weekofyear*12/52
        res[i] = array[year*12+monthofyear]
    return res
    

def highestcorlag(var,nino):
    if(len(var)!=len(nino)):
        print 'length not the same'
    maxi = 0.
    maxlag = 0
    for lag in range(15):
        if(lag==0):
            cor = np.abs(np.corrcoef(var,nino)[0,1])
        else:
            cor = np.abs(np.corrcoef(var[:-lag],nino[lag:])[0,1])
        if(cor>maxi):
            maxi = cor
            maxlag = lag
    return maxlag,maxi

#Load nino3.4 index
nino34 = np.loadtxt('monthlynino34.txt')
nino34 = np.delete(nino34,0,1)
nino34 = np.reshape(nino34,nino34.size)


maxlag,maxi = highestcorlag(pc1,nino)

print 'maximum correlation between PC1 and nino: ',maxi
print 'at lag of nino: ', maxlag

maxlag,maxi = highestcorlag(pc2,nino)

print 'maximum correlation between PC2 and nino: ',maxi
print 'at lag of nino: ', maxlag


print 'variance of the different components: ',pca.explained_variance_ratio_


#%% plot in plotly subplot with EOF and PC2
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

"""Sign into Plotly"""
py.sign_in()


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
                      thickness=45,
                      len=0.9,
                      #position=0.9,
                      tickfont= dict(size=45),
                        nticks = 5,
                      xanchor = 'right',
                      x = 1.
                      ),

}
fig = tls.make_subplots(rows=1, cols=2, shared_yaxes=False, shared_xaxes=False,
                        subplot_titles=('(a)','(b)'))
#, shared_xaxes=True
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout']['xaxis2'].update(title='Time (year)',
                                    
                                    dtick=5,
                                    tickangle=45,
                                    showgrid=True,
                                    tickfont=dict(
                                    #color='#d62728'
                                    size=40),
                                    titlefont=dict(
                                        size=45,
                                    ),
                                    domain=[0.05,0.43])
fig['layout']['xaxis1'].update(title='$^{\\circ}\ E$',
                                    dtick=40,
                                    #showgrid = True,
                                    #tickangle=45,
                                    tickfont=dict(
                                    #color='#d62728'
                                    size=40),
                                    titlefont=dict(
                                        size=45,
                                    ),
                                    domain=[0.54,0.88])

fig['layout'].update(showlegend=False)#,title='ENSO in black, zonal skewness in red')

fig['layout']['yaxis2'].update(        
                                title='PC',
                                dtick=0.1,
                                
                                zeroline=False,
                                titlefont=dict(
                                    size=45
                                ),
                                #position=0.61,
                                tickfont=dict(
                                    #color='#d62728'
                                    size=40
                                ),
                                #showgrid = False,

                                range=[-0.15,0.15],domain=[0.12,0.95])#                                    
                                    
fig['layout']['yaxis1'].update(
                                title='$^{\\circ}\ N$',
                                dtick=5,
                                position = 0.53 ,
                                titlefont=dict(
                                    size=45
                                    ),
                                tickfont=dict(
                                    #color='#d62728'
                                    size=40
                                    ),domain=[0.08,0.91])#,title='Jan')#title='NINO34',

fig['layout'].update(height=700,width=1500,margin=go.Margin(t=20,b=80,l=30,r=10))
fig['layout']['titlefont'].update({'size':45}) 
                                               
fig['layout']['annotations'][0].update(y=0.94)
fig['layout']['annotations'][1].update(y=0.94)  
fig['layout']['annotations'][0]['font'].update(size=45)
fig['layout']['annotations'][1]['font'].update(size=45)                                             
#plot_url = py.plot(fig, filename='PC')
py.image.save_as(fig, filename='PC.png')
