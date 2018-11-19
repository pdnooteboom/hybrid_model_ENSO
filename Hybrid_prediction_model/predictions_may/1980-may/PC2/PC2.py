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

import functions_WWB as fwb

root = 'C:/Users/User/Documents/Thesis/observations/predictpresent/1980-may/PC2/'
root2 = 'C:/Users/User/Documents/Thesis/observations/predictpresent/1980-may/'


orig = True
weekly = True
monthly = True
if(orig):
    pf1 = root + 'uwnd.mon.mean.nc'
    pf2 = root + 'wspd.mon.mean.nc'
    
    datau = Dataset(pf1)
    dataw = Dataset(pf2)
    #for u
    lon = datau['lon'][(datau['lon'][:]>=140) & (datau['lon'][:]<=284)]
    lat = datau['lat'][(datau['lat'][:]<=20) & (datau['lat'][:]>=-20)]

    Y1 = 384 # choose 384 for the year 1980
    #Y2 = -1
    u = datau['uwnd'][Y1:,(datau['lat'][:]<=20) & (datau['lat'][:]>=-20),(datau['lon'][:]>=140) & (datau['lon'][:]<=280)]
    #And for w the same as u
    lon = dataw['lon'][(dataw['lon'][:]>=140) & (dataw['lon'][:]<=280)]
    lat = dataw['lat'][(dataw['lat'][:]<=20) & (dataw['lat'][:]>=-20)]
    
    w = dataw['wspd'][Y1:,(dataw['lat'][:]<=20) & (dataw['lat'][:]>=-20),(dataw['lon'][:]>=140) & (dataw['lon'][:]<=280)]
     #resulting wind stress:
    wind = w * u  

    llat = len(lat)
    llon = len(lon)
    ltime = u.shape[0]    
    #For the sst:
    
    #This dataset stops for now at februari 2017
    pf3 = root + 'HadISST_sst.nc.gz'
    pf4 = root + 'HadISST12_SST_update.nc.gz'
    datasst = Dataset(pf3)
    datasst2 = Dataset(pf4)
    
    Y1 = 1319# Choose 1319 for the year 1980
    #Y2 = -1
    #sst = datasst['sst'][Y1:Y2,(datasst['latitude'][:]<=20) & (datasst['latitude'][:]>=-20),(datasst['longitude'][:]<=140) & (datasst['longitude'][:]>=-80)]
    sst = np.append(datasst['sst'][Y1:,(datasst['latitude'][:]<=20) & (datasst['latitude'][:]>=-20),(datasst['longitude'][:]>=140)],datasst['sst'][Y1:,(datasst['latitude'][:]<=20) & (datasst['latitude'][:]>=-20),(datasst['longitude'][:]<=-80)],axis=2)

    
    wind = np.swapaxes(wind,0,2)
    wind = np.swapaxes(wind,0,1)
    sst = np.swapaxes(sst,0,2)    
    sst = np.swapaxes(sst,0,1)  

    time = np.linspace(1993,2017.3333333333333333333333,ltime) 
    
    SST = fwb.lowerres_SST(sst,llat,llon)
    
else:    
    pf8 = "C:/Users/User/Documents/Thesis/observations/weekly/u10_SST_new.nc"  
    #pf9 = "C:/Users/User/Documents/Thesis/observations/weekly/u10_SST3.nc"
    
    data_u_SST =  Dataset(pf8)
    
    lon = data_u_SST['longitude'][::18]
    llon = len(lon)
    lat = data_u_SST['latitude'][::6]
    llat = len(lat)
    
    wind = data_u_SST['u10'][:,::6,::18]
    SST = data_u_SST['sst'][:,::6,::18]
#    if(weekly):
#        SST = SST[::7]
#        wind = wind[::7]

    SST = np.swapaxes(SST,0,2)
    SST = np.swapaxes(SST,0,1)
    
    wind = np.swapaxes(wind,0,2)
    wind = np.swapaxes(wind,0,1)
    
    if(weekly):
        #Take the weakly mean from daily:
        wind = fwb.daily_to_weeklymean(wind)
        SST = fwb.daily_to_weeklymean(SST)
    elif(monthly):
        wind = fwb.daily_to_monthlymean(wind)
        SST = fwb.daily_to_monthlymean(SST)
    #wind = wind[:,:,::7]
    #SST = SST[:,:,::7]
    
    ltime = SST.shape[2]
    time = np.linspace(1979,2014,ltime)
    
if(orig):    
    SST = fwb.sub_season_monthly(time,SST)
    wind = fwb.sub_season_monthly(time,wind)
else:    
    if(weekly):
        SST = fwb.sub_season_weekly(time,SST)
        wind = fwb.sub_season_weekly(time,wind)
    else:
        SST = fwb.sub_season_monthly(time,SST)
        wind = fwb.sub_season_monthly(time,wind)
        
#plt.plot(time,SST[1,2])
##plt.xlim(0,365)
##plt.plot(time,wind[1,2])
#plt.show()
#plt.plot(time,wind[1,2])
##plt.xlim(0,365)
##plt.plot(time,wind[1,2])
#plt.show()


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
#SST = scipy.signal.detrend(SST)


wu_re = scipy.signal.detrend(wu_re)

#import iris
#from eofs.iris import Eof
#
#solver = Eof(wu_re)
#eofs = solver.eofs(neofs=2)

ncomp = 24

pca = PCA(n_components=ncomp)
pca.fit(wu_re)
V = pca.components_
for i in range(V.shape[0]):
    plt.plot(time,V[i])
    #plt.xlim(time[0],len(time)-1)
    plt.show()   
#plt.plot(time,V[0])
#plt.xlim(time[0],len(time)-1)
#plt.show()
#plt.plot(time,V[1])
#plt.show()
#plt.plot(time,V[2])
#plt.show()
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
plt.contourf(lon,lat,eof[:,:,2])
plt.colorbar()
plt.show()
plt.contourf(lon,lat,eof[:,:,3])
plt.colorbar()
plt.show()

pc1 = V[0]
pc2 = V[1]
pc3 = V[2]
pc4 = V[3]
pc5 = V[4]
pc6 = V[5]

plt.plot(time,pc2)
plt.show()

#np.save('firstPC_WWB_weekly.npy',pc1)
np.save('secondPC_WWB_weekly.npy',pc2)
#np.save('thirdPC_WWB_weekly.npy',pc3)
#np.save('fourthPC_WWB_weekly.npy',pc4)
#np.save('fifthPC_WWB_weekly.npy',pc5)
#np.save('sixthPC_WWB_weekly.npy',pc6)

#%%
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

#ARIMA_ninosq = np.loadtxt('ENSO1975-1981.txt')
#ARIMA_ninosq = np.delete(ARIMA_ninosq,0,1)
#ARIMA_ninosq = np.reshape(ARIMA_ninosq,ARIMA_ninosq.size)
#nino34 = np.loadtxt(root2 + 'nino1992-052017.txt')
##nino = np.append(ARIMA_ninosq[-24:],nino34[:,9])
#nino = nino34[:,9]
#nino = nino[12:-1] #Until february, not april
##ARIMA_ninosq = ARIMA_ninosq[:-24]
#
#time = np.linspace(1980,2017 + 4/12.,len(nino)+1)
#time = time[1:]

#maxlag,maxi = highestcorlag(pc2,nino)
#
#print 'maximum correlation between PC2 and nino: ',maxi
#print 'at lag of nino: ', maxlag

print 'variance of the different components: ',pca.explained_variance_ratio_


#%% plot in plotly subplot with EOF and PC2
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

#py.sign_in('peternooteboom','1pbc2kdd8l')#nootje01
#py.sign_in('Peppie','fZpfuzOdO8YiVa8mdHyN')#bootje01
py.sign_in('pepernoot','S6WBRkLkYIdjmljKPMZV')#Nootsaeck


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

