# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:51:08 2017
Functions used for PC analyses
@author: Peter Nooteboom
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
from numba import jit
from scipy.optimize import leastsq

def fitsin(time,data):
    guess_mean = np.mean(data)
    guess_std = 1.5*np.std(data)
    guess_phase = 0
    gp2 = 5.
    
    data_first_guess = guess_std*np.sin(gp2*(time+guess_phase)) + guess_mean

    optimize_func = lambda x: x[0]*np.sin(time+x[1])+x[2] - data
    est_std, est_phase, est_mean, est_gp2 = leastsq(optimize_func, [guess_std, guess_phase, guess_mean,gp2])[0]

    data_fit = est_std*np.sin(est_gp2*(time+est_phase)) + est_mean

    plt.plot(time,data,'.')
    plt.plot(time,data_fit,label='after fit')
    plt.plot(time,data_first_guess,label='first guess')
    plt.legend()
    plt.show()

    return data_fit
   
    
#@jit
def sub_season_monthly(time,resi):
    #subtract seasonal cycle from monthly data
    llat = resi.shape[0]
    llon = resi.shape[1]
    ltime = resi.shape[2]

    ano_sst = np.zeros((llat,llon,ltime))
    tyears = int(m.ceil(ltime/12.))
    SST = np.full((llat,llon,tyears*12),float('NaN'))
    SST[:,:,:ltime] = resi
    #SST[:,:,ltime:] = float('NaN')
    
    #Using the fit:
    for la in range(llat):
        for lo in range(llon):
            ts = SST[la,lo]
            avgmon = np.zeros(12)
            for i in range(12):         
                avgmon[i] = np.mean(ts[i:i+12*(tyears-1):12])
            for k in range(ltime):
                i = k%12
                ts[k] = ts[k]-avgmon[i]

#            ts = np.reshape(ts,(12,tyears))
#            annual_avg = np.nanmean(ts,axis=1)
#            for k in range(tyears):
#                ts[:,k] = ts[:,k]-annual_avg
#            ts = np.reshape(ts,ts.size)[:ltime]
            ano_sst[la,lo] = ts[:ltime]
    
    return ano_sst      
    
def sub_season_weekly(time,resi):
    #subtract seasonal data from weekly data
    llat = resi.shape[0]
    llon = resi.shape[1]
    ltime = resi.shape[2]

    ano_sst = np.zeros((llat,llon,ltime))
    tyears = int(m.ceil(ltime/52.))
    SST = np.full((llat,llon,tyears*52),float('NaN'))
    SST[:,:,:ltime] = resi
    #SST[:,:,ltime:] = float('NaN')
    
    #Using the fit:
    for la in range(llat):
        for lo in range(llon):
            ts = SST[la,lo]
            avgmon = np.zeros(52)
            for i in range(52):         
                avgmon[i] = np.mean(ts[i:i+52*(tyears-1):52])
            for k in range(ltime):
                i = k%52
                ts[k] = ts[k]-avgmon[i]

#            ts = np.reshape(ts,(12,tyears))
#            annual_avg = np.nanmean(ts,axis=1)
#            for k in range(tyears):
#                ts[:,k] = ts[:,k]-annual_avg
#            ts = np.reshape(ts,ts.size)[:ltime]
            ano_sst[la,lo] = ts [:ltime]
    
    return ano_sst      
    
def daily_to_weeklymean(data):
    new = np.zeros((data.shape[0],data.shape[1],data.shape[2]/7))
    for i in range(data.shape[2]/7):
        new[:,:,i] = np.mean(data[:,:,i*7:i*7+6],axis=2)
    return new  
    
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
    
    
def lowerres_SST(sst,llat,llon):
    for i in range(llon):
        for j in range(llat):
            if(i%2!=0 and j%2!=0):
                s = sst[((j-1)*2.5+1):((j-1)*2.5+3),((i-1)*2.5+1):((i-1)*2.5+3),:]
#                print '1',s.shape
                sst[j,i,:] = np.nanmean(s,axis=(0,1))
            elif(i%2!=0 and j%2==0):
                s = sst[((j-2)*2.5+4):((j-2)*2.5+5),((i-1)*2.5+1):((i-1)*2.5+3),:]
#                print '2',s.shape
                sst[j,i,:] = np.nanmean(s,axis=(0,1))               
            elif(i%2==0 and j%2!=0):   
                s = sst[((j-1)*2.5+1):((j-1)*2.5+3),((i-2)*2.5+4):((i-2)*2.5+5),:]
#                print '3',s.shape
                sst[j,i,:] = np.nanmean(s,axis=(0,1))  
            else:
                s = sst[((j-2)*2.5+4):((j-2)*2.5+5),((i-2)*2.5+4):((i-2)*2.5+5),:]
#                print '4',s.shape                
                sst[j,i,:] = np.nanmean(s,axis=(0,1)) 
    sst[sst==np.inf] = float('Nan')
    return sst                
