# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:13:59 2016

Hybrid model for future prediction 12 months ahead

@author: Peter Nooteboom
"""
#Load packages, define functions
import numpy as np
import matplotlib.pylab as plt
from copy import copy
import el_nino_manip as manip
import el_nino_weka as weka 
import os

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
#%% Choose attributes/ hyperparameters: 
#Setting home directory for reading and writing
root_Dir_write = 'predictions/'
root_Dir_read = 'input/'

#Attributes to include in the machine learning input:
bc2 = True #c2
bPC2 = True # second principal component of the residual of the wind stress
bseasonal_cycle = True #seasonal cycle
brunning_mean = True #Three month running mean of the attributes

# Lead time of prediction (months)
monlow = 12; monhigh = monlow + 1; monlen = monhigh - monlow;

#ARIMA order             
ARIMA_ordera = [(12,1,0)]

#amount of epochs for the ANN to train
traintimer = 700 

testset = 11; print 'testset: ',testset # test set (%)
#%% Load the data
#Include ENSO index and the time 
#ARIMA_ninosq is the NINO3.4 data before the used dataset.
#nino is the NINO3.4 in the TtimesN matrix
#NINO3.4 from http://www.cpc.ncep.noaa.gov/data/indices/ersst3b.nino.mth.81-10.ascii
ARIMA_ninosq = np.loadtxt(os.getcwd() + '/' + root_Dir_read + 'NINO34_1992-102018.txt')#
ARIMA_ninosq = np.delete(ARIMA_ninosq,0,1)
ARIMA_ninosq = np.reshape(ARIMA_ninosq,ARIMA_ninosq.size)[:12]
nino = np.loadtxt(root_Dir_read +'NINO34_1992-102018.txt')
nino = np.delete(nino,0,1)
nino = np.reshape(nino,nino.size)[13:-2]

time = 1993+np.arange(0,len(nino))/12.
timeplot = time
seascycle = -0.1*np.cos(time*2*np.pi)

attr = ['date_time','ElNino']
       
if(bc2):
    attr.append('c2')    
    c2 = np.load(os.getcwd() + '/' +root_Dir_read +'c2_h_th09_1993.npy')
    timec2 = np.load(root_Dir_read +'timec2.npy')
    c2 = (c2 - np.mean(c2))/np.nanstd(c2)    
    c2 = np.interp(time,timec2,c2)
    if(brunning_mean):
        c2 = running_mean(c2,3)    
if(bPC2):
    attr.append('PC2')    
    PC2 = np.load(os.getcwd() + '/' +root_Dir_read +'secondPC_WWB_1993.npy')[:-2]
    if(brunning_mean):       
        PC2 = running_mean(PC2, 3)              
if(bseasonal_cycle):
    attr.append('seasonal_cycle')
    
lagtot = 0
minlen = min(len(nino),len(time))

attributes = ['date_time','ElNino']    
joined = np.zeros((minlen-lagtot,1))
for i in range(minlen-lagtot):
    joined[i,0] = time[lagtot+i] #joined.append(time)  
joined = np.append(joined,np.reshape(nino[lagtot:],(minlen-lagtot,1)),axis=1)
if(bc2):
    attributes.append('c2')
    joined = np.append(joined,np.reshape(c2[lagtot:],(minlen-lagtot,1)),axis=1)
if(bPC2):
    attributes.append('PC2')
    joined = np.append(joined,np.reshape(PC2[lagtot:],(minlen-lagtot,1)),axis=1)   
if(bseasonal_cycle):
    attributes.append('seasonal_cycle')
    joined = np.append(joined,np.reshape(seascycle[lagtot:],(minlen-lagtot,1)),axis=1)           

dic = {}
for i in range(len(attributes)):
    dic[attributes[i]] = joined[:,i]
#%% Try the ARIMA model to El Nino, using only the ENSO sequence
from statsmodels.tsa.arima_model import ARIMA

length = len(dic['ElNino'][:])
size = int(len(dic['ElNino'][:]) * (1-testset/100.))
train, test = dic['ElNino'][0:size], dic['ElNino'][size:len(nino)]
traintime,testtime = dic['date_time'][0:size], dic['date_time'][size:len(nino)]

ARIMA_ts = [[]]*monlen
resid = [[]]*monlen
joined = []

for m in range(monlen):
    print 'month: ',m+monlow
    dicti = copy(dic)
    mon = m + monlow
    
    #Add zeros to the values of the future
    addlength = mon 
    for at in dic.keys():
        dicti[at] = np.append(dic[at],np.zeros(addlength))    

    ARIMA_order = ARIMA_ordera[0]
  
    #'Train' the ARIMA model and retreive the residual
    model = ARIMA(dicti['ElNino'][:size+1],order=ARIMA_order)
    results_ARIMA = model.fit(disp=0)
    params = results_ARIMA.params
    arparams = results_ARIMA.arparams
    maparams = results_ARIMA.maparams
#% prepare for the ANN regression:
#Amount of steps prediction ahead
    steps = mon

    ARIMA_ts1 = np.array([]) 
    
    dicti['ElNino_real'] = dicti['ElNino'][:]    
        
    i = 0
    resids = np.zeros(len(arparams))
    while(len(ARIMA_ts[m])<len(dicti['ElNino'][:])):
        ARIMA_ts[m] = np.append(ARIMA_ts[m],weka.forecasts(ARIMA_ninosq,steps,params,arparams,resids, ARIMA_ts)[-1])        
        ARIMA_ninosq = np.append(ARIMA_ninosq,(dicti['ElNino_real'][i]))  
        if(i>=1):
            resids = np.append(resids, ARIMA_ninosq[-1]-np.array(ARIMA_ts)[-1])    
        i+=1
        
    steps = steps - 1
    
    #Define the residual, to be predicted with ANN
    if(steps!=0):
        resid[m] = dicti['ElNino'][steps:] - np.array(ARIMA_ts[m])[:-steps]
    else:
        resid[m] = dicti['ElNino'] - np.array(ARIMA_ts[m])

    for i in range(len(attributes)):
        dicti[attributes[i]] = dicti[attributes[i]][steps:]

    dicti['ElNino'] = resid[m]

    joined.append(dicti)

#%%
attributes = np.append(attributes,'ElNino_real')
#%% Now we use the ANN regression for the residual    
# create a weka instance-friendly file with given parameters
t0 = 0. # starting date
deltat = 0.0 # start from which data point, in general we just use the first data point

res = []
time = []
actual = []     

print 'keys: ',joined[0].keys()

avg_RMSE = np.zeros(monlen) #The average RMSE of the ensemble

min_RMSE = [np.inf]*monlen
best_res = [[]]*monlen
best_NN_size = [[]]*monlen

firstlen = 2;secondlen = 1;thirdlen = 1;fourthlen = 1; #Take only one ANN structure
size_ens = (firstlen-1)*(secondlen)*(thirdlen)*(fourthlen)-(firstlen-1)*(thirdlen-1)
count = 0.
        
#print 'shape of the neural network: ',s
for i in range(1,firstlen):
    for j in range(0,secondlen):
        for k in range(0,thirdlen):
                        
            if(j==0 and k>0):
                break
            for l in range(0,fourthlen):
                if(k==0 and l>0):
                    break
                if __name__ == "__main__":

                    for mon in range(monlow,monhigh): # test different leading time
                        #if(mon==1):
                        s = [4,1,3]#ANN structure

                        steps = mon-1

                        tau = mon/12.0 # The leading time of prediction in years
                    
                        nn = manip.el_nino_weka_regr(joined[mon-monlow],t0,deltat,tau) # repares the dataset used for regression problems
                    
                        pop = np.array(['t0-deltat','ElNino_0'])
                    
                        
                        name_train = 'train_UU_regression'
                        name_test = 'test_UU_regression'
                        train_set = root_Dir_write + name_train
                        test_set = root_Dir_write + name_test                
                    		
                        p = manip.training_test_sets(nn, 100, 100-testset , testset , name_train, name_test , root_Dir_write, pop = pop , typ = 'arff') # build test set
                        result = weka.NN_regression(train_set,test_set,print_feat = p,layers = s,train_time = traintimer) # use ANN with the default layer structure "a" 
                        # the default layer structure "a" is (# of attributes + # of classes) / 2, here is layers= [2]
                        
                        if(steps!=0):
                            prediction = np.array(result['predicted'][:] + np.array(ARIMA_ts[mon-monlow])[-steps-len(result['predicted'][:]):-steps])
                        else:
                            prediction = np.array(result['predicted'][:] + np.array(ARIMA_ts[mon-monlow])[-len(result['predicted'][:]):])
                            
                        error = weka._NorRMSE(joined[mon-monlow]['ElNino_real'][-len(result['predicted']):],prediction) 
                        t = nn['t0'][-len(result['predicted'])-1:-1]
                    
                        if(i==1 and j==0 and k==0 and l==0):
                            res.append(prediction)
                            actual.append(joined[mon-monlow]['ElNino_real'][-len(result['predicted']):])
                            time.append(t)
    
                            avg_RMSE[mon-monlow] += error
                            
                            if(mon==monhigh-1):
                                res = np.array(res)
                                time = np.array(time)
                                actual = np.array(actual)
                        else:
                            for el in range(len(res[mon-monlow])):
                                if(steps!=0):
                                    res[mon-monlow][el] += result['predicted'][el] + np.array(ARIMA_ts[mon-monlow])[-len(result['predicted'][:])-steps:-steps][el]
                                else:
                                    res[mon-monlow][el] += result['predicted'][el] + np.array(ARIMA_ts[mon-monlow])[-len(result['predicted'][:]):][el]
                      
                            if(mon==monhigh-1):
                                res = np.array(res)
                                time = np.array(time)
                                actual = np.array(actual)
                    
                        for el in range(len(res[mon-monlow])):
                            if(steps!=0):
                                res[mon-monlow][el] += result['predicted'][el] + np.array(ARIMA_ts[mon-monlow])[-len(result['predicted'][:])-steps:-steps][el]
                            else:
                                res[mon-monlow][el] += result['predicted'][el] + np.array(ARIMA_ts[mon-monlow])[-len(result['predicted'][:]):][el]
                                
                            avg_RMSE[mon-monlow] += error
                            
                        if(error<min_RMSE[mon-monlow]):
                            min_RMSE[mon-monlow] = error
                            best_res[mon-monlow] = prediction
                            best_NN_size[mon-monlow] = s
                                      
                else:
                    continue
            else:
                continue  


#%%  Plot and save prediction
for i in range(len(avg_RMSE)):
    avg_RMSE[i] = avg_RMSE[i]/float(count)

#And the results with minimal RMSE:  
print 'best ANN structure: ',best_NN_size             
for i in range(monlen):
    plt.plot(timeplot[-len(actual[i]):],actual[i],'r--',label = "actual")
    plt.plot(timeplot[-len(actual[i]):],running_mean(np.array(best_res[i]),1),'b',label = "predicted")
    plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
    plt.xlabel("time (years)")
    plt.ylabel("NINO3.4")
    plt.title(" ENSO predition " + str(monlow+i) + " months ahead (RMSE = " + str(min_RMSE[i]) + " )")
    stri = ''
    for j in range(len(attributes)):
        stri += '_' + attributes[j]
    plt.show()        
            
for i in range(monlow,monhigh):
    np.save(root_Dir_write + 'pred_mon{}.npy'.format(i),np.array(best_res[i-monlow]))
    