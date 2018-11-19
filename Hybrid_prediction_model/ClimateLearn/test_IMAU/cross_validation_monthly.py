# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:51:45 2017

@author: Peter Nooteboom 
"""

import numpy as np
import matplotlib.pylab as plt
from copy import copy
import manip_grmse as manip
import el_nino_weka as weka 
import pandas as pd
from sklearn.metrics import mean_squared_error

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
    
sdata1=1827
lenseq = 52
nslides=1
sq=4.0
lensliding=(sdata1-lenseq)/int(sq)

#If data has to be loaded from the data prepare part
root_Dir_read = 'C:/Users/User/Documents/Thesis/Hybrid_prediction_model/ClimateLearn/input/'
timec2 = np.linspace(1979+lenseq/52.,1979+(lenseq+lensliding*sq)/52.,lensliding*nslides)       

#Always include ENSO index and the time
nino34 = np.loadtxt(root_Dir_read+'monthlynino34.txt')
nino34 = np.delete(nino34,0,1)
nino0 = np.reshape(nino34,nino34.size)
nino = nino0[12:]
    
time = np.load(root_Dir_read + 'time.npy')

seascycle = -0.1*np.cos(time*2*np.pi)
#To include in the machine learning input:
bc2 = False
bPC2 = True
bseasonal_cycle = True
bwwv = True
brunning_mean = True

attr = ['date_time','ElNino']
     
if(bc2):
    attr.append('c2')    
    c2 = np.load(root_Dir_read + 'c2_h_th0.9.pny.npy')
    c2 = (c2 - np.mean(c2))/np.nanstd(c2)
    c2 = np.interp(time,timec2,c2)  
    if(brunning_mean):
        c2 = running_mean(c2,3)  
if(bPC2):
    attr.append('PC2')    
    PC2 = np.load(root_Dir_read + 'secondPC_WWB_weekly.npy')
    PC2 = PC2[lenseq::4]#lenseq:lenseq+lensliding*int(sq):int(sq)]
    PC2 = PC2[:len(timec2)]
    PC2 = (PC2- np.nanmean(PC2))/np.nanstd(PC2)              
    PC2 = np.interp(time,timec2[:-1],PC2) 
    if(brunning_mean):       
        PC2 = running_mean(PC2, 3)                     
if(bseasonal_cycle):
    attr.append('seasonal_cycle')
if(bwwv):
    attr.append('wwv')
    wwv = np.loadtxt(root_Dir_read +'warmwatervolume.txt')
    wwv = wwv[:,2]
    wwv = (wwv - np.nanmean(wwv))/np.std(wwv)
    if(brunning_mean):
        running_mean(wwv, 3) 
    
#Make all time series the same length:
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
if(bwwv):
    attributes.append('wwv')
    joined = np.append(joined,np.reshape(wwv[lagtot:],(minlen-lagtot,1)),axis=1)   

dic = {}
for i in range(len(attributes)):
    dic[attributes[i]] = joined[:,i]
#%% Try the ARIMA model to El Nino, using only the ENSO sequence
# constants used in all hybrid models with different testsets
#Setting home directory for reading and writing
root_Dir_write = 'C:/Users/User/Documents/Thesis/Hybrid_prediction_model/ClimateLearn/output/IMAU/'

# create a weka instance-friendly file with given parameters
t0 = 0.#1960.0  # starting date
deltat = 0.0 # start from which data point, in general we just use the first data point
     
pop = np.array(['t0-deltat','ElNino_0'])
print 'pop: ',pop

mon = 4
print 'mon: ',mon
steps = mon - 1

#ANN structure:
s=[4,3,1]
traintimer = 700

# cross validate by keeping certain percentage splits between training set and test set 200 times each    
N = 200
EE85 = np.zeros(N)
EE80 = np.zeros(N)
EE75 = np.zeros(N)
EE70 = np.zeros(N)

attributes = np.append(attributes,'ElNino_real')
#%%
from statsmodels.tsa.arima_model import ARIMA

dicti = copy(dic)
dicti['ElNino_real'] = dic['ElNino'][:]

testset = 30
print 'testset: ',testset

ARIMA_ts = []

#Determine the testset and train set
size = int(len(dic['ElNino'][:]) * (1-testset/100.))
train, test = dic['ElNino'][0:size], dic['ElNino'][size+1:len(nino)]
traintime,testtime = dic['date_time'][0:size+1], dic['date_time'][size+2:len(nino)]

joined = []

ARIMA_ordera = np.zeros(1, dtype=(int,3))
ARIMA_ordera = [(12,1,0)]
ARIMA_order = ARIMA_ordera[0]

#%% Now we use the ANN regression for the residual    
#print 'keys: ',joined[0].keys()
m = 0
paramsa = []
maparama = []

resultsar = np.zeros((N,12))
resultsma = np.zeros((N,1))
for k in range(N):
    if(k%10==0):
        print k
    dicti = copy(dic)
    dicti['ElNino_real'] = dic['ElNino'][:]
    #Determine the testset and train set
    size = int(len(dic['ElNino'][:]) * (1-testset/100.))

    np.random.seed(k)
    tr0 = np.random.randint(0,len(dic['ElNino'][:])-size-1)
    
    #init_test = manip.training_test_sets(dic['ElNino'][:], 100, 100-testset , testset , name_train, name_test , root_Dir_write, pop = pop , typ = 'arff') # build test set
    train = dic['ElNino'][tr0:tr0+size]#, dic['ElNino'][size:len(nino)]
    traintime = dic['date_time'][tr0:tr0+size]#, dic['date_time'][size:len(nino)]
        

    dicti = copy(dic)
  
    model = ARIMA(dicti['ElNino'][tr0:tr0+size],order=ARIMA_order)
    results_ARIMA = model.fit(disp=0)
    params = results_ARIMA.params
    arparams = results_ARIMA.arparams

    resultsar[k] = arparams    
    
    ARIMA_ninosq = copy(nino0[lenseq/4-(len(params)-2):lenseq/4])
    
    i = 0
    resids = np.zeros(len(arparams))
    dicti['ElNino_real'] = dic['ElNino'][:]    
    while(len(ARIMA_ts)<len(dicti['ElNino_real'][:])):
        ARIMA_ts = np.append(ARIMA_ts,weka.forecasts(ARIMA_ninosq,mon,params,resids, ARIMA_ts)[-1])
        ARIMA_ninosq = np.append(ARIMA_ninosq,(dicti['ElNino_real'][i]))  
        if(i>=0):
            resids = np.append(resids, ARIMA_ninosq[-1]-np.array(ARIMA_ts)[-1])    
        i+=1
        
    resid = dicti['ElNino_real'][steps:] - np.array(ARIMA_ts)[:-steps]
    for i in range(len(attributes)):
        dicti[attributes[i]] = dicti[attributes[i]][steps:]  
    
    dicti['ElNino'] = resid
    
    joined.append(dicti)        

    tau = mon/12.0 # The lead time of prediction in years

    nn = manip.el_nino_weka_regr(joined[0],t0,deltat,tau) # repares the dataset used for regression problems
 
    name_train = 'train_UU_crossvalid'
    name_test = 'test_UU_crossvalid'
    train_set = root_Dir_write + name_train
    test_set = root_Dir_write + name_test             
		
    p,mintest,maxtest = manip.random_training_test_sets(nn, 100-testset , testset , name_train, name_test , root_Dir_write, pop = pop , typ = 'arff',seed=k)
    result = weka.NN_regression(train_set,test_set,print_feat = p,layers = s,train_time = traintimer) # use ANN with the default layer structure "a" 
    # the default layer structure "a" is (# of attributes + # of classes) / 2, here is layers= [2]
    
    prediction = np.array(result['predicted'][:] + np.array(ARIMA_ts)[-steps-len(result['predicted'][:]):-steps])
    
    RMSE = mean_squared_error(result['predicted'][:],result['actual'][:])**0.5
    error = RMSE/(maxtest-mintest)
    t = nn['t0'][-len(result['predicted'])-1:-1]
    print "RMSE=",error
    
    EE70[k] = error
        
        
#%% Here for the 25 % testset
        
testset = 25
print 'testset: ',testset

dicti = copy(dic)
dicti['ElNino_real'] = dic['ElNino'][:]

ARIMA_ts = []

#Determine the testset and train set
size = int(len(dic['ElNino'][:]) * (1-testset/100.))
#init_test = manip.training_test_sets(dic['ElNino'][:], 100, 100-testset , testset , name_train, name_test , root_Dir_write, pop = pop , typ = 'arff') # build test set
train, test = dic['ElNino'][0:size], dic['ElNino'][size+1:len(nino)]
traintime,testtime = dic['date_time'][0:size+1], dic['date_time'][size+2:len(nino)]

joined = []
#%% Now we use the ANN regression for the residual   
m = 0

paramsa = []
maparama = []

resultsar = np.zeros((N,12))
resultsma = np.zeros((N,1))

for k in range(N):
    if(k%10==0):
        print k
    dicti = copy(dic)
    dicti['ElNino_real'] = dic['ElNino'][:]
    #Determine the testset and train set
    size = int(len(dic['ElNino'][:]) * (1-testset/100.))

    np.random.seed(k)
    tr0 = np.random.randint(0,len(dic['ElNino'][:])-size-1)
    
    train = dic['ElNino'][tr0:tr0+size]
    traintime = dic['date_time'][tr0:tr0+size]
        
    dicti = copy(dic)
  
    model = ARIMA(dicti['ElNino'][tr0:tr0+size],order=ARIMA_order)
    results_ARIMA = model.fit(disp=0)
    params = results_ARIMA.params
    arparams = results_ARIMA.arparams
#    maparams = results_ARIMA.maparams

    resultsar[k] = arparams    
    
    ARIMA_ninosq = copy(nino0[lenseq/4-(len(params)-2):lenseq/4])
    
    i = 0
    resids = np.zeros(len(arparams))
    dicti['ElNino_real'] = dic['ElNino'][:]    
    while(len(ARIMA_ts)<len(dicti['ElNino_real'][:])):
        ARIMA_ts = np.append(ARIMA_ts,weka.forecasts(ARIMA_ninosq,mon,params,resids, ARIMA_ts)[-1])
        ARIMA_ninosq = np.append(ARIMA_ninosq,(dicti['ElNino_real'][i]))  
        if(i>=0):
            resids = np.append(resids, ARIMA_ninosq[-1]-np.array(ARIMA_ts)[-1])    
        i+=1
        
    resid = dicti['ElNino_real'][steps:] - np.array(ARIMA_ts)[:-steps]
    for i in range(len(attributes)):
        dicti[attributes[i]] = dicti[attributes[i]][steps:] 
    
    dicti['ElNino'] = resid
    
    joined.append(dicti)        

    tau = mon/12.0 # The leading time of prediction in years
    #print "calculating tau=", tau

    nn = manip.el_nino_weka_regr(joined[0],t0,deltat,tau) # repares the dataset used for regression problems
 
    name_train = 'train_UU_crossvalid'
    name_test = 'test_UU_crossvalid'
    train_set = root_Dir_write + name_train
    test_set = root_Dir_write + name_test             
		
    p,mintest,maxtest = manip.random_training_test_sets(nn, 100-testset , testset , name_train, name_test , root_Dir_write, pop = pop , typ = 'arff',seed=k)
    result = weka.NN_regression(train_set,test_set,print_feat = p,layers = s,train_time = traintimer) # use ANN with the default layer structure "a" 
    # the default layer structure "a" is (# of attributes + # of classes) / 2, here is layers= [2]
    
    prediction = np.array(result['predicted'][:] + np.array(ARIMA_ts)[-steps-len(result['predicted'][:]):-steps])
    
    RMSE = mean_squared_error(result['predicted'][:],result['actual'][:])**0.5
    error = RMSE/(maxtest-mintest)
    t = nn['t0'][-len(result['predicted'])-1:-1]
    print "RMSE=",error
    
    EE75[k] = error
                
#%% Here for the 20 % testset
testset = 20
print 'testset: ',testset

dicti = copy(dic)
dicti['ElNino_real'] = dic['ElNino'][:]

ARIMA_ts = []
#Determine the testset and train set
size = int(len(dic['ElNino'][:]) * (1-testset/100.))
#init_test = manip.training_test_sets(dic['ElNino'][:], 100, 100-testset , testset , name_train, name_test , root_Dir_write, pop = pop , typ = 'arff') # build test set
train, test = dic['ElNino'][0:size], dic['ElNino'][size+1:len(nino)]
traintime,testtime = dic['date_time'][0:size+1], dic['date_time'][size+2:len(nino)]

joined = []
#%% Now we use the ANN regression for the residual    
m = 0

paramsa = []
maparama = []

resultsar = np.zeros((N,12))
resultsma = np.zeros((N,1))

for k in range(N):
    if(k%10==0):
        print k 
    dicti = copy(dic)
    dicti['ElNino_real'] = dic['ElNino'][:]
    #Determine the testset and train set
    size = int(len(dic['ElNino'][:]) * (1-testset/100.))

    np.random.seed(k)
    tr0 = np.random.randint(0,len(dic['ElNino'][:])-size-1)
    
    train = dic['ElNino'][tr0:tr0+size]
    traintime = dic['date_time'][tr0:tr0+size]
        
    dicti = copy(dic)
  
    model = ARIMA(dicti['ElNino'][tr0:tr0+size],order=ARIMA_order)
    results_ARIMA = model.fit(disp=0)
    params = results_ARIMA.params
    arparams = results_ARIMA.arparams

    resultsar[k] = arparams    
    
    ARIMA_ninosq = copy(nino0[lenseq/4-(len(params)-2):lenseq/4])
    
    i = 0
    resids = np.zeros(len(arparams))
    dicti['ElNino_real'] = dic['ElNino'][:]    
    while(len(ARIMA_ts)<len(dicti['ElNino_real'][:])):
        ARIMA_ts = np.append(ARIMA_ts,weka.forecasts(ARIMA_ninosq,mon,params,resids, ARIMA_ts)[-1])
        ARIMA_ninosq = np.append(ARIMA_ninosq,(dicti['ElNino_real'][i]))  
        if(i>=0):
            resids = np.append(resids, ARIMA_ninosq[-1]-np.array(ARIMA_ts)[-1])    
        i+=1
        
    resid = dicti['ElNino_real'][steps:] - np.array(ARIMA_ts)[:-steps]
    for i in range(len(attributes)):
        dicti[attributes[i]] = dicti[attributes[i]][steps:]
    dicti['ElNino'] = resid
    
    joined.append(dicti)        

    tau = mon/12.0 # The lead time of prediction in years

    nn = manip.el_nino_weka_regr(joined[0],t0,deltat,tau) # repares the dataset used for regression problems
 
    name_train = 'train_UU_crossvalid'
    name_test = 'test_UU_crossvalid'
    train_set = root_Dir_write + name_train
    test_set = root_Dir_write + name_test             
		
    p,mintest,maxtest = manip.random_training_test_sets(nn, 100-testset , testset , name_train, name_test , root_Dir_write, pop = pop , typ = 'arff',seed=k)
    result = weka.NN_regression(train_set,test_set,print_feat = p,layers = s,train_time = traintimer) # use ANN with the default layer structure "a" 
    # the default layer structure "a" is (# of attributes + # of classes) / 2, here is layers= [2]
    
    prediction = np.array(result['predicted'][:] + np.array(ARIMA_ts)[-steps-len(result['predicted'][:]):-steps])
    
    RMSE = mean_squared_error(result['predicted'][:],result['actual'][:])**0.5
    error = RMSE/(maxtest-mintest)
    t = nn['t0'][-len(result['predicted'])-1:-1]
    print "RMSE=",error
    
    EE80[k] = error

#%% Here for the 15 % testset 
testset = 35
print 'testset: ',testset

dicti = copy(dic)
dicti['ElNino_real'] = dic['ElNino'][:]

ARIMA_ts = []

#Determine the testset and train set
size = int(len(dic['ElNino'][:]) * (1-testset/100.))
train, test = dic['ElNino'][0:size], dic['ElNino'][size+1:len(nino)]
traintime,testtime = dic['date_time'][0:size+1], dic['date_time'][size+2:len(nino)]

joined = []
#%% Now we use the ANN regression for the residual    
#print 'keys: ',joined[0].keys()
m = 0
paramsa = []
maparama = []

resultsar = np.zeros((N,12))
resultsma = np.zeros((N,1))

for k in range(N):
    if(k%10==0):
        print k     
    dicti = copy(dic)
    dicti['ElNino_real'] = dic['ElNino'][:]
    #Determine the testset and train set
    size = int(len(dic['ElNino'][:]) * (1-testset/100.))

    np.random.seed(k)
    tr0 = np.random.randint(0,len(dic['ElNino'][:])-size-1)
    train = dic['ElNino'][tr0:tr0+size]
    traintime = dic['date_time'][tr0:tr0+size]
        
    dicti = copy(dic)
  
    model = ARIMA(dicti['ElNino'][tr0:tr0+size],order=ARIMA_order)
    results_ARIMA = model.fit(disp=0)
    params = results_ARIMA.params
    arparams = results_ARIMA.arparams

    resultsar[k] = arparams
  
    ARIMA_ninosq = copy(nino0[lenseq/4-(len(params)-2):lenseq/4])
    
    i = 0
    resids = np.zeros(len(arparams))
    dicti['ElNino_real'] = dic['ElNino'][:]    
    while(len(ARIMA_ts)<len(dicti['ElNino_real'][:])):
        ARIMA_ts = np.append(ARIMA_ts,weka.forecasts(ARIMA_ninosq,mon,params,resids, ARIMA_ts)[-1])
        ARIMA_ninosq = np.append(ARIMA_ninosq,(dicti['ElNino_real'][i]))  
        if(i>=0):
            resids = np.append(resids, ARIMA_ninosq[-1]-np.array(ARIMA_ts)[-1])    
        i+=1
        
    resid = dicti['ElNino_real'][steps:] - np.array(ARIMA_ts)[:-steps]
    for i in range(len(attributes)):
        dicti[attributes[i]] = dicti[attributes[i]][steps:]

    dicti['ElNino'] = resid
    
    joined.append(dicti)        

    tau = mon/12.0 # The lead time of prediction in years

    nn = manip.el_nino_weka_regr(joined[0],t0,deltat,tau) # repares the dataset used for regression problems
 
    name_train = 'train_UU_crossvalid'
    name_test = 'test_UU_crossvalid'
    train_set = root_Dir_write + name_train
    test_set = root_Dir_write + name_test             
		
    p,mintest,maxtest = manip.random_training_test_sets(nn, 100-testset , testset , name_train, name_test , root_Dir_write, pop = pop , typ = 'arff',seed=k)
    result = weka.NN_regression(train_set,test_set,print_feat = p,layers = s,train_time = traintimer) # use ANN with the default layer structure "a" 
    # the default layer structure "a" is (# of attributes + # of classes) / 2, here is layers= [2]
    
    prediction = np.array(result['predicted'][:] + np.array(ARIMA_ts)[-steps-len(result['predicted'][:]):-steps])
    
    RMSE = mean_squared_error(result['predicted'][:],result['actual'][:])**0.5
    error = RMSE/(maxtest-mintest)
    t = nn['t0'][-len(result['predicted'])-1:-1]
    print "RMSE=",error
    
    EE85[k] = error


#%% Saving the RMSE's for the different testsets

np.save(root_Dir_write + '/crossval/' + 'c_trainset85.txt',EE85)
np.save(root_Dir_write+ '/crossval/' + 'c_trainset80.txt',EE80)
np.save(root_Dir_write+ '/crossval/' + 'c_trainset75.txt',EE75)
np.save(root_Dir_write+ '/crossval/' + 'c_trainset70.txt',EE70)


df85 = pd.DataFrame(EE85)
df80 = pd.DataFrame(EE80)
df75 = pd.DataFrame(EE75)
df70 = pd.DataFrame(EE70)
df85.plot(kind='density')
df80.plot(kind='density')
df75.plot(kind='density')
df70.plot(kind='density')
plt.show()
