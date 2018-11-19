# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:13:59 2016

In this model, first the ARIMA model is fitted to the nino series. Then this 
sequence is used as input for the ANN

@author: Peter Nooteboom
"""
import numpy as np
import matplotlib.pylab as plt
from copy import copy
import el_nino_manip as manip
import el_nino_weka as weka 

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

#To include in the machine learning input:
bPC2 = True
bseasonal_cycle = True
bwwv = True
brunning_mean = True

testset = 4 #(%)
print 'testset: ',testset

monlow = 1; monhigh = 7; monlen = monhigh - monlow;
print 'month ',monlow,' to ',monhigh-1,' ahead'

#The ARIMA order:                
ARIMA_ordera = [(12,1,0)]
                

traintimer = 700 #Amount of epochs to train

s = [2,1,1]   #ANN structure              
#%% Load the attributes  
#Always include ENSO index and the time 
#ARIMA_ninosq is the NINO3.4 data before the used dataset.
#nino is the NINO3.4 in the TtimesN matrix
ARIMA_ninosq = np.loadtxt('ENSO1975-1981.txt')
ARIMA_ninosq = np.delete(ARIMA_ninosq,0,1)
ARIMA_ninosq0 = np.reshape(ARIMA_ninosq,ARIMA_ninosq.size)
ARIMA_ninosq = ARIMA_ninosq0[:-24]
nino = np.append(ARIMA_ninosq0[-24:],np.loadtxt('ENSO1982-052017.txt')[:,9])

time = np.linspace(1980+1/12.,2017.333333333333333333333333,len(nino))

seascycle = -0.1*np.cos(time*2*np.pi)

attr = ['date_time','ElNino']

if(bPC2):
    attr.append('PC2')    
    PC2 = np.array(np.load('secondPC_WWB_weekly.npy'))
    if(brunning_mean):       
        PC2 = running_mean(PC2, 3)     
    #PC2 = PC2[lenseq:lenseq+lensliding*int(sq):int(sq)]           
if(bseasonal_cycle):
    attr.append('seasonal_cycle')
if(bwwv):
    attr.append('wwv')
    wwv = np.loadtxt('wwv1980-062017.txt')
    timeni = np.linspace(1980,2017.5,len(wwv[:,2]))
    wwv = np.interp(time,timeni,wwv[:,2])/(10**14)

lagtot = 0
minlen = min(len(nino),len(time))

attributes = ['date_time','ElNino']    
    
joined = np.zeros((minlen-lagtot,1))
for i in range(minlen-lagtot):
    joined[i,0] = time[lagtot+i] #joined.append(time)  
joined = np.append(joined,np.reshape(nino[lagtot:],(minlen-lagtot,1)),axis=1)
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
from statsmodels.tsa.arima_model import ARIMA

length = len(dic['ElNino'][:])

size = int(len(dic['ElNino'][:]) * (1-testset/100.))
train, test = dic['ElNino'][0:size], dic['ElNino'][size:len(nino)]
traintime,testtime = dic['date_time'][0:size], dic['date_time'][size:len(nino)]

ARIMA_ts = [[]]*monlen

#Determine the testset and train set
resid = [[]]*monlen

joined = []

for m in range(monlen):
    print 'month: ',m+monlow
    dicti = copy(dic)
    mon = m + monlow
    
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
        ARIMA_ts1 = np.append(ARIMA_ts1,weka.forecasts(ARIMA_ninosq,1,params,resids)[-1])
        
        ARIMA_ninosq = np.append(ARIMA_ninosq,(dicti['ElNino_real'][i]))  
        if(i>=1):
            resids = np.append(resids, ARIMA_ninosq[-1]-np.array(ARIMA_ts1)[-1])    
        i+=1
        
    steps = steps - 1
    
    #Define the residual, to be predicted with ANN
    if(steps!=0):
        resid[m] = dicti['ElNino'][steps:] - np.array(ARIMA_ts[m])[:-steps]
    else:
        resid[m] = dicti['ElNino'] - np.array(ARIMA_ts[m])

    for i in range(len(attributes)):
        dicti[attributes[i]] = dicti[attributes[i]][steps:]

    print 'mean resid: ',np.mean(resid[m])
    

    dicti['ElNino'] = resid[m]
    joined.append(dicti)
#%%
attributes = np.append(attributes,'ElNino_real')
#%% Now we use the ANN regression for the residual    
#Setting home directory for writing
root_Dir_write = 'C:/Users/User/Documents/Thesis/Hybrid_prediction_model/predictions_may/1980-may/prediction'

# create a weka instance-friendly file with given parameters
t0 = 0.#1960.0  # starting date
deltat = 0.0 # start from which data point, in general we just use the first data point

res = []
time = []
actual = []
            
print 'keys: ',joined[0].keys()

#pop = np.array(['t0-deltat','ElNino_0','c2_0','PC2_0','wwv','seasonal_cycle_0'])

avg_RMSE = np.zeros(monlen) #The average RMSE of the ensemble

min_RMSE = [np.inf]*monlen
best_res = [[]]*monlen
best_NN_size = [[]]*monlen

firstlen = 2; secondlen = 1; thirdlen = 1; fourthlen = 1;
size_ens = (firstlen-1)*(secondlen)*(thirdlen)*(fourthlen)-(firstlen-1)*(thirdlen-1)
count = 0.
        
#print 'shape of the neural network: ',s
for i in range(1,firstlen):
    print 'i = ',i
    for j in range(0,secondlen):
        print 'j = ',j
        for k in range(0,thirdlen):
                        
            if(j==0 and k>0):
                break
            for l in range(0,fourthlen):
                if(k==0 and l>0):
                    break
                if __name__ == "__main__":
                    for mon in range(monlow,monhigh):
                        #Choose an ANN structure for every leading time
                        if(mon==1):
                            s = [4,2,4]#[1]#
                        elif(mon==2):
                            s = [2,3,4]#[2,1,1]#
                        elif(mon==3):
                            s = [3,3,3]#[2,4,2],[2,1,1],[3,1,4]
                        elif(mon==4):
                            s = [3,4,3]#[4,3,1]#[4,4,4]#,[3,4,3],[4,3,2],[3,4,4]
                        elif(mon==5):
                            s = [4,4]#[4,3,4],[4,1,1],[3]
                        elif(mon==6):
                            s = [4,2,2]#[4,1,2],[3,3]
#                        elif(mon==7):
#                            s = [2,3,3]
#                        elif(mon==8):
#                            s = [3,1,3]
#                        elif(mon==9):
#                            s = [3,2]

                        steps = mon-1
                        tau = mon/12.0 # The lead time of prediction in years
                    
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
                            min_RMSE[mon-monlow] = error#weka._NorRMSE(actual[mon-monlow],np.array(result['predicted'][:]) + np.array(ARIMA_ts[mon-monlow])[-len(result['predicted'][:]):]) 
                            best_res[mon-monlow] = prediction#np.array(result['predicted'][:]) + np.array(ARIMA_ts[mon-monlow])[-len(result['predicted'][:]):]
                            best_NN_size[mon-monlow] = s
                        
                    print 'pop: ',pop
                    print 'keys: ',joined[m].keys()
                    
                    pprediction = np.full(best_res[0].shape,np.float('nan'))
                    pactual = actual[monlen-1]
                    time = time + 1
                    
                    for i in range(monlen):
                        pprediction[-monlen+i] = np.array(best_res[i])[-1]
                     
                    plt.plot(time[monlow-1],pactual,'r--',label = "actual")
                    plt.plot(time[monlow-1],pprediction,'b',label = "predicted")
                    plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
                    plt.xlabel("time (years)")
                    plt.ylabel("NINO3.4")
                    plt.title(" ENSO predition " )
                    plt.show()
                else:
                    continue
            else:
                continue  


#%%           	

print 'ensemble size:', size_ens
for i in range(len(avg_RMSE)):
    avg_RMSE[i] = avg_RMSE[i]/float(count)

#And the results with minimal RMSE:  
print 'best ANN structure: ',best_NN_size             
for i in range(monlen):
    plt.plot(time[i],actual[i],'r--',label = "actual")
    plt.scatter(time[i],np.array(best_res[i]))#,'b',label = "predicted")
    plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
    plt.xlabel("time (years)")
    plt.ylabel("NINO3.4")
    plt.title(" ENSO predition " + str(monlow+i) + " months ahead (RMSE = " + str(min_RMSE[i]) + " )")
    stri = ''
    for j in range(len(attributes)):
        stri += '_' + attributes[j]
    plt.show()

pactual[-6:] = np.float('nan') 
#pprediction[:-6] = np.float('nan')  
pprediction[-7] = pactual[-7] 
    #plot prediction of the coming 9 months:
plt.plot(time[monlow-1],pactual,'r--',label = "actual")
plt.plot(time[monlow-1],pprediction,'b',label = "predicted")
plt.scatter(time[monlow-1],pprediction)
plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
plt.xlabel("time (years)")
plt.ylabel("NINO3.4")
plt.title(" ENSO predition " )
plt.show()             

for i in range(monlow,monhigh):
    np.save(root_Dir_write + 'pred_mon{}.npy'.format(i),np.array(best_res[i-monlow]))
np.save(root_Dir_write + 'actual.npy',pactual)
