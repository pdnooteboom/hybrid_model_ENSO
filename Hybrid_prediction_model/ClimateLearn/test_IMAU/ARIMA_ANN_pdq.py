# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:13:59 2016

Hindcast for several p,d,q parameters.

@author: Peter Nooteboom
"""
import numpy as np
import matplotlib.pylab as plt
from copy import copy
import el_nino_manip as manip
import el_nino_weka as weka 

def running_mean(l, N):
    #function to calculate N-time running mean of list l
    sum = 0
    result = list( 0 for x in l)

    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)

    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N

    return result
#%% Hyperparameters and attributes to include
#Attributes to include in the machine learning input:
bc2 = True
bPC2 = True
bseasonal_cycle = True
bwwv = False
brunning_mean = True  #3 month running mean of all input variables    

ens10 = True # If true save the ensemble of 10 best

testset = 20
print 'testset: ',testset

#minimum and maximum prediction lead time
monlow = 12; monhigh = monlow + 1; monlen = monhigh - monlow;
print 'month ',monlow,' to ',monhigh-1,' ahead'

traintimer = 700#Amount of epochs the algorithm trains:
#%% Load the attributes
#Root where to read data from:
root_Dir_read = 'C:/Users/User/Documents/Thesis/Code_in_thesis/ClimateLearn/input/'

#Include ENSO index and the time (monthly)
nino34 = np.loadtxt(root_Dir_read+'monthlynino34.txt')
nino34 = np.delete(nino34,0,1)
nino0 = np.reshape(nino34,nino34.size)
nino = nino0[12:]
time = np.load(root_Dir_read + 'time.npy')
    
#Define seasonal cycle:
seascycle = -0.1*np.cos(time*2*np.pi)

#list containing the considered attributes
attr = ['date_time','ElNino']
  
#Load inputvariables/ attributes, standardize them, interpolate on correct(monthly) time interval,
# Calculate the running mean  
if(bc2):
    sdata1=1827
    lenseq = 52
    nslides=1
    sq=4.0
    lensliding=(sdata1-lenseq)/int(sq)
    #time of c2(weekly):
    timec2 = np.linspace(1979+lenseq/52.,1979+(lenseq+lensliding*sq)/52.,lensliding*nslides)       
    
    attr.append('c2')    
    c2 = np.load(root_Dir_read + 'c2_h_th0.9.pny.npy')
    c2 = (c2 - np.mean(c2))/np.nanstd(c2)
    c2 = np.interp(time,timec2,c2)  
    if(brunning_mean):
        c2 = running_mean(c2,3)
if(bPC2):
    attr.append('PC2')    
    PC2 = np.load(root_Dir_read +'secondPC_WWB_weekly.npy')
    PC2 = PC2[lenseq::4]
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
#Add all attributes in the matrix 'joined'   
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
#%% ARIMA optimisation and prediction 
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA

#Choose the p,d,q parameters here
for p1 in range(12,13):
    for d in range(1,2):
        for q in range(0,1):
            
            #Determine the testset and train set
            size = int(len(dic['ElNino'][:]) * (1-testset/100.))
            train, test = dic['ElNino'][0:size], dic['ElNino'][size:len(nino)]
            traintime,testtime = dic['date_time'][0:size], dic['date_time'][size:len(nino)]

            #Array containg the ARIMA predictions
            ARIMA_ts = [[]]*monlen
            
            #Array containing the residual between ARIMA sequence and observed NINO3.4
            resid = [[]]*monlen            
            joined = []
                      
            #The order of ARIMA     
            ARIMA_ordera = [(p1,d,q)]
            
            for m in range(monlen):
                print 'month: ',m+monlow    #lead time prediction
                dicti = copy(dic)
                mon = m + monlow
                ARIMA_order = ARIMA_ordera[0]
              
                #'Train' the ARIMA model and retreive the residual
                model = ARIMA(dicti['ElNino'][:size+1],order=ARIMA_order)
                results_ARIMA = model.fit(disp=0)
                params = results_ARIMA.params #all ARIMA parameters
                arparams = results_ARIMA.arparams # AR parameters
                maparams = results_ARIMA.maparams #MA parameters
            
            #    plt.plot(dicti['ElNino'][:size+1])
            #    plt.plot(results_ARIMA.fittedvalues,color='red')
            #    plt.title('RSS:%d'% sum((results_ARIMA.fittedvalues-dicti['ElNino'][:size])**2))
            #    plt.show()
                #residual errors
                residuals = DataFrame(results_ARIMA.resid)
            #    residuals.plot()
            #    plt.show()
            #    residuals.plot(kind='kde')
            #    plt.show()
            #%% prepare for the ANN regression, which will be optimized to predict the
                #residual between the ARIMA prediction and observations :
            #Amount of steps prediction ahead
                steps = mon
                ARIMA_ninosq = copy(nino0[lenseq/4-(len(params)-2):lenseq/4])
            
                ARIMA_ts1 = np.array([]) 
                    
                i = 0
                #Array keeping track of past residuals to apply the AR part
                resids = np.zeros(len(arparams))
                #Apply ARIMA until reaching the end of the sequence
                while(len(ARIMA_ts[m])<len(dicti['ElNino'][:])):
                    ARIMA_ts[m] = np.append(ARIMA_ts[m],weka.forecasts(ARIMA_ninosq,steps,params,resids,ARIMA_ts[m])[-1])                   
                    ARIMA_ninosq = np.append(ARIMA_ninosq,(dic['ElNino'][i]))  
                    if(i>=1):
                        resids = np.append(resids, ARIMA_ninosq[-1]-np.array(ARIMA_ts)[-1])    
                    i+=1
                    
                steps = steps - 1
            
                #Calculate Normalized Root Mean Squared Error
                if(steps!=0):
                    nrmse_ARIMA = weka._NorRMSE(np.array(ARIMA_ts[m])[:-steps],dicti['ElNino'][steps:])     
                else:
                    nrmse_ARIMA = weka._NorRMSE(np.array(ARIMA_ts[m]),dicti['ElNino'])     
            
                if(steps!=0):
                    plt.plot(np.array(ARIMA_ts[m])[:-steps])
                    plt.plot(dicti['ElNino'][steps:])
                else:
                    plt.plot(np.array(ARIMA_ts[m]))
                    plt.plot(dicti['ElNino'])        
                plt.title('NRMSE:' + str(nrmse_ARIMA))
                plt.show() 
                
                #Define the residual, to be predicted with ANN
                if(steps!=0):
                    resid[m] = dicti['ElNino'][steps:] - np.array(ARIMA_ts[m])[:-steps]
                else:
                    resid[m] = dicti['ElNino'] - np.array(ARIMA_ts[m])
            
                for i in range(len(attributes)):
                    dicti[attributes[i]] = dicti[attributes[i]][steps:]
            
                print 'mean resid: ',np.mean(resid[m])
                
                dicti['ElNino_real'] = dicti['ElNino'][:]
                dicti['ElNino'] = resid[m]
            
                joined.append(dicti)
            
            #%%
            attributes = np.append(attributes,'ElNino_real')
            
            #%% Now we use the ANN regression for the residual    
            #Setting home directory for reading and writing
            root_Dir_write = 'C:/Users/User/Documents/Thesis/Hybrid_prediction_model/ClimateLearn/output/IMAU/'    
                    
            # create a weka instance-friendly file with given parameters
            t0 = 0.#1960.0  # starting date
            deltat = 0.0 # start from which data point, in general we just use the first data point
            
            res = []
            time = []
            actual = []
            
            print 'keys: ',joined[0].keys()
                        
            avg_RMSE = np.zeros(monlen) #The average RMSE of the ensemble         
            min_RMSE = [np.inf]*monlen
            best_res = [[]]*monlen
            best_NN_size = [[]]*monlen
            
            #Length of the different layers of the ANN to be considered:
            firstlen = 5
            secondlen = 5
            thirdlen = 5
            fourthlen = 1
            size_ens = (firstlen-1)*(secondlen)*(thirdlen)-(firstlen-1)*(thirdlen-1)
            count = 0.
            
            ensemble10 = np.array([{},{},{},{},{},{},{},{},{},{}]) #Keeps track of the ensemble of 10 lowest RMSE
            RMSE = np.arange(1,11)*10.  #keeps track of all RMSE's
            for i in range(len(ensemble10)):
                ensemble10[i]['NRMSE'] = i*10. #Initially the RMSE must be large of every member of the top 10 ensemble
            
            for i in range(1,firstlen):
                for j in range(0,secondlen):
                    for k in range(0,thirdlen):
                        if(j==0 and k>0):
                            break
                        for l in range(0,fourthlen):
                            #print i,j,k,l
                            if(k==0 and l>0):
                                break
                            if __name__ == "__main__":
                                if(j==0):
                                    s = [i]
                                elif(k==0):
                                    s=[i,j]   #The layer structure for the MLP
                                elif(l==0):
                                    s=[i,j,k]
                                else:
                                    s=[i,j,k,l]
                                for mon in range(monlow,monhigh): # test different leading time
                                    print 'mon: ',mon
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
                                    
                                    if(ens10):
                                        imax = np.where(RMSE==max(RMSE))[0][0]
                                        if(RMSE[imax]>error and (np.absolute(prediction)<5.).all()):
                                            RMSE[imax] = error
                                            ensemble10[imax]['prediction'] = prediction 
                                            ensemble10[imax]['NRMSE'] = error
                                            ensemble10[imax]['s'] = s
                                            ensemble10[imax]['actual'] =  joined[mon-monlow]['ElNino_real'][-len(result['predicted']):]                       
                                            ensemble10[imax]['time'] =  t
                                    
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
                                                
                                        avg_RMSE[mon-monlow] += error
            
            #                        plt.plot(time[mon-monlow],actual[mon-monlow],'r--',label = "actual")
            #                        plt.plot(time[mon-monlow],np.array(prediction),'b',label = "predicted")
            #                        #plt.xlim(time[i][0],t[len(t)-1])
            #                        plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
            #                        #plt.xlim(2004, 2015)
            #                        #plt.ylim(-3.3, 2.0)
            #                        plt.xlabel("time (years)")
            #                        plt.ylabel("NINO3.4")
            #                        plt.title(" ENSO predition " + str(monlow) + " months ahead (RMSE = " + str(error) + ' size: '+str(s)+ " )")
            #                        plt.show()
                                        
                                    if(error<min_RMSE[mon-monlow]):
                                        min_RMSE[mon-monlow] = error#weka._NorRMSE(actual[mon-monlow],np.array(result['predicted'][:]) + np.array(ARIMA_ts[mon-monlow])[-len(result['predicted'][:]):]) 
                                        best_res[mon-monlow] = prediction#np.array(result['predicted'][:]) + np.array(ARIMA_ts[mon-monlow])[-len(result['predicted'][:]):]
                                        best_NN_size[mon-monlow] = s
                                count += 1.
                        else:
                            continue
                    else:
                        continue
            
            if(ens10):
                ens = ensemble10[0]['prediction']
                for i in range(1,10):
                    ens +=  ensemble10[i]['prediction']  
                ens = ens / float(len(ensemble10))      
                ensRMSE = weka._NorRMSE(ensemble10[0]['actual'] ,ens)
                           
            print 'ensemble size:', size_ens
            for i in range(len(avg_RMSE)):
                avg_RMSE[i] = avg_RMSE[i]/float(count)
             
            
            #%%           	
            # plot
            for i in range(monlen):
                plt.plot(time[i],actual[i],'r--',label = "actual")
                plt.plot(time[i],np.array(res[i])/float(count),'b',label = "predicted")
                plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
                plt.xlabel("time (years)")
                plt.ylabel("NINO3.4")
                plt.title(" ENSO prediction " + str(monlow+i) + " mon ahead (avgRMSE= " + str(avg_RMSE[i]) +', RMSE='+ str(weka._NorRMSE(actual[i],np.array(res[i])/float(size_ens)))+" )")
                stri = ''
                for j in range(len(attributes)):
                    stri += '_' + attributes[j]
                plt.show()
             
            #And the results with minimal RMSE:  
            print 'best ANN structure: ',best_NN_size             
            for i in range(monlen):
                plt.plot(time[i],actual[i],'r--',label = "actual")
                plt.plot(time[i],np.array(best_res[i]),'b',label = "predicted")
                plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
                plt.xlabel("time (years)")
                plt.ylabel("NINO3.4")
                plt.title(" ENSO predition " + str(monlow+i) + " months ahead (RMSE = " + str(min_RMSE[i]) + " )")
                stri = ''
                for j in range(len(attributes)):
                    stri += '_' + attributes[j]
                plt.show()
                
            if(ens10):    
                plt.plot(ensemble10[0]['time'],ensemble10[0]['actual'],'r--',label = "actual")
                plt.plot(ensemble10[0]['time'],ens,'b',label = "predicted")
                plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
                plt.xlabel("time (years)")
                plt.ylabel("NINO3.4")
                plt.title(" 10 best of ensemble ENSO predition " + str(monlow) + " months ahead  (avgRMSE= " + str(np.mean(RMSE)) +', RMSE='+ str(ensRMSE)+ " )")
                stri = ''
                for j in range(len(attributes)):
                    stri += '_' + attributes[j]
                plt.show()    
            
            if(ens10):
                np.save('pdq/p/lead' + str(monlow) + '/best/best_keys'+ stri+ 'size' + str(best_NN_size[0]) +'ARIMA' + str(ARIMA_order) +'.npy',np.array(best_res))
                np.save('pdq/p/lead' + str(monlow) + '/ensemble/ensemble_keys'+ stri +'ARIMA' + str(ARIMA_order) +'.npy',np.array(res)/size_ens)
                np.save('pdq/p/lead' + str(monlow) + '/ensemble/actual_ARIMA' + str(ARIMA_order) + '.npy',actual)
                np.save('pdq/p/lead' + str(monlow) + '/best/actual_ARIMA' + str(ARIMA_order) + '.npy',actual)
                np.save('pdq/p/lead' + str(monlow) + '/ensemble/actual_ARIMA' + str(ARIMA_order) + '.npy',time)
                np.save('pdq/p/lead' + str(monlow) + '/best/actual_ARIMA' + str(ARIMA_order) + '.npy',time)
            
            if(ens10):
                np.save('pdq/p/lead' + str(monlow) + '/ens10/ens10_keys'+ stri +'ARIMA'  + str(ARIMA_order)  +'.npy',ensemble10)
            attributes = attributes[:-1]