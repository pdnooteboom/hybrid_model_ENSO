# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:12:34 2017

@author: Peter Nooteboom
"""

import statsmodels.stats.stattools as stt
import statsmodels.stats.diagnostic as st2

import numpy as np
import matplotlib.pylab as plt

ensemble10 = np.load('ens10_keys_date_time_ElNino_c2_PC2_seasonal_cycle_ElNino_realARIMA.npy')

for i in range(10):
#
    plt.plot(ensemble10[i]['prediction'])
    plt.plot(ensemble10[i]['actual'])
    plt.title('size: '+str(ensemble10[i]['s'])+' and RMSE: ' + str(ensemble10[i]['NRMSE']))
    plt.show()
    
    plt.plot(ensemble10[i]['prediction']-ensemble10[i]['actual'])
    plt.show()
    
    rt = stt.durbin_watson(ensemble10[i]['prediction']-ensemble10[i]['actual'])    
    lb = st2.acorr_ljungbox(ensemble10[i]['prediction']-ensemble10[i]['actual'], lags=3)    
    print rt