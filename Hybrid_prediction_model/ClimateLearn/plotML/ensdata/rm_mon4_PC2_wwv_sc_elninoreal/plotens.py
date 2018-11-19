# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:12:34 2017

@author: Peter Nooteboom
"""

import numpy as np
import matplotlib.pylab as plt

ensemble10 = np.load('ens10_keys_date_time_ElNino_PC2_seasonal_cycle_wwv_ElNino_realARIMA.npy')

for i in range(10):

#    plt.plot(ensemble10[i]['prediction'])
#    plt.plot(ensemble10[i]['actual'])
#    plt.title('size: '+str(ensemble10[i]['s'])+' and RMSE: ' + str(ensemble10[i]['NRMSE']))
#    plt.show()

    plt.plot(ensemble10[i]['prediction']-ensemble10[i]['actual'])
    plt.show()