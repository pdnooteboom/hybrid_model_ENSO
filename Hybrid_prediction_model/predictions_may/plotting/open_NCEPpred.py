# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 12:06:33 2017

@author: Peter Nooteboom
"""

import csv
import numpy as np
import matplotlib.pylab as plt



#%%
y = []
 


ifile = open('ncep_may.csv', 'rb')
reader = csv.reader(ifile)

for row in reader:
    y.append(row[1])
    
np.save('NCEPpred.npy',np.array(y))

y = []

ifile = open('upper_bound.csv', 'rb')
reader = csv.reader(ifile)

for row in reader:
    y.append(row[1])
    
np.save('upper_bound.npy',np.array(y))

y = []

ifile = open('lower_bound.csv', 'rb')
reader = csv.reader(ifile)

for row in reader:
    y.append(row[1])
    
np.save('lower_bound.npy',np.array(y))