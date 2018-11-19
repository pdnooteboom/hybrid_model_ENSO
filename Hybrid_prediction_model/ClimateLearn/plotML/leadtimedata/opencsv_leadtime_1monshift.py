# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:06:21 2017

This script opens, plots and saves the .csv files which
contain the 3 and 6 month lead time prediction of the CFSv2 
prediction. 

@author: peter
"""

import csv
import numpy as np
import matplotlib.pylab as plt



#%%
x = []
y = []
 
til2015 = True

ifile = open('obs_2006-2012.csv', 'rb')
reader = csv.reader(ifile)

for row in reader:
    x.append(row[0])
    y.append(row[1])

if(til2015):    
    ifile = open('obs_2012-2015.csv', 'rb')
    reader = csv.reader(ifile)
    
    for row in reader:
#        if(float(row[0])<2014.5):
        x.append(row[0])
        y.append(row[1])       
    
obs = np.array(y)
obsx = np.array(x)


#%%
 
x = []
y = []
 

ifile = open('lead3_2006-2012.csv', 'rb')


reader = csv.reader(ifile)

for row in reader:
    x.append(row[0])
    y.append(row[1])
    
if(til2015):    
    ifile = open('lead3_2012-2015.csv', 'rb')
    reader = csv.reader(ifile)
    
    for row in reader:
        print row
#        if(float(row[0])<2014.5):
        x.append(row[0])
        y.append(row[1])        
        
monl3 = np.array(y)
monl3x = np.array(x)

#%%
 
x = []
y = []

ifile = open('lead6_2006_2012.csv', 'rb')

reader = csv.reader(ifile)

for row in reader:
    x.append(row[0])
    y.append(row[1])
    
if(til2015):    
    ifile = open('lead6_2012-2015.csv', 'rb')
    reader = csv.reader(ifile)
    
    for row in reader:
        #if(float(row[0])<2014.5):
        x.append(row[0])
        y.append(row[1])        
    
monl6 = np.array(y)
monl6x = np.array(x)

#%%



plt.plot(obsx,obs,'k')
plt.plot(monl6x,monl6,'g')
plt.plot(monl3x,monl3,'b')
plt.show()

if(til2015):
    ifile = np.save('6monl_2006to2015_NCEP.npy', monl6)    
    ifile = np.save('time6_2006to2015_NCEP.npy', monl6x)   
    ifile = np.save('timeobs_2006to2015_NCEP.npy', obsx)       
    ifile = np.save('obs_2006to2015_NCEP.npy', obs)     
    ifile = np.save('3monl_2006to2015_NCEP.npy', monl3)
    ifile = np.save('time3_2006to2015_NCEP.npy', monl3x)     
else:
    ifile = np.save('6monl_2006to2012_NCEP.npy', monl6)
    ifile = np.save('3monl_2006to2012_NCEP.npy', monl3)
    ifile = np.save('time6_2006to2012_NCEP.npy', monl6x)   
    ifile = np.save('time3_2006to2012_NCEP.npy', monl3x)   
    ifile = np.save('timeobs_2006to2012_NCEP.npy', obsx)       
    ifile = np.save('obs_2006to2012_NCEP.npy', obs)       
