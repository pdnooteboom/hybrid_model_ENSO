# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:24:19 2018

@author: nooteboom
"""

from __future__ import print_function
import os
from datetime import datetime, timedelta

from2015 = False


startTime = datetime.now()

# here we define all the bits and pieces that get put in the 'runcommand' variable used to call the motu-client
path2motuClient = '/Users/nooteboom/Documents/MA/running_predictions/motu-client-python-motuclient-python-1.8.1/src/python/'

# here we specifiy username and password to CMEMS
usrname = 'pnooteboom'
passwd = 'Nootje01!'

# here
# [lon-lower-left-corner, lon-upper-right-corner, lat-lower-left-corner, lat-upper-right-corner]
domain = [140, 280, -20, 20]

if(from2015):
    startDate = datetime(2017,1,7)
    endDate = datetime(2018,12,7)
else:
    startDate = [datetime(1993,1,1),datetime(1999,1,2),datetime(2005,1,2),datetime(2011,1,2)]
    endDate = [datetime(1999,1,1),datetime(2005,1,1),datetime(2011,1,1),datetime(2017,1,6)]#[datetime(1994,1,6)]#

#    startDate = [datetime(1993,1,1),datetime(1999,1,2),datetime(2005,1,2),datetime(2010,1,2)]
#    endDate = [datetime(1999,1,1),datetime(2005,1,1),datetime(2010,1,1),datetime(2017,1,1)]
#  zos = SSH in m, uo = Eastward velocity in m/s, vo = Northward velocity in m/s
varList = [ 'sla']
varStr = ''
for var in varList:
    varStr += ' --variable '+var

# NOTE only surface fields available hourly
depths = [0.0,0.0]#[0.493, 0.4942]

path2saveData = os.getcwd()+'/'
if(from2015):
    fname = 'DUACS_from2017.nc'
else:
    fname = 'DUACS_from1993'
    
#http://nrtcmems.mercator-ocean.fr/motu-web/Motu
#GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS
#global-analysis-forecast-phy-001-024-weekly-ssh
# create the runcommand string
if(from2015):
    runcommand = 'python '+path2motuClient+'/motuclient.py --quiet'+ \
            ' --user '+usrname+' --pwd '+passwd+ \
            ' --motu http://nrt.cmems-du.eu/motu-web/Motu'+ \
            ' --service-id SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046-TDS'+ \
            ' --product-id dataset-duacs-nrt-global-merged-allsat-phy-l4'+ \
            ' --longitude-min '+str(domain[0])+' --longitude-max '+str(domain[1])+ \
            ' --latitude-min '+str(domain[2])+' --latitude-max '+str(domain[3])+ \
            ' --date-min "'+str(startDate.strftime('%Y-%m-%d %H:%M:%S'))+'" --date-max "'+str(endDate.strftime('%Y-%m-%d %H:%M:%S'))+'"'+ \
            ' --depth-min '+str(depths[0])+' --depth-max '+str(depths[1])+ \
            varStr+ \
            ' --out-dir '+path2saveData+' --out-name '+fname
    # run the runcommand, i.e. download the data specified above
    print('fetching latest mercator ocean forecast from CMEMS and making datastack')
    
    print (runcommand)
    
    os.system(runcommand)

    print('Time exceeded:   ',datetime.now() - startTime)
else:
    for le in range(len(startDate)):
        print('dates: '+ str(startDate[le])+'     ' + str(endDate[le]))
        runcommand = 'python '+path2motuClient+'/motuclient.py --quiet'+ \
                ' --user '+usrname+' --pwd '+passwd+ \
                ' --motu http://my.cmems-du.eu/motu-web/Motu'+ \
                ' --service-id SEALEVEL_GLO_PHY_CLIMATE_L4_REP_OBSERVATIONS_008_057-TDS'+ \
                ' --product-id dataset-duacs-rep-global-merged-twosat-phy-l4'+ \
                ' --longitude-min '+str(domain[0])+' --longitude-max '+str(domain[1])+ \
                ' --latitude-min '+str(domain[2])+' --latitude-max '+str(domain[3])+ \
                ' --date-min "'+str(startDate[le].strftime('%Y-%m-%d %H:%M:%S'))+'" --date-max "'+str(endDate[le].strftime('%Y-%m-%d %H:%M:%S'))+'"'+ \
                ' --depth-min '+str(depths[0])+' --depth-max '+str(depths[1])+ \
                varStr+ \
                ' --out-dir '+path2saveData+' --out-name '+fname +'-'+ str(le) + '.nc'

        # run the runcommand, i.e. download the data specified above
        print('fetching latest mercator ocean forecast from CMEMS and making datastack')
        
        print (runcommand)
        
        os.system(runcommand)

        print('Time exceeded:   ',datetime.now() - startTime)   