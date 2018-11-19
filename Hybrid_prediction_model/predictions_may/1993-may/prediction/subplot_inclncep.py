# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 11:34:18 2017

@author: Peter Nooteboom
"""


import matplotlib.pylab as plt
import numpy as np
#import el_nino_weka as weka

#import el_nino_weka as weka

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

#act = np.load('actual.npy')
#timeact = np.array([2016+11/12.,2017,2017+1/12.,2017+2/12.,2017+3/12.,2017+4/12.,2017+5/12.,2017+6/12.,2017+7/12.,2017+8/12.,2017+9/12.])+1/12.
act = np.loadtxt('nino_last11months.txt')[:-1,9]
timeact = np.linspace(2016+7/12.,2017+3/12.,len(act))

monlow = 1
monhigh = 7
monlen= monhigh-monlow
res = []
timerestot = np.array([2016+6/12.,2016+7/12.,2016+8/12.,2016+9/12.,2016+10/12.,2016+11/12.,2017,2017+1/12.,2017+2/12.,2017+3/12.,2017+4/12.,2017+5/12.,2017+6/12.,2017+7/12.,2017+8/12.,2017+9/12.,2017+10/12.])
timeres = []
scat = [act[-1]]   #Contains all last points of the lead time predictions
scattime = [timeact[-1]]

for i in range(monlen):
    res.append(np.load('pred_mon{}.npy'.format(i+monlow)))
        
lent = res[0].shape[0]
timeres = np.zeros((monlen,lent))
for i in range(monlen):
    timeres[i] = (np.arange(0,17) - 16)/12. + i/12. +2017+4/12.
     
res12 = np.load('pred_mon{}.npy'.format(12))
timeres12 = (np.arange(0,13) - 14)/12. + 12/12. +2017+4/12.#np.linspace(2017+5/12.,2018+5/12.,13) - 1/12. 

#scat = np.array(scat)  
#scattime = np.array([2017+5/12.,2017+6/12.,2017+7/12.,2017+8/12.,2017+9/12.,2017+10/12.,2018+5/12.])      

"""Sign in with plotly account"""
py.sign_in()

tracepredl = []
for i in range(monlen):
    if(i==0 or i==2 or i==5):
        scat.append(res[i][-1])
        scattime.append(2017+4/12.+i/12.)
        tracepredl.append(go.Scatter(                                 
                                    y = res[i] ,
                                    x =  timeres[i],
                                    name = '{} month lead'.format(i+monlow) ,
                                    line = dict(
    #                                            color = ('rgb(0,0,255)'),
                                                width = 3,
                                                dash = 'dash',                                                    
                                                ),
                                    showlegend = True,
                                    #'showscale': False                                 
                                     
                                     )
                      
                      
                      )
scat.append(res12[-1])
scattime.append(timeres12[-1])  
scat = np.array(scat)
scattime = np.array(scattime)                                  
                                    
trace12pred = go.Scatter(
            y = res12 ,
            x =  timeres12,
            name = '12 month lead' ,
            line = dict(
#                        color = ('rgb(0,0,255)'),
                        width = 3,
                        dash = 'dash',                        
                        ),
            showlegend = True,
            #'showscale': False
        )
            
nceppred = np.load('NCEPpred.npy').astype(np.float)
timencep = (np.arange(1,10) - 1)/12. +2017 + 4/12.      
tracencep = go.Scatter(
            y = nceppred ,
            x =  timencep,
            name = 'CFSv2' ,
            line = dict(
                        color = ('rgb(255,0,0)'),
                        width = 3,
#                        dash = 'dash',
                        ),
            marker = dict(
                size = 14,
                color = 'rgb(255,0,0)',
                ),                        
            showlegend = True,
            #'showscale': False
        )
tracescat = go.Scatter(
            y = scat,
            x =  scattime,
            #name = '12 month lead' ,
            mode = 'markers',
            marker = dict(
                size = 14,
                color = 'rgb(0,0,255)',
#                line = dict(
#                    width = 2,
#                            )
                ),
            showlegend = False,  
                       )
traceprede = go.Scatter(
            y = np.append(act[-1],scat),
            x =  np.append(timeact[-1],scattime),
            #name = '12 month lead' ,
            name = 'Hybrid model' ,
            line = dict(
                        color = ('rgb(0,0,255)'),
                        width = 3,
                        
#                        dash = 'dash',
                        ),
            marker = dict(
                size = 14,
                color = 'rgb(0,0,255)',
#                line = dict(
#                    width = 2,
#                            )
                ),                        
            showlegend = True, 
                       )
traceact = go.Scatter(
            y = np.append(act,nceppred[0]) ,
            x =  np.append(timeact,timencep[0]),
            name = 'Observation' ,
            line = dict(
                        color = ('rgb(0,0,0)'),
                        width = 3
                        ),
            marker = dict(
                size = 14,
                color = 'rgb(0,0,0)',
                ),                         
            showlegend = True,
            #'showscale': False
        )
traceact2 = go.Scatter(
            y = act ,
            x =  timeact,
            name = 'Observation' ,
            line = dict(
                        color = ('rgb(0,0,0)'),
                        width = 3
                        ),
            showlegend = False,
            #'showscale': False
        )            

fig = tls.make_subplots(rows=1, cols=1, shared_yaxes=True, shared_xaxes=False)

for i in range(len(tracepredl)):
    fig.append_trace(tracepredl[i], 1, 1)
fig.append_trace(trace12pred, 1, 1)

#fig.append_trace(tracescat, 1, 1)
fig.append_trace(tracencep, 1, 1)
fig.append_trace(traceprede, 1, 1)
fig.append_trace(traceact, 1, 1)
#fig.append_trace(traceact2, 1, 1)

fig['layout']['xaxis1'].update(range=[timeact[0],timeres12[-1]],
                                tickfont=dict(
                                              size=23
                                              ),
                                tickangle=45,
                                dtick = 0.25,
                                title = 'Time (Year)',
                                titlefont=dict(
                                        size=30
                                        ),
                        )
                                
#fig['layout']['xaxis2'].update(range=[timeact[0],timeres12[-1]],
#                                tickfont=dict(
#                                              size=23
#                                              ),
#                                tickangle=45,
#                                dtick = 0.25,
#                                title = 'Time (Year)',
#                                titlefont=dict(
#                                        size=30
#                                        ),
#                        )
                                
fig['layout']['yaxis1'].update(range=[-2,2], tickfont=dict(
                                size=23
                                ),
                                dtick=0.5,
                                title='NINO3.4 (°C)',
                                titlefont=dict(
                                        size=30
                                        ),
                                    
                            )

fig['layout'].update(#showlegend=True,

                        margin=go.Margin(
                                         l=100,
                                         r=100,
                                         b=200,
                                         t=100,
                                         pad=4
                                         ),
                        width=900,
                        height=700,    
                        legend=dict(font=dict(size=25),
                                    x=0.97,
                                    y=1.3,
                                    borderwidth=3,
                                    orientation = 'h'
                                    ),
                        )

plot_url = py.plot(fig, filename='Prediction_from_april')
