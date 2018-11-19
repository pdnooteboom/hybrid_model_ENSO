# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 11:34:18 2017

@author: Peter Nooteboom
"""
"""Login to plotly in order to make plot!"""


import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import datetime 

def yeararray_to_datearray(array):
    year = array.astype(int)
    month = ((array - array.astype(int))*12 + 1)

    result = np.zeros(len(array)).astype(datetime.datetime)
    for i in range(len(array)):
        result[i] = datetime.datetime(year=year[i],month=int(round(month[i])),day=1)
    return result
    
def timearray(base,length):
    result = []
    for i in range(length):
        year = base.strftime('%Y') + i / 12
        month = (int(base.strftime('%m')) - 1 + i)%12   + 1
        result.append(datetime.datetime(year,month,1))
        
    return result

act = np.loadtxt('nino_last11months.txt')[:,9]
timeact = np.linspace(2016+5/12.,2017+4/12.,len(act))

monlow = 1
monhigh = 7
monlen= monhigh-monlow
res = []
timerestot = np.array([2016+6/12.,2016+7/12.,2016+8/12.,2016+9/12.,2016+10/12.,2016+11/12.,2017,2017+1/12.,2017+2/12.,2017+3/12.,2017+4/12.,2017+5/12.,2017+6/12.,2017+7/12.,2017+8/12.,2017+9/12.,2017+10/12.])
#timearray(datetime.datetime(2016,7,1),17)#
timeres = []
scat = [act[-1]]   #Contains all last points of the lead time predictions
scattime = [timeact[-1]]

for i in range(monlen):
    res.append(np.load('pred_mon{}.npy'.format(i+monlow)))
        
lent = res[0].shape[0]
timeres = np.zeros((monlen,lent))
for i in range(monlen):
    timeres[i] = (np.arange(0,17) - 16)/12. + i/12. +2017+5/12.
     
res12 = np.load('pred_mon{}.npy'.format(12))
timeres12 = (np.arange(0,28) - 27)/12. + 12/12. +2017+5/12.#np.linspace(2017+5/12.,2018+5/12.,13) - 1/12. 

tracepredl = []
for i in range(monlen):
    if(i!=5):
        scat.append(res[i][-1])
        scattime.append(2017+5/12.+i/12.)
        tracepredl.append(go.Scatter(                                 
                                    y = res[i] ,
                                    x =  yeararray_to_datearray(timeres[i]),
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
            x =  yeararray_to_datearray(timeres12),
            name = '12 month lead' ,
            line = dict(
#                        color = ('rgb(0,0,255)'),
                        width = 3,
                        dash = 'dash',                        
                        ),
            showlegend = True,
            #'showscale': False
        )
maxi  = np.load('upper_bound.npy').astype(np.float) 
mini = np.load('lower_bound.npy').astype(np.float)
nceppred = np.load('NCEPpred.npy').astype(np.float)
timencep = (np.arange(1,10) - 1)/12. +2017 + 5/12.  
timencep =     np.append(timeact[-1],timencep)
maxi = np.append(act[-1],maxi)
mini = np.append(act[-1],mini)
tracencep = go.Scatter(
            y = np.append(act[-1],nceppred) ,
            x =  yeararray_to_datearray(timencep),
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
trace_ncep_fill = go.Scatter(showlegend=False,
                            x = yeararray_to_datearray(np.append(timencep ,timencep[::-1])),
                            y = np.append(maxi,mini[::-1]),
                            fill='tozerox',
                            fillcolor='rgba(255,0,0,0.3)',
                            line=go.Line(color='transparent'),
                            name='ncep',
                            mode = 'lines'
                            
                            )            
            
tracescat = go.Scatter(
            y = scat,
            x =  yeararray_to_datearray(scattime),
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
            x =  yeararray_to_datearray(np.append(timeact[-1],scattime)),
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
            y = act,#np.append(act,nceppred[0]) ,
            x =  yeararray_to_datearray(timeact),#np.append(timeact,timencep[0])),
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
            x =  yeararray_to_datearray(timeact),
            name = 'Observation' ,
            line = dict(
                        color = ('rgb(0,0,0)'),
                        width = 3
                        ),
            showlegend = False,
            #'showscale': False
        )            

fig = tls.make_subplots(rows=1, cols=1, shared_yaxes=True, shared_xaxes=False)

#for i in range(len(tracepredl)):
#    fig.append_trace(tracepredl[i], 1, 1)
fig.append_trace(trace12pred, 1, 1)

#fig.append_trace(tracescat, 1, 1)
fig.append_trace(tracencep, 1, 1)
fig.append_trace(traceprede, 1, 1)
fig.append_trace(traceact, 1, 1)
fig.append_trace(trace_ncep_fill, 1, 1)

#fig.append_trace(traceact2, 1, 1)


def to_unix_time(dt):
    epoch =  datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000


fig['layout']['xaxis1'].update(range=[to_unix_time(yeararray_to_datearray(timeact)[0]),to_unix_time(yeararray_to_datearray(timeres12)[-1])],
                                tickfont=dict(
                                              size=30
                                              ),
                                tickangle=45,
     #                           dtick = 0.25,
                                title = 'Time (Year)',
                                titlefont=dict(
                                        size=45
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
                                size=30
                                ),
                                dtick=0.5,
                                title='NINO3.4 (°C)',
                                titlefont=dict(
                                        size=35
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
                        height=600,    
                        legend=dict(font=dict(size=30),
                                    x=0.97,
                                    y=1.4,
                                    borderwidth=2,
                                    orientation = 'h'
                                    ),
                        )

#plot_url = py.plot(fig, filename='Prediction_from_may')
py.image.save_as(fig, filename='Prediction_from_may.png')