
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

def monthly_to_weekly(array):
    #This function returns from an array with monthly data, an array 
    #that contains the value of that month for the week you are in
    res = np.zeros(len(array)*52/12)
    for i in range(len(res)):
        weekofyear = i%52
        year = i/52
        monthofyear = weekofyear*12/52
        res[i] = array[year*12+monthofyear]
    return res

def highestcorlag(var,nino):
    if(len(var)!=len(nino)):
        print 'length not the same'
    maxi = 0.
    maxlag = 0
    for lag in range(200):#int(104/sq)):
        if(lag==0):
            cor = np.abs(np.corrcoef(var,nino)[0,1])
        else:
            cor = np.abs(np.corrcoef(var[:-lag],nino[lag:])[0,1])
        if(cor>maxi):
            maxi = cor
            maxlag = lag
    return maxlag,maxi


#lengths of both the h ans SST timeseries:
sdata1 =  1827
lenseq = 52 #the length of every timeseries in months
sq = 4. #amount of months shifted 
nslides = 1

spart = int((sdata1-lenseq)/nslides)# #size of one part of the timeserie
lensliding = int(spart/sq)#int(sdata1/sq-lenseq)#3 #amount of times the window is shifted one month ahead.

time = np.linspace(1979+lenseq/52.,1979+(lensliding*sq+lenseq)/52.,lensliding*nslides)


SST = False

if(SST):
    c2 = np.load('c2_SST_mu325_th0.999.pny.npy')
else:
    #c2 = np.load('c2_h_mu27_th0.999.pny.npy')
    c2 = np.load('c2_h_th0.9.pny.npy')
#    c2 = np.load('c2_mu27_th0.9999.pny.npy')
#    c2 = np.load('c2_mu27_th0.99992.pny.npy')
#    c2 = np.load('c2_mu27_th0.9999995.pny.npy')
#    c2 = np.load('c2_h_mu27_th0.999.pny.npy')    
  
nino34 = np.loadtxt('monthlynino34.txt')
nino34 = np.delete(nino34,0,1)
nino34 = np.reshape(nino34,nino34.size)

nino = monthly_to_weekly(nino34)
nino2 = nino[lenseq::4]
nino2 = nino2[:lensliding+1]
c2 = c2[:len(nino2)]

warmwatervolume = np.loadtxt('warmwatervolume.txt')
wwv = warmwatervolume[:,2]/np.float(10**14)

timewwv = np.linspace(1980,2014,len(warmwatervolume[:,2])+1)
timewwv = timewwv[:-1]
#wwv = np.zeros(lensliding)
#for i in range(lensliding):
#    wwv[i] = warmwatervolume[(int(i*sq+lenseq)/52-2)*12 + int((int(sq*i+lenseq-5)%52)/52.*12),2]    

#nino34 = nino[lenseq::int(sq)]#np.zeros(int((sdata1-lenseq)/sq))
#nino34 = nino34[:int((sdata1-lenseq)/sq)]
#time = np.linspace(5+lenseq/52.,5+sdata1/52.,int((sdata1-lenseq)/sq))

#Granger causality:
    
#import statsmodels.tsa.stattools as st
#
#x = np.zeros((2,len(c2)))
#x[0] = nino2
#x[1] = c2
#x = np.swapaxes(x,0,1)
#
#res = st.grangercausalitytests(x, 14)

#x = np.zeros((2,len(wwv)))
#x[0] = nino2
#x[:,1] = wwv/np.float(10**14)
#x = np.swapaxes(x,0,1)

#res2 = st.grangercausalitytests(x, 7)

#%% The subplot plot


time82 = time[(time>1995) & (time<2000)]
nino282 = nino2[(time>1995) & (time<2000)]
c282 = c2[(time>1995) & (time<2000)]
timewwv82 = timewwv[(timewwv>1995) & (timewwv<2000)]
wwv82 = wwv[(timewwv>1995) & (timewwv<2000)]          
    
    
"""Sign into plotly"""

trace1 = go.Scatter(
    x=time,
    y=c2,
    showlegend=False,
    line = dict(
        width = 3,
        color = ('rgb(255,12,24)')),
    name='$c_2$'
)
trace2 = go.Scatter(
    x=time,
    y=nino2,
    showlegend=False,
    name='NINO3.4',
    yaxis='y2',
    #fill='tozeroy'
    line = dict(
        color = 'rgb(0,0,0)',
        width = 3),
)
trace3 = go.Scatter(
    x=timewwv,
    y=wwv,
    showlegend=False,
    name='WWV',
    yaxis='y3',
    #fill='tozeroy'
    line = dict(
        color = 'rgb(0,0,255)',
        width = 3),
)
trace4 = go.Scatter(
    x=time82,
    y=c282,
    line = dict(
        width = 3,
        color = ('rgb(255,12,24)')),
    name='$\quad \ c_2$ ',
    yaxis = 'y1'
)
trace5 = go.Scatter(
    x=time82,
    y=nino282,
    name='$\\ \\ \\ NINO3.4$',
    yaxis='y2',
    #fill='tozeroy'
    line = dict(
        color = 'rgb(0,0,0)',
        width = 3),
)
trace6 = go.Scatter(
    x=timewwv82,
    y= wwv82,
    name='$\quad \ WWV$',
    yaxis='y3',
    #fill='tozeroy'
    line = dict(
        color = 'rgb(0,0,255)',
        width = 3),
)

fig = tls.make_subplots(rows=1, cols=2, shared_yaxes=True, shared_xaxes=False, subplot_titles=('(a)',
                                                        '(b)'))
#, shared_xaxes=True
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 1)
fig.append_trace(trace4, 1, 2)
fig.append_trace(trace5, 1, 2)
fig.append_trace(trace6, 1, 2)

fig['layout']['xaxis1'].update(title='Time (year)',
                                    dtick=5,
                                    tickangle=45,
                                    tickfont=dict(
                                    #color='#d62728',
                                    size=35),
                                    titlefont=dict(
                                        size=45,
                                    ),
                                        position = 0.15,
                                    range=[1980,2014],domain=[0.05,0.6])
fig['layout']['xaxis2'].update(title='Time (year)',
                                    dtick=0.5,
                                    tickangle=45,
                                    tickfont=dict(
                                    #color='#d62728'
                                    size=35),
                                    titlefont=dict(
                                        size=45,
                                    ),
                                    position = 0.15,
                                    range=[1995,2000],domain=[0.68,0.94])

fig['layout'].update(showlegend=True,
                        legend=dict(font=dict(size=40),
                                    x=0.25,
                                    y=1.,
                                    borderwidth=3,
                                    orientation = 'h'
                                    ),
                        
                        )#,title='ENSO in black, zonal skewness in red')

fig['layout']['yaxis1'].update(
                                title='$c_2$',
                                zeroline=False,
                                showgrid=False,
                                titlefont=dict(
                                    size=45
                                    ),
                                tickfont=dict(
                                    #color='#d62728'
                                    size=30
                                    ),range=[0.019,0.13],domain=[0.15,0.9])#,title='Jan')#title='NINO34',
fig['layout']['yaxis2'].update(        
                                title='NINO3.4 (°C)',
                                zeroline=False,
                                titlefont=dict(
                                    size=45
                                ),
                                tickfont=dict(
                                    #color='#d62728'
                                    size=30
                                    ), 
                                dtick = 1,                                              
                                position=0.6,
                                #showgrid = False,
                                overlaying='y',
                                side='right',
                                range=[-2.5,3],
                                domain=[0.15,0.9])#
fig['layout']['yaxis3'].update(        
                                title='$WWV\ (10^{14}\ m^3)$',
                                zeroline=False,
                                titlefont=dict(
                                    size=45
                                ),
                                position=0.95,
                                tickfont=dict(
                                    #color='#d62728'
                                    size=30
                                ),
                                showgrid = False,
                                overlaying='y',
                                side='right',
                                range=[-4,3],domain=[0.15,0.9])#
fig['layout'].update(height=800,width=1500,margin=go.Margin(t=45,b=80,l=30,r=10))
fig['layout']['titlefont'].update({'size':45}) 
                                             
fig['data'][0]['yaxis'] = 'y1'#.update(yaxis='y1')
fig['data'][1]['yaxis'] = 'y2'#.update(yaxis='y2')
fig['data'][2]['yaxis'] = 'y3'#.update(yaxis='y2')
fig['data'][3]['yaxis'] = 'y1'#.update(yaxis='y1')
fig['data'][4]['yaxis'] = 'y2'#.update(yaxis='y2') 
fig['data'][5]['yaxis'] = 'y3'#.update(yaxis='y2')

fig['layout']['annotations'][0]['font']['size'] = 45
fig['layout']['annotations'][1]['font']['size'] = 45
                                              
#plot_url = py.plot(fig, filename='c2_h_obs')
py.image.save_as(fig, filename='c2_h_obs.png')
#%%
#only82 = True
#if(only82):
#    time = time[(time>1995) & (time<2000)]
#    nino2 = nino2[(time>1995) & (time<2000)]
#    c2 = c2[(time>1995) & (time<2000)]
#    
#    
##py.sign_in('peternooteboom','1pbc2kdd8l')#nootje01
#py.sign_in('Peppie','fZpfuzOdO8YiVa8mdHyN')#bootje01
##
#trace1 = go.Scatter(
#    x=time,
#    y=c2,
#    line = dict(
#        width = 3,
#        color = ('rgb(255,12,24)')),
#    name='$c_2$'
#)
#trace2 = go.Scatter(
#    x=time,
#    y=nino2,
#    name='NINO3.4',
#    yaxis='y2',
#    line = dict(
#        color = 'rgb(0,0,0)',
#        width = 4),
##    fill='tozeroy'
#)
##trace3 = go.Scatter(
##    x=time,
##    y=wwv,
##    name='WWV',
##    yaxis='y3',
##    #fill='tozeroy'
##)
#data = [trace1,trace2]
#layout = go.Layout(
#                   height=700,#700,
#                   width=500,#1200,
#    showlegend = False,            
##    title='',  
#    xaxis=dict(
#        title='Time (years)',
#        dtick=0.5,
#        titlefont=dict(
#            size=20
#        ),
#        tickfont=dict(
#            #color='#d62728'
#            size=18
#            ),                       
#    ),    
#    yaxis=dict(
#        title='$c_2$',
#        showgrid=False,
#        titlefont=dict(
#            size=20
#            ),
#        tickfont=dict(
#            #color='#d62728'
#            size=18
#            ),
#    ),
#    yaxis2=dict(
#        title='NINO3.4',
#        titlefont=dict(
#            size=20
#        ),
#        tickfont=dict(
#            #color='#d62728'
#            size=18
#        ),
#        overlaying='y',
#        side='right',
#        #range=[0,1.]
#    ),
##    yaxis3=dict(
##        title='WWV',
##        titlefont=dict(
##            size=20
##        ),
##        tickfont=dict(
##            #color='#d62728'
##            size=18
##        ),
##        overlaying='y',
##        side='right',
##        #range=[0,1.]
##    ),                      
#    legend=dict(
#        font=dict(
#            #family='sans-serif',
#            size=20,
#            #color='#000'
#        )
#    )
#)
#               
#fig = go.Figure(data=data, layout=layout)
#update = {'data':[{'fill': 'tonexty'}]}  # this updates BOTH traces now
#
##if(SST):
##    plot_url = py.plot(fig,update=update, filename='c2_h_th0.999_mu325')
##else:
##    plot_url = py.plot(fig,update=update, filename='c2_h_th09_obs')
#  
#plot_url = py.plot(fig,update=update, filename='c2_obs')
#  
##plot_url = py.plot(fig, update=update, filename='mpl-multi-fill')