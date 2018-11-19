# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 08:52:46 2017

@author: Peter Nooteboom
"""

import matplotlib.pylab as plt
import numpy as np
import el_nino_weka as weka
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

# function that creates latex-table
def latex_table(celldata,rowlabel,collabel):
    table = r'\begin{table} \begin{tabular}{|1|'
    for c in range(0,len(collabel)):
        # add additional columns
        table += r'1|'
    table += r'} \hline'

    # provide the column headers
    for c in range(0,len(collabel)-1):
        table += collabel[c]
        table += r'&'
    table += collabel[-1]
    table += r'\\ \hline'

    # populate the table:
    # this assumes the format to be celldata[index of rows][index of columns]
    for r in range(0,len(rowlabel)):
        table += rowlabel[r]
        table += r'&'
        for c in range(0,len(collabel)-2):
            if not isinstance(celldata[r][c], basestring):
                table += str(celldata[r][c])
            else:
                table += celldata[r][c]
            table += r'&'

        if not isinstance(celldata[r][-1], basestring):
            table += str(celldata[r][-1])
        else:
            table += celldata[r][-1]
        table += r'\\ \hline'

    table += r'\end{tabular} \end{table}'

    return table

trace70_3 = []
trace80_3 = []
trace75_3 = []
trace85_3 = []
trace70_6 = []
trace80_6 = []
trace75_6 = []
trace85_6 = []
trace70_12 = []
trace80_12 = []
trace75_12 = []
trace85_12 = []
 
root2 = 'C:/Users/User/Documents/Thesis/Hybrid_prediction_model/ClimateLearn/test_IMAU/ens10/ARIMA12-1-0/ens444/'
    

ens3 = np.load(root2 + 'rm_mon4_PC2_wwv_sc_elninoreal/ens10_keys_date_time_ElNino_PC2_seasonal_cycle_wwv_ElNino_realARIMA.npy')
ens6 = np.load(root2 + 'rm_mon6_PC2_wwv_sc_elninoreal/ens10_keys_date_time_ElNino_PC2_seasonal_cycle_wwv_ElNino_realARIMA.npy')
ens12 = np.load(root2 + 'rm_mon12_PC2_c2_sc_elninoreal/ens10_keys_date_time_ElNino_c2_PC2_seasonal_cycle_ElNino_realARIMA.npy')
ens11 = np.load(root2 + 'rm_mon11/norm/ens10_keys_date_time_ElNino_c2_PC2_seasonal_cycle_ElNino_realARIMA.npy')


best_res3 = ens3[4]['prediction']
best_s3 = ens3[4]['s']
rmse3 = ens3[4]['NRMSE']
actual3 =  ens3[4]['actual']
time3 = ens3[4]['time']

best_res6 = ens6[4]['prediction']
best_s6 = ens6[4]['s']
rmse6 = ens6[4]['NRMSE']
actual6 =  ens6[4]['actual']
time6 = ens6[4]['time']

best_res11 = ens11[1]['prediction']
best_s11 = ens11[1]['s']
rmse11 = ens11[1]['NRMSE']
actual11 =  ens11[1]['actual']
time11 = ens11[1]['time']

best_res12 = ens12[6]['prediction']
best_s12 = ens12[6]['s']
rmse12 = ens12[6]['NRMSE']
actual12 =  ens12[6]['actual']
time12 = ens12[6]['time']


for i in range(10):
    plt.plot(ens3[i]['time'],ens3[i]['prediction'])
    plt.plot( ens3[i]['time'],ens3[i]['actual'])
    plt.title('ens 3 with size ' + str(ens3[i]['s']) +' and NRMSE '+ str(ens3[i]['NRMSE']))
    plt.show()    
for i in range(10):
    plt.plot(ens6[i]['time'],ens6[i]['prediction'])
    plt.plot( ens6[i]['time'],ens6[i]['actual'])
    plt.title('ens 6 with size ' + str(ens6[i]['s']) +' and NRMSE '+ str(ens6[i]['NRMSE']))
    plt.show()    
for i in range(10):
    plt.plot(ens11[i]['time'],ens11[i]['prediction'])
    plt.plot( ens11[i]['time'],ens11[i]['actual'])    
    plt.title('ens 11 with size ' + str(ens11[i]['s']) +' and NRMSE '+ str(ens11[i]['NRMSE']))
    plt.show()    
for i in range(10):
    plt.plot(ens12[i]['time'],ens12[i]['prediction'])
    plt.plot( ens12[i]['time'],ens12[i]['actual'])    
    plt.title('ens 12 with size ' + str(ens12[i]['s']) +' and NRMSE '+ str(ens12[i]['NRMSE']))
    plt.show()


ncep_monl3 = np.load('C:/Users/User/Documents/Thesis/Hybrid_prediction_model/ClimateLearn/plotML/leadtimedata/2006-2012new/3monl_2006to2015_NCEP.npy').astype(np.float)[:109]
ncep_timel3 = np.load('C:/Users/User/Documents/Thesis/Hybrid_prediction_model/ClimateLearn/plotML/leadtimedata/2006-2012new/time3_2006to2015_NCEP.npy').astype(np.float)[:109]
ncep_monl6 =np.load('C:/Users/User/Documents/Thesis/Hybrid_prediction_model/ClimateLearn/plotML/leadtimedata/2006-2012new/6monl_2006to2015_NCEP.npy').astype(np.float)[:109]
ncep_timel6 = np.load('C:/Users/User/Documents/Thesis/Hybrid_prediction_model/ClimateLearn/plotML/leadtimedata/2006-2012new/time6_2006to2015_NCEP.npy').astype(np.float)[:109]

ncep_obs = np.load('C:/Users/User/Documents/Thesis/Hybrid_prediction_model/ClimateLearn/plotML/leadtimedata/2006-2012new/obs_2006to2015_NCEP.npy').astype(np.float)[:109]
ncep_obs_time = np.load('C:/Users/User/Documents/Thesis/Hybrid_prediction_model/ClimateLearn/plotML/leadtimedata/2006-2012new/timeobs_2006to2015_NCEP.npy').astype(np.float)[:109]

               
#timen_ncep = ncep_time[(ncep_time[i]>=2005) & (ncep_time[i]<=maxtime)]
#ncep_monl3n = ncep_monl3[(ncep_time[i]>=2005) & (ncep_time[i]<=maxtime)]
                          
rmse_ncep_3monl = weka._NorRMSE(ncep_monl3[15:],ncep_obs[15:]) 

cor3 = 0.32

mintimecor3 = 2004 + cor3
maxtimecor3 = 2014 + cor3   
mintime3 = 2004 - cor3
maxtime3 = 2014                  
                          
for i in range(1):
    bestresa = np.array(best_res3)[(time3>=mintime3) & (time3<=maxtime3)]
    timen = time3[(time3>=mintime3) & (time3<=maxtime3)] + cor3

    rmse_ncep_3monl = weka._NorRMSE(ncep_monl3[(ncep_timel3>=timen[0]) & (ncep_timel3<=timen[timen.shape[0]-1])],ncep_obs[(ncep_timel3>=timen[0]) & (ncep_timel3<=timen[timen.shape[0]-1])])                 

    plt.plot(timen,actual3[(time3>=mintime3) & (time3<=maxtime3)],'r--',label = "actual")
    plt.plot(ncep_timel3,ncep_monl3,'k',label='cfs')
    plt.plot(timen,bestresa,'b',label = "predicted")
    plt.plot(ncep_obs_time,ncep_obs,'k--',label='cfs')    
    plt.xlim(max(timen[0],ncep_timel3[0]),min(timen[len(timen)-1],ncep_timel3[len(ncep_timel3)-1]))
    plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
    plt.xlabel("time (years)")
    plt.ylabel("NINO3.4")
    plt.ylim(-2.5,2.5)  
    plt.title("" + str(3) + " mon (RMSE = " + str(weka._NorRMSE(actual3[(time3>=mintime3) & (time3<=maxtime3)],np.array(best_res3)[(time3>=mintime3) & (time3<=maxtime3)])) +", RMSEncep:" + str(rmse_ncep_3monl) + " )")    
    #plt.title(" ENSO predition " + str(monlow+i) + " months ahead (RMSE = " + str(weka._NorRMSE(actual[i][(time[i]>=2005) & (time[i]<=maxtime)],np.array(best_res[i])[(time[i]>=2005) & (time[i]<=maxtime)])) + " )")
    plt.show()

#And the 6 month lead prediction    
    
cor6 = 0.54
mintimecor6 = 2005 + cor6
maxtimecor6 = 2014 + cor6   
mintime6 = 2005
maxtime6 = 2014 

                          
for i in range(1):
    bestresa = np.array(best_res6)[(time6>=mintime6) & (time6<=maxtime6)]
    timen = time6[(time6>=mintime6) & (time6<=maxtime6)] + cor6

    rmse_ncep_6monl = weka._NorRMSE(ncep_monl6[15:],ncep_obs[15:])                 


    plt.plot(timen,actual6[(time6>=mintime6) & (time6<=maxtime6)],'r--',label = "actual")
    plt.plot(ncep_timel6,ncep_monl6,'k',label='cfs')
    plt.plot(timen,bestresa,'b',label = "predicted")
    plt.plot(ncep_obs_time,ncep_obs,'k--',label='cfs')    
    plt.xlim(max(timen[0],ncep_timel6[0]),min(timen[len(timen)-1],ncep_timel6[len(ncep_timel6)-1]))
    plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
    plt.xlabel("time (years)")
    plt.ylabel("NINO3.4")
    plt.ylim(-2.5,2.5)  
    plt.title("" + str(6) + " mon (RMSE = " + str(weka._NorRMSE(actual6[(time6>=mintime6) & (time6<=maxtime6)],np.array(best_res6)[(time6>=mintime6) & (time6<=maxtime6)])) +", RMSEncep:" + str(rmse_ncep_6monl) + " )")    
    #plt.title(" ENSO predition " + str(monlow+i) + " months ahead (RMSE = " + str(weka._NorRMSE(actual[i][(time[i]>=2005) & (time[i]<=maxtime)],np.array(best_res[i])[(time[i]>=2005) & (time[i]<=maxtime)])) + " )")
    plt.show()

    
#And the 12 month lead prediction    
for i in range(1):
    plt.plot(time12,actual12,'r--',label = "actual")
    plt.plot(time12,best_res12,'b',label = "predicted")
    #plt.xlim(time[i][0],t[len(t)-1])
    plt.legend(bbox_to_anchor=(0.0,0.0),loc = 1)
    #plt.xlim(2004, 2015)
    #plt.ylim(-3.3, 2.0)
    plt.xlabel("time (years)")
    plt.ylabel("NINO3.4")
    plt.title(" ENSO predition " + str(12) + " months ahead (RMSE = " + str(weka._NorRMSE(best_res12,actual12)) + " )")
    #fig_title = root_Dir_write + "ANN " + str(mon) + " mon ahead" + ".pdf"
    plt.show()
#%% And plotting the lead times in plotly
"""Sign into plotly:"""
py.sign_in()
cor3 = 0.32

cor6 = 0.54
  
mintimecor3 = 2005 + cor3
maxtimecor3 = 2014 + cor3
mintimecor6 = 2005 + cor6
maxtimecor6 = 2014 + cor6
mintime = 2005
maxtime = 2014

for i in range(1):
    bestresa3 = np.array(best_res3)[(time3>=mintime) & (time3<=maxtime)]
    timen3 = time3[(time3>=mintime) & (time3<=maxtime)] + cor3
    act3 = actual3[(time3>=mintime) & (time3<=maxtime)]

for i in range(1):
    bestresa6 = np.array(best_res6)[(time6>=mintime) & (time6<=maxtime)]
    timen6 = time6[(time6>=mintime) & (time6<=maxtime)] + cor6
    act6 = actual6[(time6>=mintime) & (time6<=maxtime)]

for i in range(1):
    bestresa12 = np.array(best_res12)[(time12>=mintime) & (time12<=maxtime)]
    timen12 = time12[(time12>=mintime) & (time12<=maxtime)] 
    act12 = actual12[(time12>=mintime) & (time12<=maxtime)]

                     
#import numpy as np
#import matplotlib.pyplot as plt
#from math import pi
#from matplotlib import rc

#rc('text', usetex=True)

# set up your data:

#celldata = [  [r'$4\times3\times1$', r'$4\times2\times2$', r'3\times1\times4'],
#              [ r'0.17/ 0.16', r'0.21/ 0.18', r'-/0.17'],
#              [ r'$PC_2,\ WWV,\ seasonal\ cycle,\$', r'$PC_2,\ WWV,\ seasonal\ cycle,\$', r'$PC_2,\ c_2,\ seasonal\ cycle,\$']]
#
#rowlabel = [r'structure', r'NRMSE (CFSv2/hybrid)', r'Attributes']
#collabel = [r'lead time',r'3/4 months', r'6 months', r'12 months']

#celldata = [[32, r'$\alpha$', 123],[200, 321, 50]]
#rowlabel = [r'1st row', r'2nd row']
#collabel = [r' ', r'$\alpha$', r'$\beta$', r'$\gamma$']
#print 'joe'
#table = latex_table(celldata,rowlabel,collabel)
#
## set up the figure and subplots
#f = plt.figure()
#
#ax1 = f.add_subplot(121)
#ax2 = f.add_subplot(122)
##ax3 = f.add_subplot(123)
##ax4 = f.add_subplot(124)
##f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharey='row')
#
#ax1.plot(timen3,bestresa3)
#ax1.plot(ncep_timel3[15:],ncep_monl3[15:])
#ax1.plot(timen3,act3)
#ax1.set_title('3/4 month lead time')
##ax2.plot(timen6,bestresa6)
##ax2.set_title('6 month lead time')
##ax2.plot(ncep_timel6[15:],ncep_monl6[15:])
##ax3.plot(timen12,act12)
##ax3.plot(timen12,bestresa12)
##ax3.set_title('12 month lead time')
##ax3.plot(timen12,act12)
#ax2.text(.1,.5,table, size=50)
#ax2.axis('off')
#
#plt.show()





#%%
#fig = plt.figure()
#
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
#ax3 = fig.add_subplot(123)
#ax4 = fig.add_subplot(124)
#
#ax1.plot(np.arange(100))
#ax2.text(.1,.5,table, size=50)
#ax2.axis('off')
#
#plt.show()                      
                     
                     
                     
                     
trace3pred = go.Scatter(
            y = bestresa3 ,
            x =  timen3,
            name = 'Hybrid model' ,
            line = dict(
                        color = ('rgb(0,0,255)'),
                        width = 3
                        ),
            showlegend = True,
        xaxis = 'x2',
        yaxis = 'y2',
            #'showscale': False
        )
trace_ncep3 = go.Scatter(
        x=ncep_timel3[15:],
        y=ncep_monl3[15:],
        name='CFSv2',
        line = dict(
                width = 3,
                color = ('rgb(255,12,24)')),
        showlegend = True,
        xaxis = 'x2',
        yaxis = 'y2',
    )        
trace_ncep6 = go.Scatter(
        x=ncep_timel6[15:],
        y=ncep_monl6[15:],
        name='CFSv2',
        line = dict(
                width = 3,
                color = ('rgb(255,12,24)')),
        showlegend = False,
        xaxis = 'x3',
        yaxis = 'y3',
    )     
            
trace6pred = go.Scatter(
            y = bestresa6,
            x =  timen6, 
            name = 'Prediction' ,
            line = dict(
                        color = ('rgb(0,0,255)'),
                        width = 3
                        ),
            showlegend = False,
        xaxis = 'x3',
        yaxis = 'y3',
            #'showscale': False
        )
trace12pred = go.Scatter(
            y = best_res12,
            x =  time12 + 1.04, 
            name = 'Prediction' ,
            line = dict(
                        color = ('rgb(0,0,255)'),
                        width = 3
                        ),
            showlegend = False,
        xaxis = 'x4',
        yaxis = 'y4',
            #'showscale': False
        )
            
trace3act = go.Scatter(
    x=timen3,
    y=act3,
    name = 'Observation',
    line = dict(
    width = 3,
    color = ('rgb(0,0,0)')),    
                showlegend = True,
        xaxis = 'x2',
        yaxis = 'y2',
)
            
trace6act = go.Scatter(
    x=timen6,
    y=act6,
    name = 'Observation',
    line = dict(
    width = 3,
    color = ('rgb(0,0,0)')), 
                showlegend=False,
        xaxis = 'x3',
        yaxis = 'y3',
)
   
trace12act = go.Scatter(
    x=time12 + 1.04,
    y=actual12,
    name = 'Observation',
    line = dict(
    width = 3,
    color = ('rgb(0,0,0)')), 
                showlegend=False,
        xaxis = 'x4',
        yaxis = 'y4',
)             




#fig = tls.make_subplots(rows=2, cols=2, shared_yaxes=True,
#                        shared_xaxes=False, subplot_titles=('$4 \\text{ month lead}$','$6 \\text{ month lead}$','$12 \\text{ month lead}$'))

# Add table data #(CFSv2/hybrid) #\ \ \ \ \ 
table_data = [['Lead time', '3/4 months', '6 months', '12 months'],
              ['structure', '$4\\times3\\times1$', '$4\\times2\\times2$', '$3\\times1\\times4$'],
              ['NRMSE', '$0.17/ 0.16$', '$0.21/ 0.18$', '$-/ 0.17$'],
              ['Attributes', '$PC_2,\ WWV,$', '$PC_2,\ WWV,$', '$PC_2,\ c_2,$'],
              ['', '$SC$', '$SC$', '$SC$']]


x1 = timen3
y1 = act3
fig = ff.create_table(table_data,height_constant=20)

fig['data'].extend(go.Data([trace3pred,trace_ncep3,trace3act]))
fig['data'].extend(go.Data([trace6pred,trace_ncep6,trace6act]))
fig['data'].extend(go.Data([trace12pred,trace12act]))


#figure.layout.xaxis.update({'domain': [0, .5]})
#figure.layout.xaxis2.update({'domain': [0.6, 1.]})
## The graph's yaxis MUST BE anchored to the graph's xaxis
#figure.layout.yaxis2.update({'anchor': 'x2'})
#figure.layout.yaxis2.update({'title': 'Goals'})
# Update the margins to add a title and see graph x-labels. 
fig.layout.margin.update({'t':50, 'b':100})
  
#fig.append_trace(trace3pred, 1, 1)
#fig.append_trace(trace_ncep3, 1, 1)
#fig.append_trace(trace3act, 1, 1) 
#fig.append_trace(trace6pred, 1, 2)
#fig.append_trace(trace_ncep6, 1, 2) 
#fig.append_trace(trace6act, 1, 2)  
#fig.append_trace(trace12pred, 2, 1) 
#fig.append_trace(trace12act, 2, 1) 

for i in range(len(fig.layout.annotations)):
    fig.layout.annotations[i].font.size = 32               
                        


fig['layout']['xaxis'].update(#range=[ncep_timel3[15],2014],
#                                tickfont=dict(
#                                              size=23
#                                              ),
#                                tickangle=45,
#                                dtick = 1,
                                domain=[0.47,1.],
#                                anchor = 'x1',
                        position = 0.,
                        )
fig['layout']['xaxis2'].update(range=[ncep_timel6[15],2014],
                                tickfont=dict(
                                              size=30
                                              ),
                                tickangle=45,
                                dtick = 1,
                                title='Time (year)',
                                titlefont=dict(
                                        size=40
                                        ),
                                domain=[0.05,0.45],
                                anchor = 'x2',  
                                position = 0.6,
                        )                                

                                
fig['layout']['xaxis3'].update(range=[time12[0]+1,time12[-1]+1],
                                tickfont=dict(
                                              size=30
                                              ),
                                tickangle=45,
                                dtick = 1,
                                title='Time (year)',
                                titlefont=dict(
                                        size=40
                                        ),
                                domain=[0.55,1.05],
                                anchor = 'x3',
                                position = 0.6,
                             
                        )   
fig['layout']['xaxis4'].update(range=[time12[0]+1,time12[-1]+1],
                                tickfont=dict(
                                              size=30
                                              ),
                                tickangle=45,
                                title='Time (year)',
                                titlefont=dict(
                                        size=40
                                        ),
                                dtick = 1,
                                domain=[0.05,0.45],
                                anchor = 'x4',
                                position = 0.,
                             
                        )                                  
                                 
                                
fig['layout'].update(#showlegend=True,

                        margin=go.Margin(
                                         l=50,
                                         r=5,
                                         b=200,
                                         t=50,
                                         pad=4
                                         ),
                        width=1600,
                        height=1200,    
                        legend=dict(font=dict(size=40),
                                    x=0.47,
                                    y=0.43,
                                    borderwidth=3,
                                    orientation = 'h'
                                    ),
                        )

fig['layout']['yaxis'].update(#range=[-2.6,2.2], 
#                                tickfont=dict(
#                                              size=23
#                                              ),
#                                dtick=1,
#                                title = 'NINO3.4 (°C)',
#                                titlefont=dict(
#                                        size=30
#                                        ),
                                               domain=[-0.5,0.2],
                                    position = 0.15,
#                                anchor = 'y1',
)
                                
fig['layout']['yaxis2'].update(range=[-2.6,2.2], 
                                tickfont=dict(
                                              size=30
                                              ),
                                dtick=1,
                                title = 'NINO3.4 (°C)',
                                titlefont=dict(
                                        size=40
                                        ),
                                               domain=[0.65,1],
                                position = 0.05,
                                anchor = 'y2',)
fig['layout']['yaxis3'].update(range=[-2.6,2.2], 
                                tickfont=dict(
                                              size=30
                                              ),
                                dtick=1,
                                title = 'NINO3.4 (°C)',
                                titlefont=dict(
                                        size=40
                                        ),
                                               position = 0.55,
                                               domain=[0.65,1.],
                                anchor = 'y3',)
fig['layout']['yaxis4'].update(range=[-2.6,2.2], 
                                tickfont=dict(
                                              size=30
                                              ),
                                dtick=1,
                                title = 'NINO3.4 (°C)',
                                titlefont=dict(
                                        size=40
                                        ),
                                               position = 0.05,
                                               domain=[0.,0.4],
                                anchor = 'y4',)
                                
fig['layout']['annotations'].append({ 'font': {'size': 40},
                                     'showarrow': False,
                                     'text':'(a)',
                                     'x': 0.2,
                                     'xanchor': 'center',
                                     'xref': 'paper',
                                     'y': 1.0,
                                     'yanchor': 'bottom',
                                     'yref': 'paper'})
fig['layout']['annotations'].append({ 'font': {'size': 40},
                                     'showarrow': False,
                                     'text':'(b)',
                                     'x': 0.7,
                                     'xanchor': 'center',
                                     'xref': 'paper',
                                     'y': 1.0,
                                     'yanchor': 'bottom',
                                     'yref': 'paper'})
fig['layout']['annotations'].append({ 'font': {'size': 40},
                                     'showarrow': False,
                                     'text':'(c)',
                                     'x': 0.2,
                                     'xanchor': 'center',
                                     'xref': 'paper',
                                     'y': 0.4,
                                     'yanchor': 'bottom',
                                     'yref': 'paper'})
fig['layout']['annotations'].append({ 'font': {'size': 40},
                                     'showarrow': False,
                                     'text':'(d)',
                                     'x': 0.7,
                                     'xanchor': 'center',
                                     'xref': 'paper',
                                     'y': 0.25,
                                     'yanchor': 'bottom',
                                     'yref': 'paper'})

fig['data'][0].update(colorscale=[[0, '#000000'], [0.5, '#ffffff'], [1, '#ffffff']])

#py.iplot(fig)
#plot_url = py.plot(fig, filename='predictionsbest_3mon_6mon_12monleadtime_incltab')
py.image.save_as(fig, filename='predictionsbest_3mon_6mon_12monleadtime_incltab.png')

print '3 month lead predition size: ',best_s3 ,' rmse: ', rmse3
print '6 month lead predition size: ',best_s6 ,'  rmse: ', rmse6
print '11 month lead predition size: ',best_s11 ,'  rmse: ', rmse11
print '12 month lead predition size: ',best_s12 ,'  rmse: ', rmse12