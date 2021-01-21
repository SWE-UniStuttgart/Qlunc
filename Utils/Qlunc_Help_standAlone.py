# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:03:43 2020

@author: fcosta
"""

from Qlunc_ImportModules import *
import numpy as np

#import Qlunc_Help_standAlone as SA
#from Main.Qlunc_inputs import inputs

#%%# used to flatt at some points along the code:
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (list,tuple)) else (a,))) 

#%% Sum of dB:
def Sum_dB(dB_data):
   #Sum of decibels:
#    pdb.set_trace()
    Sumat= []
    Sum_in_decibels=[]
    for ii in dB_data:
        watt_data=((10**(ii/10)))    
        Sumat.append (watt_data)
    Sum_in_watt = sum(Sumat)
    Sum_in_decibels.append(10*np.log10(Sum_in_watt) )
#    pdb.set_trace()
    return list (flatten(Sum_in_decibels))

#%% Combine uncertainties:
def unc_comb(data): # data is provided as a list of elements want to add on. Data is expected to be in dB (within this functions dB are transformed into watts). The uncertainty combination is made following GUM
    data_watts  = []
    res_dB      = []
    res_watts   = []
    zipped_data = []
    if not isinstance (data,np.ndarray):
        data=np.array(data)    
    for data_row in range(np.shape(data)[0]):# trnasform into watts
        
        try:    
            data2=data[data_row,:]
        except:
            data2=data[data_row][0]
             
        data_watts.append(10**(data2/10))
    for i in range(len(data_watts[0])): # combining all uncertainties making sum of squares and the sqrt of the sum
        zipped_data.append(list(zip(*data_watts))[i])
        res_watts.append(np.sqrt(sum(map (lambda x: x**2,zipped_data[i]))))
        res_dB=10*np.log10(res_watts)
    del data2
    return np.array(res_dB)

#%% Spherical into cartesian  coordinate transformation
    #    xcart = rho * cos(phi)*sin(theta)
    #    ycart = rho * sin(phi)*sin(theta)
    #    zcart = rho * cos(theta)
    
def sph2cart(Lidar): 
    x=[]
    y=[]
    z=[]
    
    for i in range(len(Lidar.optics.scanner.focus_dist)):
        x=Lidar.optics.scanner.focus_dist[i]*np.cos(np.deg2rad(Lidar.optics.scanner.phi))*np.sin(np.deg2rad(Lidar.optics.scanner.theta))
        y=Lidar.optics.scanner.focus_dist[i]*np.sin(np.deg2rad(Lidar.optics.scanner.phi))*np.sin(np.deg2rad(Lidar.optics.scanner.theta)) 
        z=Lidar.optics.scanner.focus_dist[i]*np.cos(np.deg2rad(Lidar.optics.scanner.theta)) 
    return(x,y,z)


#%% Get data frame builds up a data frame with the uncertainty data that user queries
def Get_DataFrame(H_UQ,Temperature):  
#    pdb.set_trace()
    indexesDF=list(H_UQ.keys())
    
#    for fromModkeys in list(H_UQ.keys()):    
#        indexesDF
##        indexesDF.update((reduce(getitem,[fromModkeys],H_UQ)))
#    indexesDF=list(indexesDF.keys())
    columnsDF=['T= {} °K'.format(Temperature[i]) for i in range(len(Temperature))] # Getting data frame columns          
#    pdb.set_trace()
#    Full_df=([H_UQ[components].get(noise_type,{}) for components  in H_UQ.keys() for noise_type in H_UQ [components].keys() ])
    
    Full_df=pd.DataFrame(H_UQ, columns=indexesDF,index=columnsDF)
#    Full_df=Full_df.T
#    #Sum af decibels:
#    in_dB=0
#    Sum_decibels= []
#    for valcols in range(0,Full_df.shape[1]):
#        Sum_in_dB     = sum([(10**(Full_df.iloc[valrows,valcols]/10)) for valrows in range(0,Full_df.shape[0])])
#    #    Sum_in_dB = sum(in_dB)
#        Sum_decibels.append(10*np.log10(Sum_in_dB) )
#    
#    
#    Full_df.loc['Total UQ']= Sum_decibels# for now sum the uncertainties. Here have to apply Uncertainty expansion.
#    
#
#   #transform in watts. We supose that raw data is in dB:
#    df_UQ['Hardware (w)']=(10**(df_UQ['Hardware (dB)']/10))



    return Full_df  
