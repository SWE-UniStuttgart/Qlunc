# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:03:43 2020

@author: fcosta
"""

from Qlunc_ImportModules import *
import numpy as np

#import Qlunc_Help_standAlone as SA
#from Main.Qlunc_inputs import inputs

#%%# used to flatt in som points along the code:
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (list,tuple)) else (a,))) 

#%% Sum of dB:
def Sum_dB(W_data,*args,**kwargs):
   #Sum af decibels:
#    pdb.set_trace()
    Sumat= []
    Sum_decibels=[]
    for ii in W_data:
        gg=((10**(ii/10)))    
        Sumat.append (gg)
    Sum_in_dB = sum(Sumat)
    Sum_decibels.append(10*np.log10(Sum_in_dB) )
#    pdb.set_trace()
    return list (flatten(Sum_decibels))

#%% Combine uncertainties:
def unc_comb(data): # data is provided as a list of elements want to add on. Data is expected to be in dB (within this functions dB are transformed into watts). The uncertainty combination is made following GUM
    data_watts  = []
    res_dB      = []
    res_watts   = []
    zipped_data = []
    if not isinstance (data,np.ndarray):
        data=np.array(data)
#    pdb.set_trace() 
    
    for data_row in range(np.shape(data)[0]):# trnasform into watts
        
        try:    
            data2=data[data_row,:]
        except:
            data2=data[data_row][0]
             
        data_watts.append(10**(data2/10))
#    pdb.set_trace()
    for i in range(len(data_watts[0])): # combining all uncertainties making sum of squares and the sqrt of the sum
        zipped_data.append(list(zip(*data_watts))[i])
        res_watts.append(np.sqrt(sum(map (lambda x: x**2,zipped_data[i]))))
        res_dB=10*np.log10(res_watts)
    del data2
    return res_dB

#%% Spherical into cartesian  coordinate transformation
def sph2cart(Lidar): 
#    pdb.set_trace()
    x=[]
    y=[]
    z=[]
    
    for i in range(len(Lidar.optics.scanner.focus_dist)):
        x=Lidar.optics.scanner.focus_dist[i]*np.cos(np.deg2rad(Lidar.optics.scanner.phi))*np.sin(np.deg2rad(Lidar.optics.scanner.theta))
        y=Lidar.optics.scanner.focus_dist[i]*np.sin(np.deg2rad(Lidar.optics.scanner.phi))*np.sin(np.deg2rad(Lidar.optics.scanner.theta)) 
        z=Lidar.optics.scanner.focus_dist[i]*np.cos(np.deg2rad(Lidar.optics.scanner.theta)) 
    return(x,y,z)

    # POLARS:
    #def pol2cart(rho, phi):
    #    xcart= rho * np.cos(phi)
    #    ycart = rho * np.sin(phi)
    #    return(xcart, ycart)
        
    # SPHERICAL:
    #def sph2cart(rho, theta,phi): # Spherical into cartesian  coordinate transformation
    #    xcart = rho * np.cos(phi)*np.sin(theta)
    #    ycart = rho * np.sin(phi)*np.sin(theta)
    #    zcart = rho*np.cos(theta)
    #    return(xcart, ycart,zcart) 

