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

#%%# Get_Components is a function that extracts, for each module inside the class Hardware_U, in order to include 
#    in calculations, the components and the uncertainties (noise_type)
def Get_Components(module,CLASS,METHODS):
    
    for components in METHODS.keys(): #for components        
        CLASS[components]={}            
        for noise_type in inputs.modules[module][components]:
            CLASS[components][noise_type]=METHODS[components][noise_type]
    return CLASS


#%% Sum of dB:
def Sum_dB(W_data,*args,**kwargs):
   #Sum af decibels:
    pdb.set_trace()
    Sumat= []
    Sum_decibels=[]
    for ii in W_data:
        gg=((10**(ii/10)))    
        Sumat.append (gg)
    Sum_in_dB = sum(Sumat)
    Sum_decibels.append(10*np.log10(Sum_in_dB) )
    pdb.set_trace()
    return list (flatten(Sum_decibels))

#%% Combine uncertainties:
def unc_comb(data): # data is provided as list of elements want to add on. Data is expected to be in watts
    sqr=[]    
    sqr_db=[]
    Array=(np.array(data)) 

    for i in range(np.shape(Array)[1]):
        sq=[]
        for ii in range (len(Array)):
            sq.append(Array[ii,i]**2) # make the square of each value
        pdb.set_trace()
        sqr.append(np.sqrt(sum(sq))) #Sqrt of the values in the column
    for i_w in sqr:
        sqr_db.append(10*np.log10(i_w)) # en dB

    return sqr_db #The function offer both results, in watts and in dB
# %% Getting data frame:
    
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

