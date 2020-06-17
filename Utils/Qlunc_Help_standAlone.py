# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:03:43 2020

@author: fcosta
"""
from Qlunc_ImportModules import *
from functools import reduce
from operator import getitem
from Qlunc_inputs import inputs
import pdb

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
def Sum_dB(W_data):
   #Sum af decibels:
    in_dB=0
    Sumat= []
    Sum_decibels=[]
    for ii in W_data:
        gg=((10**(ii/10)))    
        Sumat.append (gg)
    Sum_in_dB = sum(Sumat)
    Sum_decibels.append(10*np.log10(Sum_in_dB) )
#    pdb.set_trace()
    return Sum_decibels
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