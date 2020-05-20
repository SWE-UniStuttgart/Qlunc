# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:03:43 2020

@author: fcosta
"""
from ImportModules import *

# ext_comp(exctract_component) is a function that extracts, for each module inside the class Hardware_U, in order to include 
# in calculations the components and the uncertainties (noise_type)
def ext_comp(module,method,CLASS):
    CLASS[module]={}
    for components in inputs.modules[module].keys(): #for components        
        CLASS[module][components]={}            
        for noise_type in inputs.modules[module][components]:
                CLASS[module][components][noise_type]=method[components][noise_type]
    return 



# fill missing values of data frame when time series is selected:
def fill_values(DFm):  
    Full_df=[DFm[module].get(components,{}).get(methods) for module  in DFm.keys() for components in DFm [module].keys() for methods in  DFm [module][components].keys() ]
    long=[(len(i1)) for i1 in Full_df]    
    if len(np.unique(long))>1:
        indices = [i2 for i2, v in enumerate(long) if v == min(long)]
        for index_min in indices:
            Full_df[index_min]=Full_df[index_min]*max(long)
    return Full_df  