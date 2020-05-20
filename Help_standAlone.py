# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:03:43 2020

@author: fcosta
"""
from ImportModules import *
from functools import reduce
from operator import getitem

# ext_comp(exctract_component) is a function that extracts, for each module inside the class Hardware_U, in order to include 
# in calculations the components and the uncertainties (noise_type)

def ext_comp(module,method,CLASS):
    CLASS[module]={}
    for components in inputs.modules[module].keys(): #for components        
        CLASS[module][components]={}            
        for noise_type in inputs.modules[module][components]:
                CLASS[module][components][noise_type]=method[components][noise_type]
    return 


# fill missing values of data frame:
#def fill_values(DFm):
#    mapeado=[]
#    for module in DFm.keys():
#    #     for comp in inputs.modules[module].keys():
#             mapeado.append(list(map(len,DFm[module]))) 
#    mapeado=tuple(*list(zip(*mapeado)))
#    DFm=list(itertools.chain(*DFm.values()))
#    long=[]
#    for i in DFm:
#        long.append(len(DFm))
#    number_long=np.unique (long)
#    if len(number_long)>1:       
#        ind=(min(mapeado))
#        indices = [i for i, v in enumerate(mapeado) if v == ind]
#        for i in indices:
#            print(i)
#            DFm[i]=DFm[i]*(max(mapeado))
#    return DFm
def fill_values(DFm):
  
    Full_df=[] 
    long=[]  
    for module in DFm.keys():
        for components in DFm[module].keys():
            for method in DFm[module][components].keys():
                Full_df.append(reduce(getitem,[module,components,method],DFm))
    
    long=[(len(i)) for i in Full_df]
    
    if len(np.unique(long))>1:
        indices = [i for i, v in enumerate(long) if v == min(long)]
        for index_min in indices:
            Full_df[index_min]=Full_df[index_min]*max(long)
    return Full_df  