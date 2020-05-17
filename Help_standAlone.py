# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:03:43 2020

@author: fcosta
"""
from ImportModules import *
from LiUQ_inputs import inputs
# function used in core to crseate the possible components
def ext_comp(module,method,CL):
    CL[module]={}
    for ii in inputs.modules[module].keys(): #for components
        if ii in inputs.modules[module]: # if component is in the module
            CL[module][ii] =method[ii]
    return  