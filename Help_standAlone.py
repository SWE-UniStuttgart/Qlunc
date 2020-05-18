# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:03:43 2020

@author: fcosta
"""
from ImportModules import *


# Function applied for each module inside the class Hardware_U to include in calculations the components and the uncertainties (noise_type) for each component.
def ext_comp(module,method,CLASS):
    CLASS[module]={}
    for components in inputs.modules[module].keys(): #for components        
        CLASS[module][components]={}            
        for noise_type in inputs.modules[module][components]:
            CLASS[module][components][noise_type]=method[components][noise_type]
    return 