# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:58:24 2020

@author: fcosta
"""


#import sys
#sys.path.insert(0, direct.Main_directory)
#from Qlunc_ImportModules import *
#from Utils.Qlunc_ImportModules import *

def UQ_Telescope(inputs,cts,direct,**Scenarios):
    UQ_telescope=[]
    for i in range(len(Scenarios.get('Temperature'))):
        UQ_telescope.append(Scenarios.get('Temperature')[i]*0.5+Scenarios.get('VAL_H')[i]*0.1+Scenarios.get('VAL_CURVE_LENS_TELESCOPE')[i]*0.1+Scenarios.get('VAL_ABERRATION_TELESCOPE')[i]+Scenarios.get('VAL_OC_TELESCOPE')[i])

#    UQ_telescope=[round(UQ_telescope[i_dec],3) for i_dec in range(len(UQ_telescope))]
#    toreturn['telescope_atm_unc']=UQ_telescope
#    toreturn['telescope_losses']=Telescope_Losses
    return UQ_telescope

def Losses_Telescope(inputs,cts,direct,**Scenarios):
    Losses_telescope=[]
    for i in range(len(Scenarios.get('Temperature'))):
        Losses_telescope.append(Scenarios.get('VAL_LOSSES_TELESCOPE')[i])
    return Losses_telescope