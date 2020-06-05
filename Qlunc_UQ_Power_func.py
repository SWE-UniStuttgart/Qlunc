# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:50:18 2020

@author: fcosta
"""

from Qlunc_ImportModules import *


def UQ_PowerSource(user_inputs,inputs,cts,**Scenarios):
    UQ_power_source=[]
    for i in range(len(Scenarios.get('VAL_T'))):
        UQ_power_source.append(Scenarios.get('VAL_T')[i]*1+Scenarios.get('VAL_H')[i]*1+Scenarios.get('VAL_WAVE')[i]/1000+\
                        Scenarios.get('VAL_NOISE_POWER_SOURCE')[i]+Scenarios.get('VAL_OC_POWER_SOURCE')[i])

#    UQ_power_source=[round(UQ_power_source[i_dec],3) for i_dec in range(len(UQ_power_source))]
    return UQ_power_source

def UQ_Converter(user_inputs,inputs,cts,**Scenarios):
    UQ_converter=[]
    for i in range(len(Scenarios.get('VAL_T'))):
        UQ_converter.append(Scenarios.get('VAL_T')[i]*1+Scenarios.get('VAL_H')[i]*1+Scenarios.get('VAL_WAVE')[i]/1100+Scenarios.get('VAL_NOISE_CONVERTER')[i]+\
                     Scenarios.get('VAL_OC_CONVERTER')[i]+80*Scenarios.get('VAL_CONVERTER_LOSSES')[i])

#    UQ_converter=[round(UQ_converter[i_dec],3) for i_dec in range(len(UQ_converter))]
    return UQ_converter

def Losses_Converter(user_inputs,inputs,cts,**Scenarios):
    Losses_converter=[]
    for i in range(len(Scenarios.get('VAL_T'))):
        Losses_converter.append(Scenarios.get('VAL_CONVERTER_LOSSES')[i])
    return Losses_converter