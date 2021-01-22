# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:50:18 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c) 
"""
from Qlunc_ImportModules import *
import Qlunc_Help_standAlone as SA
import pandas as pd
import scipy.interpolate as itp
import pdb

def UQ_PowerSource(Lidar, Atmospheric_Scenario,cts):
    UQ_power_source=[]
    for i in range(len(Atmospheric_Scenario.temperature)):
        UQ_power_source.append(Atmospheric_Scenario.temperature[i]+Lidar.power.power_source.Output_power*Lidar.power.power_source.Input_power)

#    UQ_power_source=[round(UQ_power_source[i_dec],3) for i_dec in range(len(UQ_power_source))]
    return UQ_power_source

def UQ_Converter(Lidar, Atmospheric_Scenario,cts):
    UQ_converter=[]
    for i in range(len(Atmospheric_Scenario.temperature)):
        UQ_converter.append(i*Lidar.lidar_inputs.Wavelength+Lidar.power.converter.Infinit*Atmospheric_Scenario.temperature[i])

#    UQ_converter=[round(UQ_converter[i_dec],3) for i_dec in range(len(UQ_converter))]
    return UQ_converter
#
#def Losses_Converter(user_inputs,inputs,cts,direct,Wavelength,**Scenarios):
#    Losses_converter=[]
#    for i in range(len(Scenarios.get('VAL_T'))):
#        Losses_converter.append(Scenarios.get('VAL_CONVERTER_LOSSES')[i])
#    return Losses_converter

def sum_unc_power(Lidar,Atmospheric_Scenario,cts): 
    try: # ecah try/except evaluates wether the component is included
        PowerSource_Uncertainty=Lidar.power.power_source.Uncertainty(Lidar,Atmospheric_Scenario,cts)
    except:
        PowerSource_Uncertainty=None
        print('No power source in calculations!')
    try:
        Converter_Uncertainty=Lidar.power.converter.Uncertainty(Lidar,Atmospheric_Scenario,cts)
    except:
        Converter_Uncertainty=None
        print('No converter in calculations!')
    List_Unc_power1=[]
    List_Unc_power0=[PowerSource_Uncertainty,Converter_Uncertainty]
    for x in List_Unc_power0:        
        if isinstance(x,list):   
            List_Unc_power0=([10**(i/10) for i in x]) # Make the list without None values and convert in watts(necessary for SA.unc_comb)
            List_Unc_power1.append([List_Unc_power0]) # Make a list suitable for unc.comb function
    Uncertainty_Power_Module=SA.unc_comb(List_Unc_power1)
    return list(SA.flatten(Uncertainty_Power_Module))
