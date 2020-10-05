# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:55:41 2020

@author: fcosta
"""
from Qlunc_ImportModules import *
import Qlunc_Help_standAlone as SA
# Calculates the lidar global uncertainty:

def sum_unc_lidar(Lidar,Atmospheric_Scenario,cts): 
    try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations
#        if Photodetector_Uncertainty not in locals():
        Photonics_Uncertainty=Lidar.photonics.Uncertainty(Lidar,Atmospheric_Scenario,cts)
    except:
        Photonics_Uncertainty=None
        print('No photonics module in calculations!')
    try:
        Optics_Uncertainty=Lidar.optics.Uncertainty(Lidar,Atmospheric_Scenario,cts)
    except:
        Optics_Uncertainty=None
        print('No optics module in calculations!')

    try:
        Power_Uncertainty=Lidar.power.Uncertainty(Lidar,Atmospheric_Scenario,cts)
    except:
        Power_Uncertainty=None
        print('No power module in calculations!')
    
    List_Unc_lidar1=[]
    List_Unc_lidar0=[Photonics_Uncertainty,Optics_Uncertainty,Power_Uncertainty]
#    pdb.set_trace()
    for x in List_Unc_lidar0:
        
        if isinstance(x,list):
           
            List_Unc_lidar0=([10**(i/10) for i in x]) # Make the list without None values and convert in watts(necessary for SA.unc_comb)
            List_Unc_lidar1.append([List_Unc_lidar0]) # Make a list suitable for unc.comb function
    pdb.set_trace()

    Uncertainty_Lidar=SA.unc_comb(List_Unc_lidar1)
    
    
    print('Lidar unc Done')
    return Uncertainty_Lidar