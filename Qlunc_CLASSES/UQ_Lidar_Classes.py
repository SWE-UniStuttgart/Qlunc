# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:55:41 2020

@author: fcosta
"""

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
    
    print('Lidar unc Done')