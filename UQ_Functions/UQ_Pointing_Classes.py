# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 19:04:36 2022

@author: fcosta

Calculate the Vh uncertainty. The function reads the lidars in "Projects" and calculates the uncertainty in Vh if 2 lidars or 3D vector wind
velocity uncertainty if there are three lidars. 
"""
from Utils.Qlunc_ImportModules import *
import Utils.Qlunc_Help_standAlone as SA
def UQ_Vh(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    Lidars = ['Caixa1','Caixa2']
        # Read the saved dictionary
    loaded_dict=[]
    for ind in Lidars:
        with open('./Projects/'+ind, 'rb') as f:
            loaded_dict.append( pickle.load(f))


    if len (Lidars)==2:
        print('xx')        
        
        
        
        
        
        
        
        
    # Plotting
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_Scanner,True,False,False,False,False)  #Qlunc_yaml_inputs['Flags']['Scanning Pattern']