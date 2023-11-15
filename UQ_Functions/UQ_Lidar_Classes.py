# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:55:41 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c)

Here we calculate the uncertainty expansion from uncertainties obtained for
each lidar module following GUM (Guide to the expression of Uncertainties in 
Measurement) model. 
 
"""
from Utils.Qlunc_ImportModules import *
import Utils.Qlunc_Help_standAlone as SA

# Calculates the lidar global uncertainty using uncertainty expansion calculation methods:
def sum_unc_lidar(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    """
    Lidar uncertainty estimation. Location: ./UQ_Functions/UQ_Lidar_Classes.py
    
    Parameters
    ----------
    
    * Lidar
        data...
    * Atmospheric_Scenario
        Atmospheric data. Integer or Time series
    * cts
        Physical constants
    * Qlunc_yaml_inputs
        Lidar parameters data
        
    Returns
    -------
    
    list
    
    """ 
    List_Unc_lidar = []
    print(colored('Processing lidar uncertainties...','magenta', attrs=['bold']))
    

    
    # pdb.set_trace()    
    ### Photoniccs
    if Lidar.photonics != None:
        try: # each try/except evaluates whether the component is included in the module, therefore in the calculations
            Photonics_Uncertainty,DataFrame = Lidar.photonics.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
            try:
                # pdb.set_trace()
                List_Unc_lidar.append(DataFrame['Thermal noise']) 
                List_Unc_lidar.append(DataFrame['Shot noise']) 
                List_Unc_lidar.append(DataFrame['Dark current noise']) 
                List_Unc_lidar.append(DataFrame['TIA noise']) 
            except:
                print(colored('Error appending photodetetor noise components to the data frame.','cyan', attrs=['bold']))
            
        except:
            Photonics_Uncertainty = None
            print(colored('Error in photonics module calculations!','cyan', attrs=['bold']))
    else:
        print(colored('You didn´t include a photonics module in the lidar.','cyan', attrs=['bold']))
    
    ### Signal processor
    if Lidar.signal_processor != None:   
        try:
            SignalProcessor_Uncertainty,DataFrame = Lidar.signal_processor.analog2digital_converter.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
    
            List_Unc_lidar.append(SignalProcessor_Uncertainty['Stdv Vlos']*len(Atmospheric_Scenario.temperature))
        
        except:
            SignalProcessor_Uncertainty = None
            print(colored('No signal processor module in calculations!','cyan', attrs=['bold']))
    else:
        print(colored('You didn´t include a signal processor module in  the lidar.','cyan', attrs=['bold']))        

    ### Intrinsic lidar uncertainty:
    Lidar.lidar_inputs.dataframe['Intrinsic Uncertainty [m/s]']= SA.U_intrinsic(Lidar,DataFrame,Qlunc_yaml_inputs)
    
    
    ### Optics
    if Lidar.optics != None:
        DataFrame = Lidar.optics.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)     
    else:
        print(colored('You didn´t include an optics module in the lidar.','cyan', attrs=['bold']))
       
    print(colored('...Lidar uncertainty done. Lidar saved in folder "Lidar_Projects".','magenta', attrs=['bold']))
    return Lidar.lidar_inputs.dataframe
    # return Final_Output_Lidar_Uncertainty,Lidar.lidar_inputs.dataframe