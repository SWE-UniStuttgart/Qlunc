# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:41:29 2022

@author: fcosta
"""

from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
from Utils import Qlunc_Plotting as QPlot


def UQ_ADC(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    """
    Analog to digital converter uncertainty estimation. Location: ./UQ_Functions/UQ_SignalProcessor_Classes.py
    
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
    UQ_ADC.Thermal_noise =[]
    UQ_ADC.ADC_quant_err =[]
     
    # Resolution:
    ADC_resolution = (Lidar.signal_processor.analog2digital_converter.vref-Lidar.signal_processor.analog2digital_converter.vground)/(2**Lidar.signal_processor.analog2digital_converter.nbits) # =1LSB
    # ADC_FS = Lidar.signal_processor.analog2digital_converter.vref-ADC_resolution# ADC full scale

    ## Noise added by an ADC 
    UQ_ADC.ADC_quant_err  = 0.5*ADC_resolution
    UQ_ADC.Thermal_noise = cts.k*Atmospheric_Scenario.temperature[0]*Lidar.signal_processor.analog2digital_converter.BandWidth
    Final_Output_UQ_ADC = {'ADC_Uncertainty':np.sqrt(UQ_ADC.ADC_quant_err**2+UQ_ADC.Thermal_noise**2)}
    
    #Ideal ADC SNR
    SNR_ideal_dB = 6.02*Lidar.signal_processor.analog2digital_converter.nbits+1.76 # dB
    SNR_ideal_watts = 10**(SNR_ideal_dB/10)
    
    
    Lidar.lidar_inputs.dataframe['ADC_Uncertainty']=Final_Output_UQ_ADC['ADC_Uncertainty']*np.linspace(1,1,len(Atmospheric_Scenario.temperature)) # linspace to create the appropriate length for the xarray.
    # pdb.set_trace()
    return Final_Output_UQ_ADC,Lidar.lidar_inputs.dataframe


def sum_unc_signal_processor(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    # pdb.set_trace()
    List_Unc_signal_processor=[]
    if Lidar.signal_processor.analog2digital_converter != None:
        try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations               
                ADC_Uncertainty,DataFrame=Lidar.signal_processor.analog2digital_converter.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
                List_Unc_signal_processor.append(ADC_Uncertainty['ADC_Uncertainty'])      
        except:
            ADC_Uncertainty=None
            print(colored('Error in ADC uncertainty calculations!','cyan', attrs=['bold']))
    else:
        print (colored('You didn´t include an analog to digital converter in the lidar.','cyan', attrs=['bold']))       
     
    Uncertainty_SignalProcessor_Module=SA.unc_comb(List_Unc_signal_processor)
    Final_Output_UQ_SignalProcessor = {'Uncertainty_SignalProcessor':Uncertainty_SignalProcessor_Module}
    # 
    Lidar.lidar_inputs.dataframe['SignalProcessor Module']=Final_Output_UQ_SignalProcessor['Uncertainty_SignalProcessor']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))  # linspace to create the appropiate length for the xarray. 
    pdb.set_trace()
    return Final_Output_UQ_SignalProcessor,Lidar.lidar_inputs.dataframe