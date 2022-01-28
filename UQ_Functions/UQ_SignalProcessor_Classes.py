# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:41:29 2022

@author: fcosta
"""

from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
from Utils import Qlunc_Plotting as QPlot

#%% Analog to digital converter

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
    # UQ_ADC.Thermal_noise =[]
    # UQ_ADC.ADC_quant_err =[]
    # UQ_ADC.ADC_resolution_V =[]
    pdb.set_trace()
    # Resolution:
    UQ_ADC.ADC_resolution_V  = (Lidar.signal_processor.analog2digital_converter.vref-Lidar.signal_processor.analog2digital_converter.vground)/(2**Lidar.signal_processor.analog2digital_converter.nbits)# =1LSB
    ADC_resolution_dB = 10**((Lidar.signal_processor.analog2digital_converter.vref-Lidar.signal_processor.analog2digital_converter.vground)/(2**Lidar.signal_processor.analog2digital_converter.nbits)/20 )# =1LSB
    # ADC_DR_dB         = 20*np.log10(((2**Lidar.signal_processor.analog2digital_converter.nbits)-1))
    # ADC_DR_V          = 10**(ADC_DR_dB/20)

    # ADC_FS = Lidar.signal_processor.analog2digital_converter.vref-ADC_resolution# ADC full scale
    pdb.set_trace()
    ## Noise added by an ADC 
    UQ_ADC.ADC_quant_err  = 0.5*UQ_ADC.ADC_resolution_V
    UQ_ADC.Thermal_noise = cts.k*Atmospheric_Scenario.temperature[0]*Lidar.signal_processor.analog2digital_converter.BandWidth
    UQ_ADC.UQ_ADC_Total =  np.sqrt(UQ_ADC.ADC_quant_err**2+UQ_ADC.Thermal_noise**2)
    Final_Output_UQ_ADC = {'ADC_Noise':UQ_ADC.UQ_ADC_Total,'ADC_Resolution':UQ_ADC.ADC_resolution_V}
    
    #Ideal ADC SNR
    SNR_ideal_dB = 6.02*Lidar.signal_processor.analog2digital_converter.nbits+1.76 # dB --> by definition
    SNR_ideal_watts = 10**(SNR_ideal_dB/10)
    
    
    Lidar.lidar_inputs.dataframe['ADC_Noise']=Final_Output_UQ_ADC['ADC_Noise']*np.linspace(1,1,len(Atmospheric_Scenario.temperature)) # linspace to create the appropriate length for the xarray.
    # pdb.set_trace()
    return Final_Output_UQ_ADC,Lidar.lidar_inputs.dataframe

#%% Frequency analyser
def UQ_FrequencyAnalyser(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    print('Frequency analyser')
#%% Sum of uncertainties in `signal processor` module: 
def sum_unc_signal_processor(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    # pdb.set_trace()
    List_Unc_signal_processor=[]
    if Lidar.signal_processor.analog2digital_converter != None:
        try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations               
                ADC_Uncertainty,DataFrame=Lidar.signal_processor.analog2digital_converter.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
                List_Unc_signal_processor.append(ADC_Uncertainty['ADC_Noise'])      
        except:
            ADC_Uncertainty=None
            print(colored('Error in ADC uncertainty calculations!','cyan', attrs=['bold']))
    else:
        print (colored('You didn´t include an analog to digital converter in the lidar.','cyan', attrs=['bold']))       
    pdb.set_trace()
    Uncertainty_SignalProcessor_Module=SA.unc_comb(List_Unc_signal_processor)
    Final_Output_UQ_SignalProcessor = {'Uncertainty_SignalProcessor':Uncertainty_SignalProcessor_Module}
    # 
    Lidar.lidar_inputs.dataframe['SignalProcessor Module']=Final_Output_UQ_SignalProcessor['Uncertainty_SignalProcessor']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))  # linspace to create the appropiate length for the xarray. 
    pdb.set_trace()
    return Final_Output_UQ_SignalProcessor,Lidar.lidar_inputs.dataframe