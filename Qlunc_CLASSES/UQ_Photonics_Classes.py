# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:15:24 2020
Francisco Costa - SWE
Here we calculate the uncertainties related with components in the photonics module
@author: fcosta
"""
from Qlunc_ImportModules import *
import Qlunc_Help_standAlone as SA
import pandas as pd
import scipy.interpolate as itp

#%% PHOTODETECTOR
def UQ_Photodetector(Lidar,Atmospheric_Scenario,cts):
    UQ_Photodetector.Thermal_noise=[]
    UQ_Photodetector.SNR_thermal_noise=[]
    UQ_Photodetector.Shot_noise=[]
    UQ_Photodetector.SNR_Shot_noise=[]
    UQ_Photodetector.Dark_current_noise=[]
    UQ_Photodetector.SNR_DarkCurrent=[]
    UQ_Photodetector.UQ_Photo=[]
    UQ_Photodetector.SNR_TIA=[]
    UQ_Photodetector.TIA_noise=[]
    R = Lidar.photonics.photodetector.Efficiency*cts.e*Lidar.lidar_inputs.Wavelength/(cts.h*cts.c)  #[W/A]  Responsivity
    UQ_Photodetector.Responsivity = (R) # this notation allows me to get Responsivity from outside of the function 
    '''#Where are these noise definintions coming from (reference in literature)?'''
    pdb.set_trace()
    for i in range(len(Atmospheric_Scenario.temperature)):
        # Photodetector Thermal noise
        UQ_Photodetector.Thermal_noise.append(4*cts.k*Atmospheric_Scenario.temperature[i]/Lidar.photonics.photodetector.RL*Lidar.photonics.photodetector.BandWidth) #[dBm]
        UQ_Photodetector.SNR_thermal_noise.append(((R**2)/(4*cts.k*Atmospheric_Scenario.temperature[i]*Lidar.photonics.photodetector.BandWidth/Lidar.photonics.photodetector.RL))*(Lidar.photonics.photodetector.SignalPower/1000)**2)
    
    #Photodetector shot noise:
    UQ_Photodetector.Shot_noise     = [(2*cts.e*R*Lidar.photonics.photodetector.BandWidth*Lidar.photonics.photodetector.SignalPower)]*len(Atmospheric_Scenario.temperature) 
    UQ_Photodetector.SNR_Shot_noise = [(((R**2)/(2*cts.e*R*Lidar.photonics.photodetector.BandWidth))*(Lidar.photonics.photodetector.SignalPower)/1000)]*len(Atmospheric_Scenario.temperature) 
    #Photodetector dark current noise
    UQ_Photodetector.Dark_current_noise = [(2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth*Lidar.photonics.photodetector.SignalPower)]*len(Atmospheric_Scenario.temperature) 
    UQ_Photodetector.SNR_DarkCurrent    = [(((R**2)/(2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth))*((Lidar.photonics.photodetector.SignalPower/1000)**2))]*len(Atmospheric_Scenario.temperature) 

    
    if any(TIA_val == None for TIA_val in [Lidar.photonics.photodetector.Gain_TIA,Lidar.photonics.photodetector.V_Noise_TIA]): # If any value of TIA is None dont include TIA noise in estimations :
        UQ_Photodetector.UQ_Photo = [(SA.unc_comb([UQ_Photodetector.Thermal_noise,UQ_Photodetector.Shot_noise,UQ_Photodetector.Dark_current_noise]))]
    else:
        # Photodetector TIA noise
        UQ_Photodetector.TIA_noise = [(Lidar.photonics.photodetector.V_Noise_TIA**2/Lidar.photonics.photodetector.Gain_TIA**2)]*len(Atmospheric_Scenario.temperature) 
        UQ_Photodetector.SNR_TIA   = [(((R**2)/(Lidar.photonics.photodetector.V_Noise_TIA**2/Lidar.photonics.photodetector.Gain_TIA**2))*(Lidar.photonics.photodetector.SignalPower/1000)**2)]*len(Atmospheric_Scenario.temperature) 
#        pdb.set_trace()
        UQ_Photodetector.UQ_Photo.append(SA.unc_comb([UQ_Photodetector.Thermal_noise,UQ_Photodetector.Shot_noise,UQ_Photodetector.Dark_current_noise,UQ_Photodetector.TIA_noise]))
    UQ_Photodetector.UQ_Photo=list(SA.flatten(UQ_Photodetector.UQ_Photo))        
#    pdb.set_trace()
    return UQ_Photodetector.UQ_Photo



#%% OPTICAL AMPLIFIER
def UQ_Optical_amplifier(Lidar,Atmospheric_Scenario,cts): # Calculating ASE - Amplified Spontaneous Emission definition ((**Optics and Photonics) Bishnu P. Pal - Guided Wave Optical Components and Devices_ Basics, Technology, and Applications -Academic Press (2005))
    try:
    # obtain SNR from figure noise or pass directly numerical value:
        if isinstance (Lidar.photonics.optical_amp.NoiseFig, numbers.Number): #If user introduces a number or a table of values
            FigureNoise=[Lidar.photonics.optical_amp.NoiseFig]*len(Atmospheric_Scenario.temperature) #Figure noise vector
        else:
            NoiseFigure_DATA = pd.read_csv(Lidar.photonics.optical_amp.NoiseFig,delimiter=';',decimal=',') #read from a .csv file variation of dB with wavelength (for now just with wavelength)    
            figure_noise_INT  = itp.interp1d(NoiseFigure_DATA.iloc[:,0],NoiseFigure_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column SNR in dB
            NoiseFigure_VALUE = figure_noise_INT(Lidar.lidar_inputs.Wavelength) # in dB
            FigureNoise=(NoiseFigure_VALUE.tolist())
        
        UQ_Optical_amplifier = [10*np.log10((10**(FigureNoise/10))*cts.h*(cts.c/Lidar.lidar_inputs.Wavelength)*10**(Lidar.photonics.optical_amp.Gain/10))]*len(Atmospheric_Scenario.temperature) # ASE
    except:
        print('No Optical Amplifier implemented')
    return UQ_Optical_amplifier

#%% Sum of uncertainty components in photonics module: 
def sum_unc_photonics(Lidar,Atmospheric_Scenario,cts): 
    try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations
#        if Photodetector_Uncertainty not in locals():
        Photodetector_Uncertainty=Lidar.photonics.photodetector.Uncertainty(Lidar,Atmospheric_Scenario,cts)
    except:
        Photodetector_Uncertainty=None
        print('No photodetector in calculations!')
    try:
        Optical_Amplifier_Uncertainty=Lidar.photonics.optical_amp.Uncertainty(Lidar,Atmospheric_Scenario,cts)
    except:
        Optical_Amplifier_Uncertainty=None
        print('No OA in calculations!')
    
    
    
    List_Unc_photonics1=[]
    List_Unc_photonics0=[Photodetector_Uncertainty,Optical_Amplifier_Uncertainty]
    pdb.set_trace()
    for x in List_Unc_photonics0:
        
        if isinstance(x,list):
           
            List_Unc_photonics0=([10**(i/10) for i in x]) # Make the list without None values and convert in watts(necessary for SA.unc_comb)
            List_Unc_photonics1.append([List_Unc_photonics0]) # Make a list suitable for unc.comb function
#    pdb.set_trace()

    Uncertainty_Photonics_Module=SA.unc_comb(List_Unc_photonics1)
    return list(SA.flatten(Uncertainty_Photonics_Module))