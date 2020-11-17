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
    UQ_Photodetector.Thermal_noise      = []
    UQ_Photodetector.SNR_thermal_noise  = []
    UQ_Photodetector.Shot_noise         = []
    UQ_Photodetector.SNR_Shot_noise     = []
    UQ_Photodetector.Dark_current_noise = []
    UQ_Photodetector.SNR_DarkCurrent    = []
    UQ_Photodetector.UQ_Photo           = []
    UQ_Photodetector.SNR_TIA            = []
    UQ_Photodetector.TIA_noise          = []
    SNR_data={}
    
    R = Lidar.photonics.photodetector.Efficiency*cts.e*Lidar.lidar_inputs.Wavelength/(cts.h*cts.c)  #[W/A]  Responsivity
    UQ_Photodetector.Responsivity = (R) # this notation allows me to get Responsivity from outside of the function 
    
    '''
    #Where are these noise definintions coming from (reference in literature)?
    '''

    UQ_Photodetector.SNR_thermal_noise = [10*np.log10(((R**2)/(4*cts.k*300*Lidar.photonics.photodetector.BandWidth/Lidar.photonics.photodetector.Load_Resistor))*(Lidar.photonics.photodetector.Power_interval/1000)**2)]
    UQ_Photodetector.SNR_Shot_noise    = [10*np.log10(((R**2)/(2*cts.e*R*Lidar.photonics.photodetector.BandWidth))*(Lidar.photonics.photodetector.Power_interval)/1000)]
    UQ_Photodetector.SNR_DarkCurrent   = [10*np.log10(((R**2)/(2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth))*((Lidar.photonics.photodetector.Power_interval/1000)**2))]
    for i in range(len(Atmospheric_Scenario.temperature)):
    # Photodetector Thermal noise
        UQ_Photodetector.Thermal_noise.append(4*cts.k*Atmospheric_Scenario.temperature[i]/Lidar.photonics.photodetector.Load_Resistor*Lidar.photonics.photodetector.BandWidth)         
    #Photodetector shot noise:
    UQ_Photodetector.Shot_noise     = [(2*cts.e*R*Lidar.photonics.photodetector.BandWidth*Lidar.photonics.photodetector.SignalPower)]*len(Atmospheric_Scenario.temperature)     
    #Photodetector dark current noise
    UQ_Photodetector.Dark_current_noise = [(2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth*Lidar.photonics.photodetector.SignalPower)]*len(Atmospheric_Scenario.temperature) 
#    pdb.set_trace()
    SNR_data={'SNR_Shot_Noise':UQ_Photodetector.SNR_Shot_noise,'SNR_Thermal':UQ_Photodetector.SNR_thermal_noise,'SNR_Dark_Current':UQ_Photodetector.SNR_DarkCurrent}  
    if any(TIA_val == None for TIA_val in [Lidar.photonics.photodetector.Gain_TIA,Lidar.photonics.photodetector.V_Noise_TIA]): # If any value of TIA is None dont include TIA noise in estimations :
        UQ_Photodetector.UQ_Photo = [(SA.unc_comb(10*np.log10([UQ_Photodetector.Thermal_noise,UQ_Photodetector.Shot_noise,UQ_Photodetector.Dark_current_noise])))]
        print('There is NO TIA component in the photodetector')
    else:
        # Photodetector TIA noise
        UQ_Photodetector.TIA_noise = [(Lidar.photonics.photodetector.V_Noise_TIA**2/Lidar.photonics.photodetector.Gain_TIA**2)]*len(Atmospheric_Scenario.temperature)
        UQ_Photodetector.SNR_TIA   = [10*np.log10(((R**2)/(Lidar.photonics.photodetector.V_Noise_TIA**2/Lidar.photonics.photodetector.Gain_TIA**2))*(Lidar.photonics.photodetector.Power_interval/1000)**2)]       

        UQ_Photodetector.UQ_Photo  = SA.unc_comb(10*np.log10([UQ_Photodetector.Thermal_noise,UQ_Photodetector.Shot_noise,UQ_Photodetector.Dark_current_noise,UQ_Photodetector.TIA_noise]))
        SNR_data['SNR_TIA']=UQ_Photodetector.SNR_TIA
        print('There is a TIA component in the photodetector')
#    pdb.set_trace()
    UQ_Photodetector.UQ_Photo=list(SA.flatten(UQ_Photodetector.UQ_Photo))
#    pdb.set_trace()
    Final_Output_UQ_Photo={'Uncertainty_Photodetector':UQ_Photodetector.UQ_Photo,'SNR_data_photodetector':SNR_data}      
    return Final_Output_UQ_Photo



#%% OPTICAL AMPLIFIER
'''
# ASE literature:
 - Calculating ASE - Amplified Spontaneous Emission definition ((**Optics and Photonics) Bishnu P. Pal - Guided Wave Optical Components and Devices_ Basics, Technology, and Applications -Academic Press (2005))
 - EDFA Tesing with interpolation techniques - Product note 71452-1
'''
def UQ_Optical_amplifier(Lidar,Atmospheric_Scenario,cts): 
#    try:
# obtain SNR from figure noise or pass directly numerical value:
    if isinstance (Lidar.photonics.optical_amplifier.NoiseFig, numbers.Number): #If user introduces a number or a table of values
        FigureNoise=[(Lidar.photonics.optical_amplifier.NoiseFig)]*len(Atmospheric_Scenario.temperature) #Figure noise vector        
        UQ_Optical_amplifier = [np.array([10*np.log10((10**(FigureNoise[0]/10))*cts.h*(cts.c/Lidar.lidar_inputs.Wavelength)*10**(Lidar.photonics.optical_amplifier.Gain/10))]*len(Atmospheric_Scenario.temperature))] # ASE
#        pdb.set_trace()
    else:
        NoiseFigure_DATA = pd.read_csv(Lidar.photonics.optical_amplifier.NoiseFig,delimiter=';',decimal=',') #read from a .csv file variation of dB with wavelength (for now just with wavelength)    
        figure_noise_INT  = itp.interp1d(NoiseFigure_DATA.iloc[:,0],NoiseFigure_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column SNR in dB
        NoiseFigure_VALUE = figure_noise_INT(Lidar.lidar_inputs.Wavelength) # in dB
        FigureNoise=(NoiseFigure_VALUE.tolist())
#        pdb.set_trace()
        UQ_Optical_amplifier = [np.array([10*np.log10((10**(FigureNoise/10))*cts.h*(cts.c/Lidar.lidar_inputs.Wavelength)*10**(Lidar.photonics.optical_amplifier.Gain/10))]*len(Atmospheric_Scenario.temperature)) ]# ASE

    
#    except:
#    print('No Optical Amplifier implemented')
    Final_Output_UQ_Optical_Amplifier={'Uncertainty_OpticalAmp':UQ_Optical_amplifier}
    return Final_Output_UQ_Optical_Amplifier

#%% Sum of uncertainty components in photonics module: 
def sum_unc_photonics(Lidar,Atmospheric_Scenario,cts): 
    List_Unc_photonics=[]
    try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations
#        if Photodetector_Uncertainty not in locals():
        Photodetector_Uncertainty=Lidar.photonics.photodetector.Uncertainty(Lidar,Atmospheric_Scenario,cts)
        List_Unc_photonics.append(Photodetector_Uncertainty['Uncertainty_Photodetector'])
    except:
        Photodetector_Uncertainty=None
        print('No photodetector in calculations!')
    try:
        Optical_Amplifier_Uncertainty=Lidar.photonics.optical_amplifier.Uncertainty(Lidar,Atmospheric_Scenario,cts)
        List_Unc_photonics.append(Optical_Amplifier_Uncertainty['Uncertainty_OpticalAmp'])
    except:
        Optical_Amplifier_Uncertainty=None
        print('No optical amplifier in calculations!')

    Uncertainty_Photonics_Module=SA.unc_comb(List_Unc_photonics)# to use SA.unc_comb we have to pass the data in watts
    Final_Output_UQ_Photonics={'Uncertainty_Photonics':Uncertainty_Photonics_Module}
    return Final_Output_UQ_Photonics