# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:51:36 2020

@author: fcosta
"""

import Qlunc_inputs
#import pandas as pd
#import scipy.interpolate as itp
from Qlunc_ImportModules import *
#import LiUQ_inputs
import pdb
import Qlunc_Help_standAlone as SA
#%% PHOTODETECTOR
def UQ_Photodetector(user_inputs,inputs,cts,**Scenarios):
    UQ_Photodetector=[]
    Photodetector_SNR_thermal_noise=[]
    Photodetector_Thermal_noise=[]
    Photodetector_Shot_noise=[]
    Photodetector_SNR_Shot_noise=[]
    Photodetector_Dark_current_noise=[]
    Photodetector_SNR_DarkCurrent=[]
    Photodetector_TIA_noise=[]
    Photodetector_SNR_TIA=[]
    for i in range(len(Scenarios.get('VAL_T'))): # To loop over all Scenarios
#        UQ_photodetector.append(Scenarios.get('VAL_T')[i]*0.4+Scenarios.get('VAL_H')[i]*0.1+Scenarios.get('VAL_NOISE_PHOTO')[i]+Scenarios.get('VAL_OC_PHOTO')[i]+Scenarios.get('VAL_WAVE')[i]/1000)
        R = Scenarios.get('VAL_PHOTO_n')[i]*cts.e*Scenarios.get('VAL_WAVE')[i]/(cts.h*cts.c)  #[W/A]  Responsivity

        # Photodetector Thermal noise
        Photodetector_Thermal_noise.append((10*np.log10(4*cts.k*Scenarios.get('VAL_T')[i]/Scenarios.get('VAL_PHOTO_RL')[i])*Scenarios.get('VAL_PHOTO_BW')[i])) #[dBm]
        Photodetector_SNR_thermal_noise.append(10*np.log10(((R**2)/(4*cts.k*Scenarios.get('VAL_T')[i]*Scenarios.get('VAL_PHOTO_BW')[i]/Scenarios.get('VAL_PHOTO_RL')[i]))*(Scenarios.get('VAL_PHOTO_SP')[i]/1000)**2))
    
        #Photodetector shot noise:
        Photodetector_Shot_noise.append(10*np.log10(2*cts.e*R*Scenarios.get('VAL_PHOTO_BW')[i]*Scenarios.get('VAL_PHOTO_SP')[i]))
        Photodetector_SNR_Shot_noise.append(10*np.log10(((R**2)/(2*cts.e*R*Scenarios.get('VAL_PHOTO_BW')[i]))*(Scenarios.get('VAL_PHOTO_SP')[i])/1000))

        #Photodetector dark current noise
        Photodetector_Dark_current_noise.append(10*np.log10(2*cts.e*Scenarios.get('VAL_PHOTO_Id')[i]*Scenarios.get('VAL_PHOTO_BW')[i]*Scenarios.get('VAL_PHOTO_SP')[i]))
        Photodetector_SNR_DarkCurrent.append(10*np.log10(((R**2)/(2*cts.e*Scenarios.get('VAL_PHOTO_Id')[i]*Scenarios.get('VAL_PHOTO_BW')[i]))*((Scenarios.get('VAL_PHOTO_SP')[i]/1000)**2) ))

        if 'TIA_noise' in list(SA.flatten(user_inputs.user_itype_noise)):     # If TIA is included in the components:
            # Photodetector TIA noise
            Photodetector_TIA_noise.append( 10*np.log10(Scenarios.get('VAL_V_NOISE_TIA')[i]**2/Scenarios.get('VAL_GAIN_TIA')[i]**2))
            Photodetector_SNR_TIA.append(10*np.log10(((R**2)/(Scenarios.get('VAL_V_NOISE_TIA')[i]**2/Scenarios.get('VAL_GAIN_TIA')[i]**2))*(Scenarios.get('VAL_PHOTO_SP')[i]/1000)**2))
            UQ_Photodetector.append(SA.Sum_dB([Photodetector_Thermal_noise[i],Photodetector_Shot_noise[i],Photodetector_Dark_current_noise[i],Photodetector_TIA_noise[i]]))
        else:
             UQ_Photodetector.append(SA.Sum_dB([Photodetector_Thermal_noise[i],Photodetector_Shot_noise[i],Photodetector_Dark_current_noise[i]]))
        pdb.set_trace()
    
#    for nT in range(len(Photodetector_Thermal_noise)):
#        UQ_Photodetector.append(SA.Sum_dB([Photodetector_Thermal_noise[nT],Photodetector_Shot_noise[nT],Photodetector_Dark_current_noise[nT],Photodetector_TIA_noise[nT]]))
#    UQ_photodetector=[round(UQ_photodetector[i_dec],3) for i_dec in range(len(UQ_photodetector))] # 3 decimals
    return UQ_Photodetector



#%% OPTICAL AMPLIFIER
def UQ_Optical_amplifier(**Scenarios):
    UQ_Optical_amplifier=[]
    for i in range(len(Scenarios.get('VAL_T'))):
        UQ_Optical_amplifier.append(Scenarios.get('VAL_T')[i]*0.5+Scenarios.get('VAL_H')[i]*0.7+Scenarios.get('VAL_NOISE_AMPLI')[i]+Scenarios.get('VAL_OC_AMPLI')[i])
#    UQ_Optical_amplifier=[round(UQ_Optical_amplifier[i_dec],3) for i_dec in range(len(UQ_Optical_amplifier))]
    return UQ_Optical_amplifier


def FigNoise(inputs,direct,**Scenarios): # This is the error of the optical amplifier
    if inputs.photonics_inp.Optical_amplifier_uncertainty_inputs['Optical_amplifier_fignoise'][0]==0:
        FigureNoise=0
    else:
        NoiseFigure_DATA = pd.read_csv(direct.Main_directory+inputs.photonics_inp.Optical_amplifier_uncertainty_inputs['Optical_amplifier_fignoise'],delimiter=';',decimal=',') #read from an excel file variation of dB with wavelength(for now just with wavelegth)
        FigureNoise = []
        for i in range(len(Scenarios.get('VAL_T'))):
            figure_noise_INT  = itp.interp1d(NoiseFigure_DATA.iloc[:,0],NoiseFigure_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
            NoiseFigure_VALUE = figure_noise_INT(Scenarios.get('VAL_WAVE')[i]) # in dB
            FigureNoise.append(NoiseFigure_VALUE.tolist())
#    FigureNoise=[round(FigureNoise[i_dec],3) for i_dec in range(len(FigureNoise))]
    return FigureNoise


#%% LASER SOURCE
def UQ_LaserSource(**Scenarios):
    UQ_laser_source=[]
    for i in range(len(Scenarios.get('VAL_T'))):
        UQ_laser_source.append(Scenarios.get('VAL_T')[i]*1+Scenarios.get('VAL_H')[i]*0.1+Scenarios.get('VAL_WAVE')[i]/1200+Scenarios.get('VAL_NOISE_LASER_SOURCE')[i]+Scenarios.get('VAL_OC_LASER_SOURCE')[i])
#    UQ_laser_source=[round(UQ_laser_source[i_dec],3) for i_dec in range(len(UQ_laser_source))]
    return UQ_laser_source






#def Photodetector_Noise(inputs):
#    Photodetector_noise_DATA=pd.read_excel(direct.Main_directory+photodetector_Noise_FILE) #read from an excel file variation of dB with Wavelength(for now just with wavelegth)
#    Photodetector_noise_INT=itp.interp1d(Photodetector_noise_DATA.iloc[:,0],Photodetector_noise_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
##    NEP_Lambda=NEP_min*(RespMAX/RespLambda) #NoiseEquivalentPower
#    Pmin=NEP_Lambda+np.sqrt(BW)#Minimum detectable signal power BW is teh band width
#    Photodetector_noise_VALUE=Photodetector_noise_INT(freq) # in dB
#    Photodetector_N=Photodetector_noise_VALUE.tolist()
#    return Photodetector_N