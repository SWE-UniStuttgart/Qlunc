# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:51:36 2020

@author: fcosta
"""

#import LiUQ_inputs
#import pandas as pd
#import scipy.interpolate as itp
from Qlunc_ImportModules import *
#import LiUQ_inputs

def UQ_Photodetector(**Scenarios):
    UQ_photodetector=[]
    for i in range(len(Scenarios.get('VAL_T'))):
        UQ_photodetector.append(Scenarios.get('VAL_T')[i]*0.4+Scenarios.get('VAL_H')[i]*0.1+Scenarios.get('VAL_NOISE_PHOTO')[i]+Scenarios.get('VAL_OC_PHOTO')[i]+Scenarios.get('VAL_WAVE')[i]/1000)
#    UQ_photodetector=[round(UQ_photodetector[i_dec],3) for i_dec in range(len(UQ_photodetector))] # 3 decimals
    return UQ_photodetector

def UQ_Optical_amplifier(**Scenarios):
    UQ_Optical_amplifier=[]
    for i in range(len(Scenarios.get('VAL_T'))):
        UQ_Optical_amplifier.append(Scenarios.get('VAL_T')[i]*0.5+Scenarios.get('VAL_H')[i]*0.7+Scenarios.get('VAL_NOISE_AMPLI')[i]+Scenarios.get('VAL_OC_AMPLI')[i])
#    UQ_Optical_amplifier=[round(UQ_Optical_amplifier[i_dec],3) for i_dec in range(len(UQ_Optical_amplifier))]
    return UQ_Optical_amplifier

def UQ_LaserSource(**Scenarios):
    UQ_laser_source=[]
    for i in range(len(Scenarios.get('VAL_T'))):
        UQ_laser_source.append(Scenarios.get('VAL_T')[i]*1+Scenarios.get('VAL_H')[i]*0.1+Scenarios.get('VAL_WAVE')[i]/1200+Scenarios.get('VAL_NOISE_LASER_SOURCE')[i]+Scenarios.get('VAL_OC_LASER_SOURCE')[i])
#    UQ_laser_source=[round(UQ_laser_source[i_dec],3) for i_dec in range(len(UQ_laser_source))]
    return UQ_laser_source


def FigNoise(inputs,direct,**Scenarios):
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





#def Photodetector_Noise(inputs):
#    Photodetector_noise_DATA=pd.read_excel(direct.Main_directory+photodetector_Noise_FILE) #read from an excel file variation of dB with Wavelength(for now just with wavelegth)
#    Photodetector_noise_INT=itp.interp1d(Photodetector_noise_DATA.iloc[:,0],Photodetector_noise_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
##    NEP_Lambda=NEP_min*(RespMAX/RespLambda) #NoiseEquivalentPower
#    Pmin=NEP_Lambda+np.sqrt(BW)#Minimum detectable signal power BW is teh band width
#    Photodetector_noise_VALUE=Photodetector_noise_INT(freq) # in dB
#    Photodetector_N=Photodetector_noise_VALUE.tolist()
#    return Photodetector_N