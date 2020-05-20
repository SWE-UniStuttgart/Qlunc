# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:51:36 2020

@author: fcosta
"""

#import LiUQ_inputs
#import pandas as pd
#import scipy.interpolate as itp
from ImportModules import *
import LiUQ_inputs

def UQ_Photodetector(inputs):
    if inputs.atm_inp.TimeSeries==True:
        UQ_photodetector=[inputs.atm_inp.Atmospheric_inputs['temperature'][i]*0.4+
                          inputs.atm_inp.Atmospheric_inputs['humidity'][i]*0.1+
                          inputs.photonics_inp.Photodetector_uncertainty_inputs['noise_photo'][0]+
                          inputs.photonics_inp.Photodetector_uncertainty_inputs['OtherChanges_photo'][0]\
                          for i in range(len(inputs.atm_inp.Atmospheric_inputs['temperature']))]
    else:               
        UQ_photodetector=[temp*0.4+hum*0.1+noise_photo+o_c_photo\
                          for temp        in inputs.atm_inp.Atmospheric_inputs['temperature'] \
                          for hum         in inputs.atm_inp.Atmospheric_inputs['humidity'] \
                          for noise_photo in inputs.photonics_inp.Photodetector_uncertainty_inputs['noise_photo'] \
                          for o_c_photo   in inputs.photonics_inp.Photodetector_uncertainty_inputs['OtherChanges_photo']]
    UQ_photodetector=[round(UQ_photodetector[i_dec],3) for i_dec in range(len(UQ_photodetector))] # 3 decimals
    return UQ_photodetector

def UQ_Amplifier(inputs):
     if inputs.atm_inp.TimeSeries==True:
        UQ_amplifier=[inputs.atm_inp.Atmospheric_inputs['temperature'][i]*0.5+
                      inputs.atm_inp.Atmospheric_inputs['humidity'][i]*0.7+
                      inputs.photonics_inp.Amplifier_uncertainty_inputs['noise_amp'][0]+
                      inputs.photonics_inp.Amplifier_uncertainty_inputs['OtherChanges_amp'][0]\
                      for i in range(len(inputs.atm_inp.Atmospheric_inputs['temperature']))]
     else:               
        UQ_amplifier=[(temp*0.5+hum*0.7+noise_amp+o_c_amp)\
                      for temp      in inputs.atm_inp.Atmospheric_inputs['temperature']\
                      for hum       in inputs.atm_inp.Atmospheric_inputs['humidity']\
                      for noise_amp in inputs.photonics_inp.Amplifier_uncertainty_inputs['noise_amp'] \
                      for o_c_amp   in inputs.photonics_inp.Amplifier_uncertainty_inputs['OtherChanges_amp']]
     UQ_amplifier=[round(UQ_amplifier[i_dec],3) for i_dec in range(len(UQ_amplifier))]
     return UQ_amplifier

def UQ_LaserSource(inputs):
    if inputs.atm_inp.TimeSeries==True:
        UQ_laser_source=[inputs.atm_inp.Atmospheric_inputs['temperature'][i]*1+
                          inputs.atm_inp.Atmospheric_inputs['humidity'][i]*0.1+
                          inputs.lidar_inp.Lidar_inputs['Wavelength'][0]/1200+
                          inputs.photonics_inp.LaserSource_uncertainty_inputs['noise_lasersource'][0]+
                          inputs.photonics_inp.LaserSource_uncertainty_inputs['OtherChanges_lasersource'][0]\
                          for i in range(len(inputs.atm_inp.Atmospheric_inputs['temperature']))]
    else: 
        UQ_laser_source=[(temp*1+hum*0.1+wave/1200+noise_ls+o_c_ls) \
                         for wave       in inputs.lidar_inp.Lidar_inputs['Wavelength'] \
                         for temp       in inputs.atm_inp.Atmospheric_inputs['temperature']\
                         for hum        in inputs.atm_inp.Atmospheric_inputs['humidity']\
                         for noise_ls   in inputs.photonics_inp.LaserSource_uncertainty_inputs['noise_lasersource']\
                         for o_c_ls     in inputs.photonics_inp.LaserSource_uncertainty_inputs['OtherChanges_lasersource']]
    UQ_laser_source=[round(UQ_laser_source[i_dec],3) for i_dec in range(len(UQ_laser_source))]
    return UQ_laser_source


def FigNoise(inputs,direct):
    NoiseFigure_DATA=pd.read_excel(direct.Main_directory+inputs.photonics_inp.Amplifier_uncertainty_inputs['NoiseFigure_FILE']) #read from an excel file variation of dB with wavelength(for now just with wavelegth)
    figure_noise_INT=itp.interp1d(NoiseFigure_DATA.iloc[:,0],NoiseFigure_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
    NoiseFigure_VALUE=figure_noise_INT(inputs.lidar_inp.Lidar_inputs['Wavelength']) # in dB
    FigureNoise=NoiseFigure_VALUE.tolist()
    FigureNoise=[round(FigureNoise[i_dec],3) for i_dec in range(len(FigureNoise))]
    return FigureNoise





#def PhotoNoise(inputs):
#    Photodetector_noise_DATA=pd.read_excel(direct.Main_directory+Photodetector_Noise_FILE) #read from an excel file variation of dB with Wavelength(for now just with wavelegth)
#    Photodetector_noise_INT=itp.interp1d(Photodetector_noise_DATA.iloc[:,0],Photodetector_noise_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
##    NEP_Lambda=NEP_min*(RespMAX/RespLambda) #NoiseEquivalentPower
#    Pmin=NEP_Lambda+np.sqrt(BW)#Minimum detectable signal power BW is teh band width
#    Photodetector_noise_VALUE=Photodetector_noise_INT(freq) # in dB
#    Photodetector_Noise=Photodetector_noise_VALUE.tolist()
#    return Photodetector_Noise