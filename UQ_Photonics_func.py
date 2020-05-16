# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:51:36 2020

@author: fcosta
"""

import LiUQ_inputs
import pandas as pd
import scipy.interpolate as itp



def UQ_LaserSource(inputs):
    UQ_laser_source=[(temp*1+hum*0.1+wave/1200+noise_ls+o_c_ls) \
                     for wave       in inputs.lidar_inp.Lidar_inputs['Wavelength'] \
                     for temp       in inputs.atm_inp.Atmospheric_inputs['temperature']\
                     for hum        in inputs.atm_inp.Atmospheric_inputs['humidity']\
                     for noise_ls   in inputs.photonics_inp.LaserSource_uncertainty_inputs['noise_lasersource']\
                     for o_c_ls     in inputs.photonics_inp.LaserSource_uncertainty_inputs['OtherChanges_LaserSource']]
    return UQ_laser_source

def UQ_Amplifier(inputs):
    UQ_amplifier=[(temp*0.5+hum*0.7+noise_amp+o_c_amp)\
                  for temp      in inputs.atm_inp.Atmospheric_inputs['temperature']\
                  for hum       in inputs.atm_inp.Atmospheric_inputs['humidity']\
                  for noise_amp in inputs.photonics_inp.Amplifier_uncertainty_inputs['noise_amp'] \
                  for o_c_amp   in inputs.photonics_inp.Amplifier_uncertainty_inputs['OtherChanges_amp']]
    return UQ_amplifier

def FigNoise(inputs):
    NoiseFigure_DATA=pd.read_excel(inputs.directory+inputs.photonics_inp.Amplifier_uncertainty_inputs['NoiseFigure_FILE']) #read from an excel file variation of dB with wavelength(for now just with wavelegth)
    figure_noise_INT=itp.interp1d(NoiseFigure_DATA.iloc[:,0],NoiseFigure_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
    NoiseFigure_VALUE=figure_noise_INT(inputs.lidar_inp.Lidar_inputs['Wavelength']) # in dB
    FigureNoise=NoiseFigure_VALUE.tolist()
    return FigureNoise



def UQ_Photodetector(inputs):
    UQ_photodetector=[temp*0.4+hum*0.1+noise_photo+o_c_photo\
                      for temp        in inputs.atm_inp.Atmospheric_inputs['temperature'] \
                      for hum         in inputs.atm_inp.Atmospheric_inputs['humidity'] \
                      for noise_photo in inputs.photonics_inp.Photodetector_uncertainty_inputs['noise_photo'] \
                      for o_c_photo   in inputs.photonics_inp.Photodetector_uncertainty_inputs['OtherChanges_photo']]
    return UQ_photodetector

#def PhotoNoise(inputs):
#    Photodetector_noise_DATA=pd.read_excel(directory+Photodetector_Noise_FILE) #read from an excel file variation of dB with Wavelength(for now just with wavelegth)
#    Photodetector_noise_INT=itp.interp1d(Photodetector_noise_DATA.iloc[:,0],Photodetector_noise_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
##    NEP_Lambda=NEP_min*(RespMAX/RespLambda) #NoiseEquivalentPower
#    Pmin=NEP_Lambda+np.sqrt(BW)#Minimum detectable signal power BW is teh band width
#    Photodetector_noise_VALUE=Photodetector_noise_INT(freq) # in dB
#    Photodetector_Noise=Photodetector_noise_VALUE.tolist()
#    return Photodetector_Noise