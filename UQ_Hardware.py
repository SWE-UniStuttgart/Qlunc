# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:54:19 2020

@author: fcosta
"""
#%%Header
#04272020 - Francisco Costa
#SWE - Stuttgart
# Here we create the functions to calculate the uncertainties related with hardware
# Amplifier
#Telescope
#Photodetector

#%%
import pandas as pd
from LiUQ_inputs import directory, NoiseFigure_FILE 
import scipy.interpolate as itp
# code to quantify amplifier uncertainty taken into account all possible methods:

# Calcuate differents uncertainties depending on atmospheric scenarios:
def UQ_Amplifier(Atmospheric_inputs,Amplifier_uncertainty_inputs):
    UQ_amplifier=[(temp*0.5+hum*0.7+noise_amp+o_c_amp) for temp in Atmospheric_inputs['temperature'] for hum in Atmospheric_inputs['humidity'] for noise_amp in Amplifier_uncertainty_inputs['noise_amp'] for o_c_amp in Amplifier_uncertainty_inputs['OtherChanges_amp']]
    return UQ_amplifier

def UQ_Telescope(Atmospheric_inputs,Telescope_uncertainty_inputs):
    UQ_telescope=[(temp*0.5+hum*0.1+curvature_lens*0.1+aberration+o_c_tele) for temp in Atmospheric_inputs['temperature'] for hum in Atmospheric_inputs['humidity'] for curvature_lens in Telescope_uncertainty_inputs['curvature_lens'] for aberration in Telescope_uncertainty_inputs['aberration'] for o_c_tele in Telescope_uncertainty_inputs['OtherChanges_tele']]
    return UQ_telescope
#
def UQ_Photodetector(Atmospheric_inputs,Photodetector_uncertainty_inputs):
    UQ_photodetector=[temp*0.4+hum*0.1+noise_photo+o_c_photo for temp in Atmospheric_inputs['temperature'] for hum in Atmospheric_inputs['humidity'] for noise_photo in Photodetector_uncertainty_inputs['noise_photo'] for o_c_photo in Photodetector_uncertainty_inputs['OtherChanges_photo']]
    return UQ_photodetector

def FigNoise(wave):
    NoiseFigure_DATA=pd.read_excel(directory+NoiseFigure_FILE) #read from an excel file variation of dB with wavelength(for now just with wavelegth)
    figure_noise_INT=itp.interp1d(NoiseFigure_DATA.iloc[:,0],NoiseFigure_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
    NoiseFigure_VALUE=figure_noise_INT(wave) # in dB
    FigureNoise=NoiseFigure_VALUE.tolist()
    return FigureNoise

def PhotoNoise(freq):
    Photodetector_noise_DATA=pd.read_excel(directory+Photodetector_Noise_FILE) #read from an excel file variation of dB with wavelength(for now just with wavelegth)
    Photodetector_noise_INT=itp.interp1d(Photodetector_noise_DATA.iloc[:,0],Photodetector_noise_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
#    NEP_Lambda=NEP_min*(RespMAX/RespLambda) #NoiseEquivalentPower
    Pmin=NEP_Lambda+np.sqrt(BW)#Minimum detectable signal power BW is teh band width
    Photodetector_noise_VALUE=Photodetector_noise_INT(freq) # in dB
    Photodetector_Noise=Photodetector_noise_VALUE.tolist()
    return Photodetector_Noise
    