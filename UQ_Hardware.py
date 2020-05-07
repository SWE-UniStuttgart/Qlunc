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
#28.04.2020

# code to quantify amplifier uncertainty taken into account all possible method


def UQ_Amplifier(Atmospheric_inputs,Amplifier_uncertainty_inputs):
    UQ_amplifier=[(temp*0.005+hum*0.0007+noise_amp+o_c_amp) for temp in Atmospheric_inputs['temperature'] for hum in Atmospheric_inputs['humidity'] for noise_amp in Amplifier_uncertainty_inputs['noise_amp'] for o_c_amp in Amplifier_uncertainty_inputs['OtherChanges_amp']]
    return UQ_amplifier

def UQ_Telescope(Atmospheric_inputs,Telescope_uncertainty_inputs):
    UQ_telescope=[(temp*0.005+hum*0.01+curvature_lens*0.1+aberration+o_c_tele) for temp in Atmospheric_inputs['temperature'] for hum in Atmospheric_inputs['humidity'] for curvature_lens in Telescope_uncertainty_inputs['curvature_lens'] for aberration in Telescope_uncertainty_inputs['aberration'] for o_c_tele in Telescope_uncertainty_inputs['OtherChanges_tele']]
    return UQ_telescope
#
def UQ_Photodetector(Atmospheric_inputs,Photodetector_uncertainty_inputs):
    UQ_photodetector=[temp*0.0004+hum*0.0001+noise_photo+o_c_photo for temp in Atmospheric_inputs['temperature'] for hum in Atmospheric_inputs['humidity'] for noise_photo in Photodetector_uncertainty_inputs['noise_photo'] for o_c_photo in Photodetector_uncertainty_inputs['OtherChanges_photo']]
    return UQ_photodetector
