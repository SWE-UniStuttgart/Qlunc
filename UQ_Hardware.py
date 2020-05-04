# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:54:19 2020

@author: fcosta
"""
#%%Header
# Here we create the functions to calculate the uncertainties related with hardware
# Amplifier
#Telescope
#Photodetector

#%%
#28.04.2020

# code to quantify amplifier uncertainty taken into account all possible methods

def UQ_Amplifier(temperature,humidity,noise_amp,o_c_amp):
    UQ_amplifier=temperature*0.005+humidity*0.0007+noise_amp+o_c_amp
    return UQ_amplifier

def UQ_Telescope(temperature,humidity,curvature_lens,aberration,o_c_tele):
    UQ_telescope=temperature*0.005+humidity*0.01+curvature_lens*0.1+aberration+o_c_tele
    return UQ_telescope

def UQ_Photodetector(temperature,humidity,noise_photo,o_c_photo):
    UQ_photodetector=temperature*0.0004+humidity*0.0001+noise_photo+o_c_photo
    return UQ_photodetector