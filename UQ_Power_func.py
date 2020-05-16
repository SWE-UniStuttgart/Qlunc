# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:50:18 2020

@author: fcosta
"""

import LiUQ_inputs



def UQ_Converter(inputs):
    UQ_converter=[(temp*1+hum*0.1+noise_conv+wave/1100+o_c_conv) \
                  for wave in inputs.lidar_inp.Lidar_inputs['Wavelength'] \                 
                  for temp in inputs.atm_inp.Atmospheric_inputs['temperature']\                 
                  for hum in inputs.atm_inp.Atmospheric_inputs['humidity']\
                  for noise_conv in inputs.power_inp.Converter_uncertainty_inputs['noise_conv'] \
                  for o_c_conv in inputs.power_inp.Converter_uncertainty_inputs['OtherChanges_conv']]
    return UQ_converter

def UQ_PowerSource(inputs):
    UQ_power_source=[(temp*1+hum*0.1+wave/1000+noise_ps+o_c_ps) \
                     for wave     in inputs.lidar_inp.Lidar_inputs['Wavelength']\
                     for temp     in inputs.atm_inp.Atmospheric_inputs['temperature']\
                     for hum      in inputs.atm_inp.Atmospheric_inputs['humidity']\
                     for noise_ps in inputs.power_inp.PowerSource_uncertainty_inputs['noise_powersource'] \
                     for o_c_ps   in inputs.power_inp.PowerSource_uncertainty_inputs['OtherChanges_PowerSource']]
    return UQ_power_source