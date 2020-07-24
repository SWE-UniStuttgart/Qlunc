# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:41:23 2020

@author: fcosta
"""

import sys
sys.path.insert(0, '../')
from Utils.Qlunc_ImportModules import *


#%% Inputs class where are stored all inputs:
modules = {'Power'     : {'Power_source'      :['Power_source_noise'],                    # for now: 'power_source_noise',...
                          'Converter'         :['Converter_noise']},                     # for now:'converter_noise', 'converter_losses'...
    
           'Photonics' : {'Photodetector'     :['Photodetector_noise','TIA_noise'],                   # for now:'photodetector_noise'; May be include 'TIA_noise' if there is a transimpedance amplifier...
#                          'Optical_amplifier' :['Optical_amplifier_noise']},#,                       #for now:  'Optical_amplifier_noise',... If user includes Optical_amplifier component in dictionary 'modules', figure noise is automatically included in calculations(if don't want to include it have to put 0 in 'Optical_amplifier_uncertainty_inputs')
                              
          }}

#%% Atmospheric inputs:

    
Atmospheric_inputs={'temperature' : [300,325], # [K] HAve to put the same number of elements for temperature and humidity. Always in paired values [T,H]
                        'humidity'    : [12,12],      # [%]
#                                'rain'        : [True],
#                                'fog'         : [False]
                        }#for rain and fog intensity intervals might be introduced [none,low, medium high]
#%% General lidar layout inputs:
Lidar_inputs = {'Wavelength' : [1550e-9,1565e-9],'SampleRate':[2]} # (wave:[m],Laser_power: [Hz])

    
#%% Photonics module inputs:

Optical_amplifier      = {'Optical_amplifier_noise':{
                                                       'Optical_amplifier_NF'         : 'NoiseFigure.csv',# [5,7],# 
                                                       'Optical_amplifier_Gain'             : [30]
                                                     } # dB
                          }


Photodetector         = {'Photodetector_noise':{
                                                         'Photodetector_Bandwidth'    : [1e9],  #[Hz] Band width
                                                         'Photodetector_RL'           : [50],#[ohms] Load resistor
                                                         'Photodetector_Efficiency'   : [0.85],#efficiency of the photodiode:
                                                         'Photodetector_DarkCurrent'  : [5e-9],#[A] Dark current intensity
                                                         'Photodetector_Signal_power' : [1e-3]#[mW] input power in the photodetector
                                                         },
                        'TIA_noise'          :{
                                                         'Gain_TIA': [5e3],
                                                         'V_noise_TIA':[160e-6]
                                                         }
                        }
  
#%% Power Modules inputs:

PowerSource = {'Power_source_noise':{'InputPower'  : [.8],
                                     'OutputPower' : [.09]
                                          }}
Converter   = {'Converter_noise' :{'Frequency'     : [4],
                                   'BandWidth'     : [.005885]},
               'Converter_losses':{'Infinit'       : [0.056]
                                          }}    
    