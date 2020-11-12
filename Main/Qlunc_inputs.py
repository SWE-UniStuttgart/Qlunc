# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:14:00 2020

@author: fcosta
"""
#%% Header:
#04272020 - Francisco Costa
#SWE - Stuttgart
#LiUQ inputs for different modules:
#Here the user can introduce parameters to calculate the uncertainty of the different modules and components.. Must be note units of the uncertainties: either dB or watts

# Inputs:
#    Modules can be: 'Optical_amplifier', 'telescope', 'photodetector'
#    Data processing methods can be: 'LOS', 'filtering methods'
#    Temperature= [-15,0,15] for example
#    Humidity = [0,25,50,75]% for example
#    rain        = False
#    fog         = False  #for rain and fog intensity intervals might be introduced [none,low, medium high]
#    Values of uncertainties will be introduced in dB (for now)
#%% Inputs:.

#import Data:
import sys
sys.path.insert(0, '../')
from Utils.Qlunc_ImportModules import *
#sys.path.insert(0, '../')
#
#import pandas as pd
#
#from functools import reduce
#from operator import getitem
#import pdb
# Which modules introduce incertainties?:
#flag_unc_Optical_amplifier     = True # if True I include Optical_amplifier uncertainty
#flag_unc_photodetector = True # if True I include photodetector uncertainty
#flag_unc_telescope     = True # if True I include telescope uncertainty
# Want to execute LiUQ?
#flag_exec_LiUQ         = True
flag_plot_signal_noise  = True


#%%Directories Class:
class direct():
    Main_directory = '../' # For now all data is stored here
    Inputs_dir         = Main_directory+'metadata/'
    Outputs_dir        = Main_directory+'Outputs/'
class outputs():
    HardwareU_DF_name= 'Hardware_Uncertainties_DF'
#%% Constants:
class cts():
    k = 1.38064852e-23 # Boltzman constant:[m^2 kg s^-2 K^-1]
    h = 6.6207004e-34 # Plank constant [m^2 kg s^-1]
    e = 1.60217662e-19 # electron charge [C]
    c = 2.99792e8 #speed of light [m s^-1]


#%% Inputs class where are stored all inputs:
class inputs():
    
    
    # Modules is a dictionary containing the lidar modules as a key. As values there is a nested dictionary containing components as keys and type of uncertainty as values.
    # Each of this values is related with a function which calculates this specific uncertainty. The relation between type of unc. and function calculating it is in LiUQ_core when defining methods.
    modules = {
#               'Power'     : {'Power_source'      :['Power_source_noise'],                    # for now: 'power_source_noise',...
#                              'Converter'         :['Converter_noise']},                     # for now:'converter_noise', 'converter_losses'...
##    
               'Photonics' : {'Photodetector'     :['Photodetector_noise','TIA_noise'],                   # for now:'Photodetector_noise'; May be include 'TIA_noise' if there is a transimpedance amplifier...
                              'Optical_amplifier' :['Optical_amplifier_noise']},#,                       #for now:  'Optical_amplifier_noise',... If user includes Optical_amplifier component in dictionary 'modules', figure noise is automatically included in calculations(if don't want to include it have to put 0 in 'Optical_amplifier_uncertainty_inputs')
#                              'Laser_source'      :['Laser_source_noise']} ,                  # for now:'laser_source_noise',...
                              
#               'Optics'    : {'Telescope'         :['Telescope_noise']   }                  # for now:'telescope_noise', 'telescope_losses'...
               }
#    DP      = ['los'] # data processing methods we want to assess

#%% Atmospheric inputs:
    class atm_inp():
        TimeSeries=False  # This defines whether we are using a time series (True) or single values (False) to describe the atmosphere (T, H, rain and fog) 
                          # If so we obtain a time series describing the noise implemented in the measurement.
        if TimeSeries:
            Atmos_TS_FILE           = 'AtmosphericScenarios.csv'
            AtmosphericScenarios_TS = pd.read_csv(direct.Inputs_dir+Atmos_TS_FILE,delimiter=';',decimal=',')
            Atmospheric_inputs={
                                'temperature' : list(AtmosphericScenarios_TS.loc[:,'T']),# [K]
                                'humidity'    : list(AtmosphericScenarios_TS.loc[:,'H']),# [%]
                                'rain'        : list(AtmosphericScenarios_TS.loc[:,'rain']),
                                'fog'         : list(AtmosphericScenarios_TS.loc[:,'fog']),
                                'time'        : list(AtmosphericScenarios_TS.loc[:,'t'])#for rain and fog intensity intervals might be introduced [none,low, medium high]
                                } 
        else:    
            Atmospheric_inputs={'temperature' : [300], # [K] HAve to put the same number of elements for temperature and humidity. Always in paired values [T,H]
                                'humidity'    : [12],      # [%]
#                                'rain'        : [True],
#                                'fog'         : [False]
                                }#for rain and fog intensity intervals might be introduced [none,low, medium high]
#%% General lidar layout inputs:
    class lidar_inp():
        Lidar_inputs = {'Wavelength' : [1550e-9],'Laser_power':[2]} # (wave:[m],Laser_power: [mW])
#        BW=  #Band width (MHz)
#        laser_input_power =  .001 #[W]
        
#%% Power Modules inputs:
    class power_inp():
        PowerSource_inputs = {'Power_source_noise':{'Power_source_noise'       : [.8],
                                          'Power_source_OtherChanges': [.09]
                                          }}
        Converter_inputs   = {'Converter_noise':{'Converter_noise'          : [4],
                                          'Converter_OtherChanges'   : [.005885]},
                              'Converter_losses':{     'Converter_losses'    :[0.056]
                                          }}    
    
#%% Photonics module inputs:
    class photonics_inp():
        Optical_amplifier_inputs      = {'Optical_amplifier_noise':{
                            #                                         'Optical_amplifier_noise'            : [0.7777],
                            #                                         'Optical_amplifier_OtherChanges'     : [.005],
                                                                     'Optical_amplifier_NF'         : 'NoiseFigure.csv',# [5,7],# 
                                                                     'Optical_amplifier_Gain'             : [30]
                                                                     } # dB
                                         }
#        LaserSource_inputs    = {'Laser_source_noise'         : [.855,70000],
#                                 'Laser_source_OtherChanges'  : [.07709]
#                                 }

        
        Photodetector_inputs          = {'Photodetector_noise':{
                                                                 'Photodetector_Bandwidth'    : [1e9],  #[Hz] Band width
                                                                 'Photodetector_RL'           : [50],#[ohms] Load resistor
                                                                 'Photodetector_Efficiency'   : [0.85],#efficiency of the photodiode:
                                                                 'Photodetector_DarkCurrent'  : [5e-9],#[A] Dark current intensity
                                                                 'Photodetector_Signal_power' : [1e-3]#[mW] input power in the photodetector
                                #                                 'photodetector_Noise_FILE'   :'Noise_Photodetector.csv'}
                                                                 },
                                         'TIA_noise'          :{
                                                                 'Gain_TIA': [5e3],
                                                                 'V_noise_TIA':[160e-6]
                                                                 }
                                     }

#%% Optics module inputs    
#    class optics_inp():
#        Telescope_inputs      = {'Telescope_noise':{
#                                    'Telescope_curvature_lens' : [.01],
#                                    'Telescope_OtherChanges'   : [.006],
#                                    'Telescope_aberration'     : [.0004],
#                                    'Telescope_losses'         : [.099]
#                                }}





class user_inputs(): # this class gathers modules, components and methods introduced by the user
    user_imodules=list(inputs.modules.keys())
    user_icomponents=[list(reduce(getitem,[i],inputs.modules).keys()) for i in inputs.modules.keys()]
    user_itype_noise= [list(inputs.modules[module].get(components,{})) for module in inputs.modules.keys() for components in inputs.modules[module].keys()]





