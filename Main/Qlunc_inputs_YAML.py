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
import pandas as pd
import sys,inspect
from functools import reduce
from operator import getitem
import yaml
import pdb
# Which modules introduce incertalistinties?:
#flag_unc_Optical_amplifier     = True # if True I include Optical_amplifier uncertainty
#flag_unc_photodetector = True # if True I include photodetector uncertainty
#flag_unc_telescope     = True # if True I include telescope uncertainty
# Want to execute LiUQ?
#flag_exec_LiUQ         = True
flag_plot_signal_noise  = True

#%% Reading from YAML file:
with open (r'../GitHub_Qlunc/Qlunc_inputs_YAML.yaml','r') as file:
    Qlunc_yaml_inputs={}
    docs = yaml.load_all(file, Loader=yaml.FullLoader)
    for doc in docs:      
        for k, v in doc.items():     
            Qlunc_yaml_inputs.setdefault(k,v)

#%%Directories Class:
class direct():
    Main_directory=Qlunc_yaml_inputs['Main_directory'] # For now all data is stored here

#%% Constants:
class cts():
    k = 1.38064852e-23 # Boltzman constant:[m^2 kg s^-2 K^-1]
    h = 6.6207004e-34 # Plank constant [m^2 kg s^-1]
    e = 1.60217662e-19 # electron charge [C]
    c = 2.99792e8 #speed of light [m s^-1]


#%% Inputs class where are stored all inputs:
class inputs():
    
    
    modules = Qlunc_yaml_inputs['modules']
#    DP      = ['los'] # data processing methods we want to assess

#%% Atmospheric inputs:
    class atm_inp():
        if Qlunc_yaml_inputs['TimeSeries']:
            AtmosphericScenarios_TS = pd.read_csv(direct.Main_directory+Qlunc_yaml_inputs['Atmos_TS_FILE'],delimiter=';',decimal=',')
            pdb.set_trace()
            Atmospheric_inputs={'temperature' : list(AtmosphericScenarios_TS.loc[:,'T']),
                                'humidity'    : list(AtmosphericScenarios_TS.loc[:,'H']),
                                'rain'        : list(AtmosphericScenarios_TS.loc[:,'rain']),
                                'fog'         : list(AtmosphericScenarios_TS.loc[:,'fog']),#for rain and fog intensity intervals might be introduced [none,low, medium high]
                                'time'        : list(AtmosphericScenarios_TS.loc[:,'t'])} 
        else:    
            Atmospheric_inputs=Qlunc_yaml_inputs['Atmosph_inputs']
    
#%% General lidar layout inputs:
    class lidar_inp():
        Lidar_inputs = Qlunc_yaml_inputs['Lidar_inputs']# (wave:[m],Laser_power: [mW])
#        BW=  #Band width (MHz)
#        laser_input_power =  .001 #[W]
        
#%% Power Modules inputs:
#    class power_inp():
#        PowerSource_uncertainty_inputs = Qlunc_yaml_inputs['PowerSource_uncertainty_inputs']
#        Converter_uncertainty_inputs   = Qlunc_yaml_inputs['Converter_uncertainty_inputs']
#%% Photonics module inputs:
    class photonics_inp():
        Optical_amplifier_inputs = Qlunc_yaml_inputs['Optical_amplifier_inputs']
#        LaserSource_inputs       = Qlunc_yaml_inputs['LaserSource_uncertainty_inputs']

        
        Photodetector_inputs  = Qlunc_yaml_inputs['Photodetector_inputs']['Photodetector_noise']
#                                 'photodetector_Noise_FILE'   :'Noise_Photodetector.csv'}                                        
        TIA_inputs            = Qlunc_yaml_inputs['Photodetector_inputs']['TIA_noise']


#%% Optics module inputs    
#    class optics_inp():
#        Telescope_uncertainty_inputs      = Qlunc_yaml_inputs['Telescope_uncertainty_inputs']


#%%
#    class VAL(): These was created to pass None values to the loop when getting Scenarios. Probably not needed any more!!! I keep it just in case
#            [VAL_T,VAL_H,VAL_WAVE]=[None]*3
#             VAL_NOISE_CONVERTER,VAL_OC_CONVERTER,VAL_CONVERTER_LOSSES,
#             VAL_NOISE_POWER_SOURCE,VAL_OC_POWER_SOURCE,
#             VAL_NOISE_AMPLI,VAL_OC_AMPLI,VAL_NOISE_FIG,
#             VAL_NOISE_LASER_SOURCE,VAL_OC_LASER_SOURCE,
#             VAL_PHOTO_BW,VAL_PHOTO_RL,VAL_PHOTO_n,VAL_PHOTO_Id,VAL_PHOTO_SP,VAL_GAIN_TIA,VAL_V_NOISE_TIA,
#             VAL_CURVE_LENS_TELESCOPE,VAL_OC_TELESCOPE,VAL_ABERRATION_TELESCOPE,VAL_LOSSES_TELESCOPE]=[None]*24

class user_inputs():
    user_imodules=list(inputs.modules.keys())
    user_icomponents=[list(reduce(getitem,[i],inputs.modules).keys()) for i in inputs.modules.keys()]
    user_itype_noise= [list(inputs.modules[module].get(components,{})) for module in inputs.modules.keys() for components in inputs.modules[module].keys()]

