# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:57:05 2020
# The steps are as follows:
1) Insert a module
    - Create a class with the lidar components you want to include
    - Create an instance
    - unc_func is the uncertainty quantification function of the module.
      A function calculating the uncertainty sum of the components
      which made up the module is created.
2) Insert a component
    - Create a class with characteristics/input parameters of the lidar components
    - Create an instance
    - unc_func is the uncertainty quantification function of the component. We will 
      create a function calculating the combined uncertainty of each module
3) Insert an uncertainty method
4) Create the atmospheric scenarios
5) Create a lidar:
    
    
@author: fcosta
"""
from Qlunc_ImportModules import *
import UQ_Photonics_Classes as uphc
import UQ_Power_Classes as upwc
import UQ_Optics_Classes as uopc
#import pandas as pd
#import numpy as np
#import pdb

#%% Constants:
class cts():
    k = 1.38064852e-23 # Boltzman constant:[m^2 kg s^-2 K^-1]
    h = 6.6207004e-34 # Plank constant [m^2 kg s^-1]
    e = 1.60217662e-19 # electron charge [C]
    c = 2.99792e8 #speed of light [m s^-1]
    
#%% 
'''
#################################################################
###################### __ Classes __ #################################
Creating components, modules and lidar classes. This classes contain the object information required to build it up, also contain information
about the  formulation used to retrieve uncertainties (unc_func contains the function calculating the uncertainty for each component, module and/or 
lidar system)

 '''   
#Component Classes:
class photodetector():
    def __init__(self,name,Photo_BW,RL,n,DC,Photo_SP,G_TIA,V_noise_TIA,unc_func):
        self.PhotodetectorID  = name
        self.BandWidth        =Photo_BW
        self.RL               = RL
        self.Efficiency       = n
        self.DarkCurrent      = DC
        self.SignalPower      = Photo_SP
        self.Gain_TIA         = G_TIA
        self.V_Noise_TIA      = V_noise_TIA
        self.Uncertainty      = unc_func
        print('Created: {}'.format(self.PhotodetectorID))
class optical_amplifier():
    def __init__(self,name,OA_NF,OA_Gain,unc_func):
        self.Optical_AmplifierID = name
        self.NoiseFig            = OA_NF
        self.Gain                = OA_Gain
        self.Uncertainty         = unc_func
        print('Created: {}'.format(self.Optical_AmplifierID))
class power_source():
    def __init__(self,name,Inp_power,Out_power,unc_func):
        self.Power_SourceID = name
        self.Input_power    = Inp_power
        self.Output_power   = Out_power
        self.Uncertainty    = unc_func
        print('Created: {}'.format(self.Power_SourceID))
class converter():
    def __init__(self,name,frequency,Conv_BW,Infinit,unc_func):
        self.ConverterID = name
        self.Frequency   = frequency
        self.BandWidth   = Conv_BW
        self.Infinit     = Infinit
        self.Uncertainty = unc_func
        print('Created: {}'.format(self.ConverterID))

class scanner():
    def __init__(self,name,origin,sample_rate,focus_dist,theta,phi,stdv_focus_dist,stdv_theta,stdv_phi,unc_func):
        self.ScannerID            = name
        self.origin          = origin
        self.sample_rate     = sample_rate
        self.focus_dist      = focus_dist
        self.theta           = theta
        self.phi             = phi
        self.stdv_focus_dist = stdv_focus_dist
        self.stdv_theta      = stdv_theta
        self.stdv_phi        = stdv_phi
        self.Uncertainty     = unc_func
        
        print('Created: {}'.format(self.ScannerID))
    
#%%modules classes

class photonics():
    def __init__(self,name,photodetector,optical_amplifier,unc_func):
        self.PhotonicModuleID  = name
        self.photodetector     = photodetector
        self.optical_amp       = optical_amplifier
        self.Uncertainty       = unc_func 
        print('Created: {}'.format(self.PhotonicModuleID))

class power():
    def __init__(self,name,power_source,converter,unc_func):
        self.PoweModuleID  = name
        self.power_source  = power_source
        self.converter     = converter
        self.Uncertainty   = unc_func  
        print('Created: {}'.format(self.PoweModuleID))

class optics():
    def __init__(self,name,scanner,unc_func):
        self.OpticsModuleID = name
        self.scanner        = scanner
        self.Uncertainty    = unc_func 
        print('Created: {}'.format(self.OpticsModuleID))
        
#atmosphere object:
class atmosphere():
    def __init__(self,name,temperature):
        self.AtmosphereID = name
        self.temperature  = temperature
        print('Created: {}'.format(self.AtmosphereID))


#%% Creating lidar general data class:
class lidar_gral_inp():
    def __init__(self,name,wave,sample_rate):
        self.Gral_InputsID = name
        self.Wavelength    = wave
        self.SampleRate    = sample_rate
        print('Created: {}'.format(self.Gral_InputsID))

#%% Lidar class
class lidar():
    def __init__(self,name,photonics,optics,power,lidar_inputs):
        self.LidarID      = name
        self.photonics    = photonics
        self.optics       = optics
        self.power        = power 
        self.lidar_inputs = lidar_inputs
        print('Created: {}'.format(self.LidarID))


##%% ################################################################
####################### __ Instances __ #################################
#
#'''
#Here the lidar device, made up with respective modules and components.
#Instances of each component are created to build up the corresponding module. Once we have the components
#we create the modules, adding up the components. In the same way, once we have the different modules that we
#want to include in our lidar, we can create it adding up the modules we have been created.
#Many different lidars, atmospheric scenarios, components and modules can be created on paralel and can be combined
#easily 
#'''
##############  Optics ###################
#
## Components:
#
#Scanner1          = scanner(name           = 'Scan1',
#                           origin          = [0,0,0], #Origin
#                           focus_dist      = 80,
#                           sample_rate     = 10,
#                           theta           = np.array([5]),
#                           phi             = np.array([np.arange(0,360,10)]) ,
#                           stdv_focus_dist = 0.0,
#                           stdv_theta      = 0.0,
#                           stdv_phi        = 0.0,
#                           unc_func        = uopc.UQ_Scanner)       
#
#Scanner2          = scanner(name            = 'Scan2',
#                           origin          = [0,0,0], #Origin
#                           focus_dist      = 80,
#                           sample_rate     = 10,
#                           theta           = np.array([5]),
#                           phi             = np.array([np.arange(0,360,10)]) ,
#                           stdv_focus_dist = 0.9,
#                           stdv_theta      = 0.1,
#                           stdv_phi        = 0.1,
#                           unc_func        = uopc.UQ_Scanner)       
#
#Scanner3          = scanner(name           = 'Scan3',
#                           origin          = [0,0,0], #Origin
#                           focus_dist      = 80,
#                           sample_rate     = 10,
#                           theta           = np.array([5]),
#                           phi             = np.array([np.arange(0,360,10)]) ,
#                           stdv_focus_dist = 1.4,
#                           stdv_theta      = 0.3,
#                           stdv_phi        = 0.2,
#                           unc_func        = uopc.UQ_Scanner)       
#
#
## Module:
#
#Optics_Module1 =  optics (name     = 'OptMod1',
#                         scanner  = Scanner1,
#                         unc_func = uopc.UQ_Scanner)
#Optics_Module2 =  optics (name     = 'OptMod2',
#                         scanner  = Scanner2,
#                         unc_func = uopc.UQ_Scanner)
#Optics_Module3 =  optics (name     = 'OptMod3',
#                         scanner  = Scanner3,
#                         unc_func = uopc.UQ_Scanner)
#
##############  Photonics ###################
#
## Components:
#
#OpticalAmplifier = optical_amplifier(name     = 'OA1',
#                                     OA_NF    = 'NoiseFigure.csv',
#                                     OA_Gain  = 30,
#                                     unc_func = uphc.UQ_Optical_amplifier)
#
#Photodetector    = photodetector(name        = 'Photo1',
#                                 Photo_BW    = 1e9,
#                                 RL          = 50,
#                                 n           = .85,
#                                 DC          = 5e-9,
#                                 Photo_SP    = 1e-3,
#                                 G_TIA       = 5e3,
#                                 V_noise_TIA = 160e-6,
#                                 unc_func    = uphc.UQ_Photodetector)
#
## Module:
#
#Photonics_Module = photonics(name              = 'PhotoMod1',
#                             photodetector     = Photodetector,
#                             optical_amplifier = OpticalAmplifier,
#                             unc_func          = uphc.sum_unc_photonics)
#
##############  Power #########################################
#
## Components:
#
#PowerSource      = power_source(name      = 'P_Source1',
#                                Inp_power = 1,
#                                Out_power = 2,
#                                unc_func  = upwc.UQ_PowerSource)
#
#Converter        = converter(name      = 'Conv1',
#                             frequency = 100,
#                             Conv_BW   = 1e5,
#                             Infinit   = .8,
#                             unc_func  = upwc.UQ_Converter)
#
## Module:
#
#Power_Module     = power(name         = 'PowerMod1',
#                         power_source = PowerSource,
#                         converter    = Converter,
#                         unc_func     = upwc.sum_unc_power)
#
#
#
########### Lidar general inputs #########################:
#Lidar_inputs     = lidar_gral_inp(name        = 'Gral_inp1', 
#                                  wave        = 1550e-9,
#                                  sample_rate = 2)
#
#
###########  LIDAR  #####################################
#
#Lidar1 = lidar(name         = 'Caixa1',
#               photonics    = Photonics_Module,
#               optics       = Optics_Module1,
#               power        = Power_Module,
#               lidar_inputs = Lidar_inputs)
##              unc_func     = UQ_Lidar)
#Lidar2 = lidar(name         = 'Caixa2',
#               photonics    = Photonics_Module,
#               optics       = Optics_Module2,
#               power        = Power_Module,
#               lidar_inputs = Lidar_inputs)
##              unc_func     = UQ_Lidar)
#
#Lidar3 = lidar(name         = 'Caixa3',
#               photonics    = Photonics_Module,
#               optics       = Optics_Module3,
#               power        = Power_Module,
#               lidar_inputs = Lidar_inputs)
##              unc_func     = UQ_Lidar)
#
##%% Creating atmospheric scenarios:
#TimeSeries=True  # This defines whether we are using a time series (True) or single values (False) to describe the atmosphere (T, H, rain and fog) 
#                  # If so we obtain a time series describing the noise implemented in the measurement.
#if TimeSeries:
#    Atmos_TS_FILE           = 'AtmosphericScenarios.csv'
#    AtmosphericScenarios_TS = pd.read_csv(Atmos_TS_FILE,delimiter=';',decimal=',')
#    Atmospheric_inputs={
#                        'temperature' : list(AtmosphericScenarios_TS.loc[:,'T']),# [K]
#                        'humidity'    : list(AtmosphericScenarios_TS.loc[:,'H']),# [%]
#                        'rain'        : list(AtmosphericScenarios_TS.loc[:,'rain']),
#                        'fog'         : list(AtmosphericScenarios_TS.loc[:,'fog']),
#                        'time'        : list(AtmosphericScenarios_TS.loc[:,'t'])#for rain and fog intensity intervals might be introduced [none,low, medium high]
#                        } 
#    Atmospheric_Scenario=atmosphere(name        = 'Atmosphere1',
#                                    temperature = Atmospheric_inputs['temperature'])
#else:    
#
#    Atmospheric_Scenario=atmosphere(name        = 'Atmosphere1',
#                                    temperature = [300])
#%% Calculate uncertainties with lidar object and atmospheric scenarios defined previously:
# Photonics: 
#pdb.set_trace()
#try :
#    Photodetector_Uncertainty     = Lidar.photonics.photodetector.Uncertainty(Lidar,Atmospheric_Scenario,cts)
#    
#except:
#    print('No photodetector in photonics module')
#try:
#    Optical_Amplifier_Uncertainty = Lidar.photonics.optical_amp.Uncertainty(Lidar,Atmospheric_Scenario,cts)
#except:
#    print('No OA in photonics module')
#try:
#    Photonics_Uncertainty = Lidar.photonics.Uncertainty(Lidar,Atmospheric_Scenario,cts)
#except:
#    print('Photonics module: No photonics module in lidar')
#    
    
#Lidar_XR=xr.DataArray.from_dict(Lidar, ('name','modules','Lidar_inputs','Photodetector_inputs','Optical_amplifier_inputs')) #  transform in xarray
    
