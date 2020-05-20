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
#    Modules can be: 'amplifier', 'telescope', 'photodetector'
#    Data processing methods can be: 'LOS', 'filtering methods'
#    Temperature= [-15,0,15] for example
#    Humidity = [0,25,50,75]% for example
#    rain        = False
#    fog         = False  #for rain and fog intensity intervals might be introduced [none,low, medium high]
#    Values of uncertainties will be introduced in dB (for now)
#%% Inputs:.

#import Data:
import pandas as pd
# Which modules introduce incertalistinties?:
#flag_unc_amplifier     = True # if True I include amplifier uncertainty
#flag_unc_photodetector = True # if True I include photodetector uncertainty
#flag_unc_telescope     = True # if True I include telescope uncertainty
# Want to execute LiUQ?
#flag_exec_LiUQ         = True
flag_plot_signal_noise  = True


#Directories Class:
class direct():
    Main_directory='../GitHub_LiUQ/'

# Inputs class where are stored all inputs:
class inputs():
    
    
    # Modules is a dictionary containing the lidar modules as a key. As values there is a nested dictionary containing components as keys and type of uncertainty as values.
    # Each of this values is related with a function which calculates this specific uncertainty. The relation between type of unc. and function calculating it is in LiUQ_core when defining methods.
    modules = {
               'power'     : {'power_source' :['power_source_noise'],                    # for now: 'power_source_noise',...
                              'converter'    :[]},                      # for now:'converter_noise', 'converter_losses'...
    
               'photonics' : {'photodetector':['photodetector_noise'],                   # for now:'photodetector_noise',....
                              'amplifier'    :['amplifier_fignoise','amplifier_noise'],  #for now: 'amplifier_fignoise', 'amplifier_noise',...
                              'laser_source' :['laser_source_noise']},                   # for now:'laser_source_noise',...
                              
               'optics'    : {'telescope'    :[]}                       # for now:'telescope_noise', 'telescope_losses'...
               }
#    DP      = ['los'] # data processing methods we want to assess

     
    # Atmospheric inputs:
    class atm_inp():
        TimeSeries=True # This defines whether we are using a time series to describe the atmosphere (T, H, rain and fog). 
                        # If so we obtain a time series describing the noise implemented in the measurement.
        if TimeSeries:
            AtmosphericScenarios_TS=pd.read_csv(direct.Main_directory+'AtmosphericScenarios.csv',delimiter=';',decimal=',')
            Atmospheric_inputs={'temperature' : list(AtmosphericScenarios_TS.loc[:,'T']),
                                'humidity'    : list(AtmosphericScenarios_TS.loc[:,'H']),
                                'rain'        : list(AtmosphericScenarios_TS.loc[:,'rain']),
                                'fog'         : list(AtmosphericScenarios_TS.loc[:,'fog']),
                                'time'        : list(AtmosphericScenarios_TS.loc[:,'t'])}#for rain and fog intensity intervals might be introduced [none,low, medium high]
        else:    
            Atmospheric_inputs={'temperature' : [.5],
                            'humidity'    : [5],
                            'rain'        : [True],
                            'fog'         : [False]}#for rain and fog intensity intervals might be introduced [none,low, medium high]
    # LIDAR layout inputs:
    class lidar_inp():
        Lidar_inputs = {'Wavelength' : [1522]}
    
    # Converter input characteristics
    class power_inp():
        Converter_uncertainty_inputs   = {'noise_conv'        : [4],
                                          'OtherChanges_conv' : [.005],
                                          'losses'   :[0.056]}
        PowerSource_uncertainty_inputs = {'noise_powersource'       : [.8],
                                          'OtherChanges_PowerSource': [.09]}
    
    
    # Photonics module uncertainty values:
    class photonics_inp():
        Amplifier_uncertainty_inputs      = {'noise_amp'        : [0.2],
                                             'OtherChanges_amp' : [.005],
                                             'NoiseFigure_FILE' : 'NoiseFigure.xlsx'}
        LaserSource_uncertainty_inputs    = {'noise_lasersource'       : [.8],
                                             'OtherChanges_lasersource': [.09]}
        Photodetector_uncertainty_inputs  = {'noise_photo'             : [0.2],
                                            'OtherChanges_photo'       : [.005],
                                            'Photodetector_Noise_FILE' :'Noise_Photodetector.xlsx'}
    class optics_inp():
        Telescope_uncertainty_inputs      = {'curvature_lens'    : [.01],
                                             'OtherChanges_tele' : [.006],
                                             'aberration'        : [.0004],
                                             'losses'            : [.09]}
        

    #%%PLotting Section:
    
    flag_plot_signal_noise=False



#%% Execute the UQ code:
#if flag_exec_LiUQ:
#    UQ=open('LiUQ_Hardware3.py') # Open the code
#    read_file=UQ.read()          # read the code
#    exec(read_file)              # execute the code
