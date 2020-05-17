# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:14:00 2020

@author: fcosta
"""
#%% Header:
#04272020 - Francisco Costa
#SWE - Stuttgart
#LiUQ inputs for different modules:
#Here the user can introduce parameters to calculate the uncertainty of the different modules. Must be note units of the uncertainties: either dB or watts

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

# Which modules introduce incertalistinties?:
#flag_unc_amplifier     = True # if True I include amplifier uncertainty
#flag_unc_photodetector = True # if True I include photodetector uncertainty
#flag_unc_telescope     = True # if True I include telescope uncertainty
# Want to execute LiUQ?
#flag_exec_LiUQ         = True
flag_plot_signal_noise  = True
# Modules and methods we want to assess:
class inputs():
    directory='../GitHub_LiUQ/'
    
    # Modules is a dictionary containing the lidar modules as a key and  components (or functions) as values
    modules = {
               'power'     : {'power_source' :[],
                              'converter'    :[]},
    
               'photonics' : {'photodetector':[], 
                              'amplifier'    :['amplifier_fignoise'],
                              'laser_source' :[]},
                              
               'optics'    : {'telescope'    :[]}
               }
    DP      = ['los'] # data processing methods we want to assess

    # Which uncertainties want to include in calculations when calling methods:
    # Ampli ('ampli_atm_unc'       includes uncertainties related with atmosphere conditions
    #        'ampli_figure_noise'  includes figure noise  )
    ampliU=['ampli_atm_unc','ampli_figure_noise']
    
     
    # Atmospheric inputs:
    class atm_inp():
        Atmospheric_inputs={'temperature' : [.5,10000],
                            'humidity'    : [5],
                            'rain'        : [True],
                            'fog'         : [False]}#for rain and fog intensity intervals might be introduced [none,low, medium high]
    # LIDAR
    class lidar_inp():
        Lidar_inputs = {'Wavelength' : [1522]}
    
    # Converter
    class power_inp():
        Converter_uncertainty_inputs   = {'noise_conv'        : [0.2],
                                          'OtherChanges_conv' : [.005]}
        PowerSource_uncertainty_inputs = {'noise_powersource'       : [.8],
                                          'OtherChanges_PowerSource': [.09]}
    
    
    # Photonics module uncertainty values:
    class photonics_inp():
        Amplifier_uncertainty_inputs      = {'noise_amp'        : [0.2],
                                             'OtherChanges_amp' : [.005],
                                             'NoiseFigure_FILE' : 'NoiseFigure.xlsx'}
        LaserSource_uncertainty_inputs    = {'noise_lasersource'       : [.8],
                                             'OtherChanges_LaserSource': [.09]}
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
