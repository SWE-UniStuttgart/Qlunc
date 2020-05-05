# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:14:00 2020

@author: fcosta
"""
#%% Header:
#04272020 - Francisco Costa
#SWE - Stuttgart
#LiUQ inputs for different modules:
#Here the user can introduce parameters to calculate the uncertainty of the different modules

# Inputs:
#    Modules can be: 'amplifier', 'telescope', 'photodetector'
#    Data processing methods can be: 'LOS', 'filtering methods'
#    Temperature= [-15,0,15] for example
#    Humidity = [0,25,50,75]% for example
#    rain        = False
#    fog         = False  #for rain and fog intensity intervals might be introduced [none,low, medium high]
#    Values of uncertainties will be introduced in dB (for now)
#%% Inputs:

# Which modules introduce incertainties?:
flag_unc_amplifier     = True # if True I include amplifier uncertainty
flag_unc_photodetector = True # if True I include photodetector uncertainty
flag_unc_telescope     = True # if True I include telescope uncertainty
# Want to execute LiUQ?
flag_exec_LiUQ         = True

# Modules and methods we want to assess:
modules = ['amplifier','telescope','photodetector'] # modules we want to assess uncertainty
DP      = [] # data processing methods we want to assess

# Atmospheric scenario (this will depend on the specific chosen scenarios):
temperature = 25 # in °C (this could be an interval)
humidity    = 20 # in %  (this could be an interval)
rain        = False
fog         = False  #for rain and fog intensity intervals might be introduced [none,low, medium high]

#%% Amplifier module uncertainty values:

o_c_amp   = 0.005  # other changes in dB
noise_amp = 5 # Noise figure in dB. Given by the manufacture.
Amplifier={'noise_amp':noise_amp,'OtherChanges_amp':o_c_amp}

#%% Photodetector module values
noise_photo = 0.01 # in dB. Given by manufacture
o_c_photo   = 0.5 # other changes

#%%Telescope
curvature_lens = 0.01 # in meters
o_c_tele       = 0.006   # other changes
aberration     = 0.0004


parameters=[temperature,humidity,rain,fog,o_c_amp,noise_amp,noise_photo,o_c_photo,curvature_lens,o_c_tele,aberration]
#%%PLotting Section:

flag_plot_signal_noise=False



#%% Execute the UQ code:
#if flag_exec_LiUQ:
#    UQ=open('LiUQ_Hardware3.py') # Open the code
#    read_file=UQ.read()          # read the code
#    exec(read_file)              # execute the code
