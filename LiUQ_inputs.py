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

# Which modules introduce incertainties?:
#flag_unc_amplifier     = True # if True I include amplifier uncertainty
#flag_unc_photodetector = True # if True I include photodetector uncertainty
#flag_unc_telescope     = True # if True I include telescope uncertainty
# Want to execute LiUQ?
#flag_exec_LiUQ         = True

# Modules and methods we want to assess:
modules = ['amplifier','telescope','photodetector'] # modules we want to assess uncertainty
DP      = [] # data processing methods we want to assess
 
# Atmospheric inputs:
Atmospheric_inputs={'temperature':[9],'humidity':[4,5],'rain':[True],'fog':[False]}#for rain and fog intensity intervals might be introduced [none,low, medium high]

# Amplifier module uncertainty values:
Amplifier_uncertainty_inputs={'noise_amp':[0.2],'OtherChanges_amp':[.005]}

# Photodetector module uncertainty values:
Photodetector_uncertainty_inputs={'noise_photo':[0.2],'OtherChanges_photo':[.005]}

# Telescope module uncertainty values:
Telescope_uncertainty_inputs={'curvature_lens':[.01],'OtherChanges_tele':[.006],'aberration':[.0004]}
#%%PLotting Section:

flag_plot_signal_noise=False



#%% Execute the UQ code:
#if flag_exec_LiUQ:
#    UQ=open('LiUQ_Hardware3.py') # Open the code
#    read_file=UQ.read()          # read the code
#    exec(read_file)              # execute the code
