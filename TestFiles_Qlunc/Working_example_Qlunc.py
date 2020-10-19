# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:08:32 2020

@author: fcosta
"""

'''Qlunc working example:
    I this example is showned how Qlunc is working:
        
        1) Run Qlunc_Classes.py to create the classes corresponding to components, modules, atmospheric scenarios and lidar device.
        2) Create the Components instances introducing parameter values for each component.
        3) Modules instances are created and components are included in the the modules.
        4) Atmospheric scenarios ara included in the data: We can create it either from a single value or or from a time series
        5) Lidar device general inputs instance is created
        6) Lidar device instance is created and modules, containing the different components are included in the lidar architecture '''
    
#%% Running Qlunc_Classes.py:
exec(open('Qlunc_Classes.py').read())


#%% Optics components and Module: #############################################

# Here we create optics components and optics module. User can create as many components as he/she want and combine them to create different module types
# Each module/component is a python object with their own technical characteristics and can be flexible combined to assess different use cases. 

# Scanner:

Scanner           = scanner(name           = 'Scan1',             # Introduce your scanner name.
                           origin          = [0,0,0],             # Origin (coordinates of the lidar deployment).
                           focus_dist      = np.array([40]*24),   # Focus distance in [meters].
                           sample_rate     = 15,                  # for now introduce it in [degrees].
                           theta           = np.array([7]*24),    # Cone angle in [degrees].
                           phi             = np.arange(0,360,15), # Azimuth angle in [degrees].
                           stdv_focus_dist = 0.8,                 # Focus distance standard deviation in [meters].
                           stdv_theta      = 0.8,                 # Cone angle standard deviation in [degrees].
                           stdv_phi        = 0.8,                 # Azimuth angle standard deviation in [degrees].
                           unc_func        = uopc.UQ_Scanner)     # here you put the function describing your scanner uncertainty. 

#Optical Circulator:

Optical_circulator = optical_circulator (name = 'OptCirc1',                      # Introduce your Optical circulator name.
                                          insertion_loss = 2.1,                  # In [dB]. Insertion loss parameters.
                                          unc_func = uopc.UQ_OpticalCirculator)  # Function describing your scanner uncertainty.  Further informaion in "UQ_Optics_Classes.py" comments.


# Optics Module:

Optics_Module =  optics (name     = 'OptMod1',                     # Introduce your Optics Module name.
                         scanner  = Scanner,                       # Scanner instance (in this example "Scanner") or "None". "None" means that you don´t want to include Scanner in Optics Module, either in uncertainty calculations.
                         optical_circulator = Optical_circulator,  # Optical Circulator instance (in this example "Optical_circulator") or "None". "None" means that you don´t want to include Optical circulator in Optics Module, either in uncertainty calculations.
                         unc_func = uopc.sum_unc_optics)


### Photonics components and Module: ##########################################

# Here we create photonics components and photonics module. User can create as many components as he/she want and combine them to create different module types.

OpticalAmplifier = optical_amplifier(name     = 'OA1',                      # Introduce your scanner name.
                                     OA_NF    = 'NoiseFigure.csv',          # In [dB]. Can introduce it as a table from manufactures (in this example the data is taken from Thorlabs.com, in section EDFA\Graps) or introduce a single well-known value
                                     OA_Gain  = 30,                         # In [dB]. (in this example the data is taken from Thorlabs.com, in section EDFA\Specs)
                                     unc_func = uphc.UQ_Optical_amplifier)  # Function describing Optical Amplifier uncertainty. Further informaion in "UQ_Photonics_Classes.py" comments.

Photodetector    = photodetector(name             = 'Photo1',               # Introduce your photodetector name.
                                 Photo_BandWidth  = 380e6,                  # In[]. Photodetector bandwidth
                                 Load_Resistor    = 50,                     # In [ohms]
                                 Photo_efficiency = .85,                    # Photodetector efficiency [-]
                                 Dark_Current     = 5e-9,                   #  In [A]. Dark current in the photodetector.
                                 Photo_SignalP    = 1e-3,
                                 Power_interval   = np.arange(0,1000,.001), # In [w]. Power interval for the photodetector domain in photodetector SNR plot. 
                                 Gain_TIA         = 5e3,                    # In [dB]. If there is a transimpedance amplifier.
                                 V_Noise_TIA      = 160e-6,                 # In [V]. If there is a transimpedance amplifier.
                                 unc_func         = uphc.UQ_Photodetector)  # Function describing Photodetector uncertainty. Further informaion in "UQ_Photonics_Classes.py" comments.

# Module:

Photonics_Module = photonics(name              = 'PhotoMod1',               # Introduce your Photonics module name
                             photodetector     = Photodetector,             # Photodetector instance (in this example "Photodetector") or "None". "None" means that you don´t want to include photodetector in Photonics Module, either in uncertainty calculations.
                             optical_amplifier =  OpticalAmplifier,         # Scanner instance (in this example "OpticalAmplifier") or "None". "None" means that you don´t want to include Optical Amplifier in Photonics Module, either in uncertainty calculations.
                             unc_func          = uphc.sum_unc_photonics)

### Lidar general inputs ######################################################
Lidar_inputs     = lidar_gral_inp(name        = 'Gral_inp1',     # Introduce the name of your lidar data folder.
                                  wave        = 1550e-9,         # In [m]. Lidar wavelength.
                                  sample_rate = 2,               # In [Hz]
                                  yaw_error   = 0,               # In [°]. Degrees of rotation around z axis because of inclinometer errors
                                  pitch_error = 0,               # In [°]. Degrees of rotation around y axis
                                  roll_error  = 0)               # In [°]. Degrees of rotation around z axis.


## Lidar device:

Lidar1 = lidar(name         = 'Caixa1',             # Introduce the name of your lidar device.
               photonics    = Photonics_Module,     # Introduce the name of your photonics module.
               optics       = Optics_Module1,       # Introduce the name of your optics module.
               power        = None,                 # Introduce the name of your power module. NOT IMPLEMENTED YET!
               lidar_inputs = Lidar_inputs,         # Introduce lidar general inputs
               unc_func     = ulc.sum_unc_lidar)    # Function estimating lidar global uncertainty