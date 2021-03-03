# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:08:32 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c) 

###############################################################################
###################### __ Instantiate __ ######################################

Digitally creating a virtual object which represents a physical, real object
could be done by instantiating their python classes:
        
        1) Run Qlunc_Classes.py to create the classes corresponding to
           components, modules, atmospheric scenarios and lidar device
        2) Instantiate the different components
        3) Instantiate modules including corresponding components
        4) Instantiate class `atmosphere` --> atmospheric conditions
        5) Instantiate class `lidar_gral_inp` --> lidar general inputs
        6) Instantiate class `lidar` by including modules, lidar general inputs 
           and atmospheric conditions
           
"""

#%% Running Qlunc_Classes.py:

# Reading data from the yaml file:
from Qlunc_Classes import *
with open (r'./Qlunc_inputs.yml') as file: # WHere the yaml file is in order to get the input data
    Qlunc_yaml_inputs={}
    docs = yaml.load_all(file, Loader=yaml.FullLoader)
    for doc in docs:      
        for k, v in doc.items():           
            Qlunc_yaml_inputs.setdefault(k,v)  # save a dictionary with the data coming from yaml file 

exec(open(Qlunc_yaml_inputs['Main_directory']+'/Qlunc_Classes.py').read())   # Execute Qlunc_Classes.py (creating classes for lidar 'objects')
#%%%%%%%%%%%%%%%%% INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## FLAG inputs: ###############################################################
flags.flag_plot_pointing_accuracy_unc    = Qlunc_yaml_inputs['Flags']['Pointing accuracy uncertainty']   # Pointing accuracy uncertainty - Keep False
flags.flag_plot_measuring_points_pattern = Qlunc_yaml_inputs['Flags']['Scanning Pattern']  # Pattern of measuring points
flags.flag_plot_photodetector_noise      = Qlunc_yaml_inputs['Flags']['Photodetector noise']  # Photodetector noise: shot noise, dark current noise, thermal noise as a function of the photodetector input signal power.



## Optics components and Module: ##############################################

# Here we create optics components and optics module. User can create as many components as he/she want and combine them to create different module types
# Each module/component is a python object with their own technical characteristics and can be flexible combined to assess different use cases. 

# Scanner:

Scanner           = scanner(name            = Qlunc_yaml_inputs['Components']['Scanner']['Name'],           # Introduce your scanner name.
                            scanner_type    = Qlunc_yaml_inputs['Components']['Scanner']['Type'],
                            origin          = Qlunc_yaml_inputs['Components']['Scanner']['Origin'],         # Origin (coordinates of the lidar deployment).
                            sample_rate     = Qlunc_yaml_inputs['Components']['Scanner']['Sample rate'],    # for now introduce it in [degrees].
                           
                           # This values for focus distance, cone_angle and azimuth define a typical VAD scanning sequence:
                            focus_dist      = np.array(Qlunc_yaml_inputs['Components']['Scanner']['Focus distance']*int(Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][1]/Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][2])),   # Focus distance in [meters]                                        
                            cone_angle      = np.array(Qlunc_yaml_inputs['Components']['Scanner']['Cone angle']*int(Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][1]/Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][2])),    # Cone angle in [degrees].
                            azimuth         = np.array(np.arange(Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][0],
                                                                Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][1],
                                                                Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][2])),#np.arange(0,360,15), # Azimuth angle in [degrees].
                            x               = np.array(Qlunc_yaml_inputs['Components']['Scanner']['x']),
                            y               = np.array(Qlunc_yaml_inputs['Components']['Scanner']['y']),
                            z               = np.array(Qlunc_yaml_inputs['Components']['Scanner']['z']),
                            stdv_focus_dist = Qlunc_yaml_inputs['Components']['Scanner']['stdv focus distance'],                 # Focus distance standard deviation in [meters].
                            stdv_cone_angle = Qlunc_yaml_inputs['Components']['Scanner']['stdv Cone angle'],                 # Cone angle standard deviation in [degrees].
                            stdv_azimuth    = Qlunc_yaml_inputs['Components']['Scanner']['stdv Azimuth'],                 # Azimuth angle standard deviation in [degrees].
                            stdv_x          = Qlunc_yaml_inputs['Components']['Scanner']['stdv x'],
                            stdv_y          = Qlunc_yaml_inputs['Components']['Scanner']['stdv y'],
                            stdv_z          = Qlunc_yaml_inputs['Components']['Scanner']['stdv z'],
                            unc_func        = eval(Qlunc_yaml_inputs['Components']['Scanner']['Uncertainty function']) )    # here you put the function describing your scanner uncertainty. 

#Optical Circulator:

Optical_circulator = optical_circulator (name           = Qlunc_yaml_inputs['Components']['Optical Circulator']['Name'],       # Introduce your Optical circulator name.
                                         insertion_loss = Qlunc_yaml_inputs['Components']['Optical Circulator']['Insertion loss'],                        # In [dB]. Insertion loss parameters.
                                         unc_func       = eval(Qlunc_yaml_inputs['Components']['Optical Circulator']['Uncertainty function']))  # Function describing your scanner uncertainty.  Further informaion in "UQ_Optics_Classes.py" comments.


# Optics Module:

Optics_Module =  optics (name               = Qlunc_yaml_inputs['Modules']['Optics Module']['Name'],     # Introduce your Optics Module name.
                         scanner            = eval(Qlunc_yaml_inputs['Modules']['Optics Module']['Scanner']),             # Scanner instance (in this example "Scanner") or "None". "None" means that you don´t want to include Scanner in Optics Module, either in uncertainty calculations.
                         optical_circulator = eval(Qlunc_yaml_inputs['Modules']['Optics Module']['Optical circulator']),  # Optical Circulator instance (in this example "Optical_circulator") or "None". "None" means that you don´t want to include Optical circulator in Optics Module, either in uncertainty calculations.
                         laser              = eval(Qlunc_yaml_inputs['Modules']['Optics Module']['Laser']),
                         unc_func           = eval(Qlunc_yaml_inputs['Modules']['Optics Module']['Uncertainty function']))


## Photonics components and Module: ###########################################
# Here we create photonics components and photonics module. User can create as many components as he/she wants and combine them to create different module types.

Optical_Amplifier = optical_amplifier(name    = Qlunc_yaml_inputs['Components']['Optical Amplifier']['Name'],        # Introduce your scanner name.
                                      OA_NF    = Qlunc_yaml_inputs['Components']['Optical Amplifier']['Optical amplifier noise figure'],          # In [dB]. Can introduce it as a table from manufactures (in this example the data is taken from Thorlabs.com, in section EDFA\Graps) or introduce a single well-known value
                                      OA_Gain  = Qlunc_yaml_inputs['Components']['Optical Amplifier']['Optical amplifier gain'],                         # In [dB]. (in this example the data is taken from Thorlabs.com, in section EDFA\Specs)
                                      unc_func = eval(Qlunc_yaml_inputs['Components']['Optical Amplifier']['Uncertainty function']))  # Function describing Optical Amplifier uncertainty. Further informaion in "UQ_Photonics_Classes.py" comments.

Photodetector    = photodetector(name             = Qlunc_yaml_inputs['Components']['Photodetector']['Name'],               # Introduce your photodetector name.
                                 Photo_BandWidth  = Qlunc_yaml_inputs['Components']['Photodetector']['Photodetector BandWidth'],                  # In[]. Photodetector bandwidth
                                 Load_Resistor    = Qlunc_yaml_inputs['Components']['Photodetector']['Load resistor'],                     # In [ohms]
                                 Photo_efficiency = Qlunc_yaml_inputs['Components']['Photodetector']['Photodetector efficiency'],                    # Photodetector efficiency [-]
                                 Dark_Current     = Qlunc_yaml_inputs['Components']['Photodetector']['Dark current'],                   #  In [A]. Dark current in the photodetector.
                                 Photo_SignalP    = Qlunc_yaml_inputs['Components']['Photodetector']['Photodetector signalP'],
                                 Power_interval   = np.array(np.arange(Qlunc_yaml_inputs['Components']['Photodetector']['Power interval'][0],
                                                                       Qlunc_yaml_inputs['Components']['Photodetector']['Power interval'][1],
                                                                       Qlunc_yaml_inputs['Components']['Photodetector']['Power interval'][2])),#np.arange(Qlunc_yaml_inputs['Components']['Photodetector']['Power interval']), # In [w]. Power interval for the photodetector domain in photodetector SNR plot. 
                                 Gain_TIA         = Qlunc_yaml_inputs['Components']['Photodetector']['Gain TIA'],                    # In [dB]. If there is a transimpedance amplifier.
                                 V_Noise_TIA      = Qlunc_yaml_inputs['Components']['Photodetector']['V Noise TIA'],                 # In [V]. If there is a transimpedance amplifier.
                                 unc_func         = eval(Qlunc_yaml_inputs['Components']['Photodetector']['Uncertainty function']))  # Function describing Photodetector uncertainty. Further informaion in "UQ_Photonics_Classes.py" comments.

# Module:

Photonics_Module = photonics(name              = Qlunc_yaml_inputs['Modules']['Photonics Module']['Name'],        # Introduce your Photonics module name
                             photodetector     = eval(Qlunc_yaml_inputs['Modules']['Photonics Module']['Photodetector']),             # Photodetector instance (in this example "Photodetector") or "None". "None" means that you don´t want to include photodetector in Photonics Module, either in uncertainty calculations.
                             optical_amplifier = eval(Qlunc_yaml_inputs['Modules']['Photonics Module']['Optical amplifier']),         # Scanner instance (in this example "OpticalAmplifier") or "None". "None" means that you don´t want to include Optical Amplifier in Photonics Module, either in uncertainty calculations.
                             unc_func          = eval(Qlunc_yaml_inputs['Modules']['Photonics Module']['Uncertainty function']))

## Lidar general inputs: ######################################################
Lidar_inputs     = lidar_gral_inp(name        = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Name'],      # Introduce the name of your lidar data folder.
                                  wave        = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Wavelength'],                    # In [m]. Lidar wavelength.
                                  yaw_error   = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Yaw error'],                          # In [°]. Degrees of rotation around z axis because of inclinometer errors
                                  pitch_error = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Pitch error'],                          # In [°]. Degrees of rotation around y axis
                                  roll_error  = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Roll error'])                          # In [°]. Degrees of rotation around z axis.


## Lidar device:

Lidar = lidar(name         = Qlunc_yaml_inputs['Lidar']['Name'],                       # Introduce the name of your lidar device.
              photonics    = eval(Qlunc_yaml_inputs['Lidar']['Photonics module']),     # Introduce the name of your photonics module.
              optics       = eval(Qlunc_yaml_inputs['Lidar']['Optics module']),        # Introduce the name of your optics module.
              power        = eval(Qlunc_yaml_inputs['Lidar']['Power module']),         # Introduce the name of your power module. NOT IMPLEMENTED YET!
              lidar_inputs = eval(Qlunc_yaml_inputs['Lidar']['Lidar inputs']),         # Introduce lidar general inputs
              unc_func     = eval(Qlunc_yaml_inputs['Lidar']['Uncertainty function'])) # Function estimating lidar global uncertainty


## Creating atmospheric scenarios: ############################################

Atmospheric_TimeSeries = Qlunc_yaml_inputs['Atmospheric_inputs']['TimeSeries'] # This defines whether we are using a time series (True) or single values (False) to describe the atmosphere (T, H, rain and fog) 
                                                                           # If so we obtain a time series describing the noise implemented in the measurement.
if Atmospheric_TimeSeries:
    Atmos_TS_FILE           = '../metadata/AtmosphericData/'+Qlunc_yaml_inputs['Atmospheric_inputs']['Atmos_TS_FILE']
    AtmosphericScenarios_TS = pd.read_csv(Atmos_TS_FILE,delimiter=';',decimal=',')
    Atmospheric_inputs = {
                          'temperature' : list(AtmosphericScenarios_TS.loc[:,'T']),    # [K]
                          'humidity'    : list(AtmosphericScenarios_TS.loc[:,'H']),    # [%]
                          'rain'        : list(AtmosphericScenarios_TS.loc[:,'rain']),
                          'fog'         : list(AtmosphericScenarios_TS.loc[:,'fog']),
                          'time'        : list(AtmosphericScenarios_TS.loc[:,'t'])     #for rain and fog intensity intervals might be introduced [none,low, medium high]
                          } 
    Atmospheric_Scenario = atmosphere(name        = 'Atmosphere1',
                                      temperature = Atmospheric_inputs['temperature'])
else:    

    Atmospheric_Scenario = atmosphere(name        = 'Atmosphere1',
                                      temperature = Qlunc_yaml_inputs['Atmospheric_inputs']['Temperature'])
