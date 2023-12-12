# -*- coding: utf-8 -*-
""".

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
        
        
        By running this script...
                        
        2) Instantiate the different components
        3) Instantiate modules including corresponding components
        4) Instantiate class `atmosphere` --> atmospheric conditions
        5) Instantiate class `lidar_gral_inp` --> lidar general inputs
        6) Instantiate class `lidar` by including modules, lidar general inputs 
           and atmospheric conditions

        This is done in a more human-readable way through the yaml template 'Qlunc_inputs.yml' provided as an example in the repository, in the 'Main' folder
"""

#Clear all variables
# from IPython import get_ipython
# get_ipython().magic('reset -sf')

import os
import pdb
import numpy as np
import yaml
# pdb.set_trace()

# Changeing to Qlunc path
# os.chdir(os.path.normpath(os.path.join(os.path.dirname(__file__),"..\\")))
# os.chdir('C:\\SWE_LOCAL\\Qlunc')
# from Utils.Qlunc_ImportModules import *

# import  Qlunc.UQ_Functions.UQ_Photonics_Classes as uphc, Qlunc.UQ_Functions.UQ_Optics_Classes as uopc, Qlunc.UQ_Functions.UQ_Lidar_Classes as ulc,  Qlunc.UQ_Functions.UQ_ProbeVolume_Classes as upbc,  Qlunc.UQ_Functions.UQ_SignalProcessor_Classes as uspc
# os.chdir(os.path.normpath(os.path.join(os.path.dirname(__file__),"..\\")))
# importing  uncertainty functions
# import  UQ_Functions.UQ_Photonics_Classes as uphc,UQ_Functions.UQ_Optics_Classes as uopc,UQ_Functions.UQ_Lidar_Classes as ulc, UQ_Functions.UQ_ProbeVolume_Classes as upbc, UQ_Functions.UQ_SignalProcessor_Classes as uspc


#%% Running Qlunc_Classes.py:
with open ('./Main/Qlunc_inputs.yml') as file: # WHere the yaml file is in order to get the input data
    Qlunc_yaml_inputs={}
    docs = yaml.load_all(file, Loader=yaml.FullLoader)
    for doc in docs:      
        for k, v in doc.items():           
            Qlunc_yaml_inputs.setdefault(k,v)  # save a dictionary with the data coming from yaml file 

os.chdir(Qlunc_yaml_inputs['Main directory'])
import  UQ_Functions.UQ_Photonics_Classes as uphc, UQ_Functions.UQ_Optics_Classes as uopc, UQ_Functions.UQ_Lidar_Classes as ulc,  UQ_Functions.UQ_ProbeVolume_Classes as upbc,  UQ_Functions.UQ_SignalProcessor_Classes as uspc


# Execute Qlunc_Classes.py (creating classes for lidar 'objects')
exec(open('.\\Main\\Qlunc_Classes.py').read()) 

#%%%%%%%%%%%%%%%%% INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%# Lidar general inputs: ######################################################
Lidar_inputs     = lidar_gral_inp(name        = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Name'],          # Introduce the name of your lidar data folder.
                                  ltype       = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Type'],
                                  yaw_error   = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Yaw error'],     # In [°]. Degrees of rotation around z axis because of inclinometer errors
                                  pitch_error = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Pitch error'],   # In [°]. Degrees of rotation around y axis
                                  roll_error  = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Roll error'],    # In [°]. Degrees of rotation around z axis.
                                  dataframe   = { })  # Final dataframe



#%%# Photonics components and Module: ###########################################
# Here we create photonics components and photonics module. Users can create as many components as they want and combine them to create different module types.

Photodetector    = photodetector(name             = Qlunc_yaml_inputs['Components']['Photodetector']['Name'],               # Introduce your photodetector name.
                                 Photo_BandWidth  = Qlunc_yaml_inputs['Components']['Photodetector']['Photodetector BandWidth'],                  # In[]. Photodetector bandwidth
                                 Load_Resistor    = Qlunc_yaml_inputs['Components']['Photodetector']['Load resistor'],                     # In [ohms]
                                 Photo_efficiency = Qlunc_yaml_inputs['Components']['Photodetector']['Photodetector efficiency'],                    # Photodetector efficiency [-]
                                 Dark_Current     = Qlunc_yaml_inputs['Components']['Photodetector']['Dark current'],                   #  In [A]. Dark current in the photodetector.
                                 Photo_SignalP    = Qlunc_yaml_inputs['Components']['Photodetector']['Photodetector signalP'],
                                 Power_interval   = np.array(np.arange(Qlunc_yaml_inputs['Components']['Photodetector']['Power interval'][0],
                                                                       Qlunc_yaml_inputs['Components']['Photodetector']['Power interval'][1],
                                                                       Qlunc_yaml_inputs['Components']['Photodetector']['Power interval'][2])),#np.arange(Qlunc_yaml_inputs['Components']['Photodetector']['Power interval']), # In [w]. Power interval for the photodetector domain in photodetector SNR plot. 
                                 Active_Surf      = Qlunc_yaml_inputs['Components']['Photodetector']['Active area'],
                                 Gain_TIA         = Qlunc_yaml_inputs['Components']['Photodetector']['Gain TIA'],                    # In [dB]. If there is a transimpedance amplifier.
                                 V_Noise_TIA      = Qlunc_yaml_inputs['Components']['Photodetector']['V Noise TIA'],                 # In [V]. If there is a transimpedance amplifier.
                                 
                                 unc_func         = uphc.UQ_Photodetector) #eval(Qlunc_yaml_inputs['Components']['Photodetector']['Uncertainty function']))  # Function describing Photodetector uncertainty. Further informaion in "UQ_Photonics_Classes.py" comments.



# Photonics Module:
Photonics_Module = photonics(name                    = Qlunc_yaml_inputs['Modules']['Photonics Module']['Name'],        # Introduce your Photonics module name
                             photodetector           = eval(Qlunc_yaml_inputs['Modules']['Photonics Module']['Photodetector']),             # Photodetector instance (in this example "Photodetector") or "None". "None" means that you don´t want to include photodetector in Photonics Module, either in uncertainty calculations.
                             unc_func                = uphc.sum_unc_photonics) #eval(Qlunc_yaml_inputs['Modules']['Photonics Module']['Uncertainty function']))





#%%# Optics components and Module: ##############################################

# Here we create optics components and optics module. User can create as many components as he/she want and combine them to create different module types
# Each module/component is a python object with their own technical characteristics and can be flexible combined to assess different use cases. 

# Scanner:
Scanner           = scanner(name            = Qlunc_yaml_inputs['Components']['Scanner']['Name'],           # Introduce your scanner name.
                            origin          = Qlunc_yaml_inputs['Components']['Scanner']['Origin'],         # Origin (coordinates of the lidar deployment).
                            N_MC            = Qlunc_yaml_inputs['Components']['Scanner']['N_MC'],
                            pattern         = Qlunc_yaml_inputs['Components']['Scanner']['Pattern'],
                            lissajous_param = Qlunc_yaml_inputs['Components']['Scanner']['Lissajous parameters'],
                            vert_plane      = Qlunc_yaml_inputs['Components']['Scanner']['Vertical plane parameters'],
                            hor_plane      = Qlunc_yaml_inputs['Components']['Scanner']['Horizontal plane parameters'],
                            Href            = Qlunc_yaml_inputs['Components']['Scanner']['Href'],                    
                            azimuth         = Qlunc_yaml_inputs['Components']['Scanner']['Psi'],   # Azimuth in [degrees]
                            focus_dist      = Qlunc_yaml_inputs['Components']['Scanner']['Rho'],   # Focus distance in [meters]                                                                                              
                            cone_angle      = Qlunc_yaml_inputs['Components']['Scanner']['Theta'], # Elevation angle in [degrees]
                            stdv_location   = Qlunc_yaml_inputs['Components']['Scanner']['Error origin'],
                            stdv_focus_dist = Qlunc_yaml_inputs['Components']['Scanner']['stdv focus distance'],                 # Focus distance standard deviation in [meters].
                            stdv_cone_angle = Qlunc_yaml_inputs['Components']['Scanner']['stdv Elevation angle'],                 # Elevation angle standard deviation in [degrees].
                            stdv_azimuth    = Qlunc_yaml_inputs['Components']['Scanner']['stdv Azimuth'],                 # Azimuth angle standard deviation in [degrees].
                            stdv_Estimation = Qlunc_yaml_inputs['Components']['Scanner']['stdv Estimation'],
                            correlations    = Qlunc_yaml_inputs['Components']['Scanner']['correlations'],
                            unc_func        = uopc.UQ_Scanner) #eval(Qlunc_yaml_inputs['Components']['Scanner']['Uncertainty function']) )    # here you put the function describing your scanner uncertainty. 

    
    


# Optics Module:
Optics_Module =  optics (name               = Qlunc_yaml_inputs['Modules']['Optics Module']['Name'],     # Introduce your Optics Module name.
                         scanner            = eval(Qlunc_yaml_inputs['Modules']['Optics Module']['Scanner']),             # Scanner instance (in this example "Scanner") or "None". "None" means that you don´t want to include Scanner in Optics Module, either in uncertainty calculations.
                         unc_func           = uopc.sum_unc_optics) #eval(Qlunc_yaml_inputs['Modules']['Optics Module']['Uncertainty function']))





#%% Data processing methods. Signal processor components and module: ###########################################################

# Analog to digital converter
ADC = analog2digital_converter (name          = Qlunc_yaml_inputs['Components']['ADC']['Name'],
                                nbits         = Qlunc_yaml_inputs['Components']['ADC']['Number of bits'],
                                vref          = Qlunc_yaml_inputs['Components']['ADC']['Reference voltage'],
                                vground       = Qlunc_yaml_inputs['Components']['ADC']['Ground voltage'],
                                fs            = Qlunc_yaml_inputs['Components']['ADC']['Sampling frequency'],
                                u_fs          = Qlunc_yaml_inputs['Components']['ADC']['Uncertainty sampling freq'],
                                u_speckle     = Qlunc_yaml_inputs['Components']['ADC']['Speckle noise'],
                                q_error       = Qlunc_yaml_inputs['Components']['ADC']['Quantization error'],
                                ADC_bandwidth = Qlunc_yaml_inputs['Components']['ADC']['ADC Bandwidth'],
                                unc_func      = uspc.UQ_ADC)
# Signal processor module
Signal_processor_Module = signal_processor(name                     = Qlunc_yaml_inputs['Modules']['Signal processor Module']['Name'],
                                           analog2digital_converter = eval(Qlunc_yaml_inputs['Modules']['Signal processor Module']['ADC']),)
                                           # f_analyser             = Qlunc_yaml_inputs['Modules']['Signal processor Module']['Frequency analyser'],
                                           # unc_func                 = uspc.sum_unc_signal_processor)




#%% LIDAR Module

Lidar = lidar(name             = Qlunc_yaml_inputs['Lidar']['Name'],                       # Introduce the name of your lidar device.
              photonics        = eval(Qlunc_yaml_inputs['Lidar']['Photonics module']),#Photonics_Module, #     # Introduce the name of your photonics module.
              optics           = eval(Qlunc_yaml_inputs['Lidar']['Optics module']), # Optics_Module, #      # Introduce the name of your optics module.
              signal_processor = eval(Qlunc_yaml_inputs['Lidar']['Signal processor']),#None, #Signal_processor_Module,
              lidar_inputs     = eval(Qlunc_yaml_inputs['Lidar']['Lidar inputs']), #  Lidar_inputs, #      # Introduce lidar general inputs
              unc_func         = ulc.sum_unc_lidar ) 



#%% Creating atmospheric scenarios: ############################################
Atmospheric_TimeSeries = Qlunc_yaml_inputs['Atmospheric_inputs']['TimeSeries'] # This defines whether we are using a time series (True) or single values (False) to describe the atmosphere (T, H, rain and fog) 
                                                                           # If so we obtain a time series describing the noise implemented in the measurement.
if Atmospheric_TimeSeries:
    Atmos_TS_FILE           = './metadata/AtmosphericData/'+Qlunc_yaml_inputs['Atmospheric_inputs']['Atmos_TS_FILE']
    AtmosphericScenarios_TS = pd.read_csv(Atmos_TS_FILE,delimiter=';',decimal=',')
    Atmospheric_inputs = {
                          'temperature' : list(AtmosphericScenarios_TS.loc[:,'T']),    # [K]
                          'humidity'    : list(AtmosphericScenarios_TS.loc[:,'H']),    # [%]
                          'rain'        : list(AtmosphericScenarios_TS.loc[:,'rain']),
                          'fog'         : list(AtmosphericScenarios_TS.loc[:,'fog']),
                          'time'        : list(AtmosphericScenarios_TS.loc[:,'t'])     #for rain and fog intensity intervals might be introduced [none,low, medium high]
                          } 
    Atmospheric_Scenario = atmosphere(name           = Qlunc_yaml_inputs['Atmospheric_inputs']['Name'],
                                      temperature    = Atmospheric_inputs['temperature'],
                                      PL_exp         = Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'],
                                      Vref           = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref'],
                                      wind_direction = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'],
                                      wind_tilt      = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind tilt'],
                                      Hg             = Qlunc_yaml_inputs['Atmospheric_inputs']['Height ground'])

else:    

    Atmospheric_Scenario = atmosphere(name           = Qlunc_yaml_inputs['Atmospheric_inputs']['Name'],
                                      temperature    = Qlunc_yaml_inputs['Atmospheric_inputs']['Temperature'],
                                      PL_exp         = Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'],
                                      Vref           = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref'],
                                      wind_direction = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'],
                                      wind_tilt     = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind tilt'],
                                      Hg             = Qlunc_yaml_inputs['Atmospheric_inputs']['Height ground'])

