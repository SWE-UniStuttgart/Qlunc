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
from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
import pdb

import yaml
import numpy as np

# pdb.set_trace()

# Changeing to Qlunc path
# os.chdir(os.path.normpath(os.path.join(os.path.dirname(__file__),"..\\")))
# os.chdir('C:\\SWE_LOCAL\\Qlunc')
# from Utils.Qlunc_ImportModules import *

# import  Qlunc.UQ_Functions.UQ_Photonics_Classes as uphc, Qlunc.UQ_Functions.UQ_Optics_Classes as uopc, Qlunc.UQ_Functions.UQ_Lidar_Classes as ulc,  Qlunc.UQ_Functions.UQ_ProbeVolume_Classes as upbc,  Qlunc.UQ_Functions.UQ_SignalProcessor_Classes as uspc
os.chdir(os.path.normpath(os.path.join(os.path.dirname(__file__),"..\\")))
# importing  uncertainty functions
# import  UQ_Functions.UQ_Photonics_Classes as uphc,UQ_Functions.UQ_Optics_Classes as uopc,UQ_Functions.UQ_Lidar_Classes as ulc, UQ_Functions.UQ_ProbeVolume_Classes as upbc, UQ_Functions.UQ_SignalProcessor_Classes as uspc


# from Utils.Qlunc_ImportModules import *


#%% Running Qlunc_Classes.py:
try:
    with open ('.\\Qlunc_inputs.yml') as file: # WHere the yaml file is in order to get the input data
        Qlunc_yaml_inputs={}
        docs = yaml.load_all(file, Loader=yaml.FullLoader)
        for doc in docs:      
            for k, v in doc.items():           
                Qlunc_yaml_inputs.setdefault(k,v)  # save a dictionary with the data coming from yaml file 
    
    # pdb.set_trace()
    import  UQ_Functions.UQ_Photonics_Classes as uphc,UQ_Functions.UQ_Optics_Classes as uopc,UQ_Functions.UQ_Lidar_Classes as ulc, UQ_Functions.UQ_ProbeVolume_Classes as upbc, UQ_Functions.UQ_SignalProcessor_Classes as uspc
    from Utils.Qlunc_ImportModules import *
    
    
    
    # Execute Qlunc_Classes.py (creating classes for lidar 'objects')
    exec(open('.\\Qlunc_Classes.py').read()) 
except:
    with open ('.\\Main\\Qlunc_inputs.yml') as file: # WHere the yaml file is in order to get the input data
        Qlunc_yaml_inputs={}
        docs = yaml.load_all(file, Loader=yaml.FullLoader)
        for doc in docs:      
            for k, v in doc.items():           
                Qlunc_yaml_inputs.setdefault(k,v)  # save a dictionary with the data coming from yaml file 

    # pdb.set_trace()
    import  UQ_Functions.UQ_Photonics_Classes as uphc,UQ_Functions.UQ_Optics_Classes as uopc,UQ_Functions.UQ_Lidar_Classes as ulc, UQ_Functions.UQ_SignalProcessor_Classes as uspc
    from Utils.Qlunc_ImportModules import *



    # Execute Qlunc_Classes.py (creating classes for lidar 'objects')
    exec(open('.\\Main\\Qlunc_Classes.py').read()) 

#%%%%%%%%%%%%%%%%% INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%# Lidar general inputs: ######################################################
# Lidar_inputs     = lidar_gral_inp(name        = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Name'],          # Introduce the name of your lidar data folder.
#                                   ltype       = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Type'],
#                                   # yaw_error   = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Yaw error'],     # In [°]. Degrees of rotation around z axis because of inclinometer errors
#                                   # pitch_error = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Pitch error'],   # In [°]. Degrees of rotation around y axis
#                                   # roll_error  = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Roll error'],    # In [°]. Degrees of rotation around z axis.
#                                   dataframe   = { })  # Final dataframe



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
                            hor_plane       = Qlunc_yaml_inputs['Components']['Scanner']['Horizontal plane parameters'],
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
                                           analog2digital_converter = eval(Qlunc_yaml_inputs['Modules']['Signal processor Module']['ADC']),
                                           # f_analyser             = Qlunc_yaml_inputs['Modules']['Signal processor Module']['Frequency analyser'],
                                            unc_func                = uspc.sum_unc_signal_processor)




#%% LIDAR Module

Lidar = lidar(name             = Qlunc_yaml_inputs['Lidar']['Name'],                       # Introduce the name of your lidar device.
              photonics        = eval(Qlunc_yaml_inputs['Lidar']['Photonics module']),#Photonics_Module, #     # Introduce the name of your photonics module.
              optics           = eval(Qlunc_yaml_inputs['Lidar']['Optics module']), # Optics_Module, #      # Introduce the name of your optics module.
              signal_processor = eval(Qlunc_yaml_inputs['Lidar']['Signal processor']),#None, #Signal_processor_Module,
              # lidar_inputs     = eval(Qlunc_yaml_inputs['Lidar']['Lidar inputs']), #  Lidar_inputs, #      # Introduce lidar general inputs
              unc_func         = ulc.sum_unc_lidar ) 



#%% Creating atmospheric scenarios: ############################################
# Atmospheric_TimeSeries = Qlunc_yaml_inputs['Atmospheric_inputs']['TimeSeries'] # This defines whether we are using a time series (True) or single values (False) to describe the atmosphere  
if  Qlunc_yaml_inputs['Atmospheric_inputs']['TimeSeries']:

    # Atmos_TS_FILE           = './metadata/AtmosphericData/'+Qlunc_yaml_inputs['Atmospheric_inputs']['Atmos_TS_FILE']
    
    date,wind_direction_ref,velocity_lidar,velocity_mast_ref2,velocity_mast_ref,alpha_ref,temperature_meas = SA.getdata('Spd_106m_Mean_m/s','wsp_140m_LMN_Mean_m/s','wsp_106m_LMN_Mean_m/s','Sdir_103m_LMN_Mean_deg','Available_106m_Mean_avail%',90,'Timestamp_datetime',202204201200,202204241200,4,16,'Tabs_103m_LMN_Mean_degC')    
   

    Atmospheric_Scenario = atmosphere(name              = Qlunc_yaml_inputs['Atmospheric_inputs']['Name'],
                                      temperature_dev   = Qlunc_yaml_inputs['Atmospheric_inputs']['Temperature'],
                                      temperature_meas  = temperature_meas,                                      
                                      wind_direction    = wind_direction_ref,
                                      availability      = 90,
                                      Vref              = velocity_mast_ref,
                                      Vref2             = velocity_mast_ref2,
                                      date              = date, 
                                                    
                                      PL_exp            = [alpha_ref],
                                      wind_tilt         = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind tilt'],
                                      Hg                = Qlunc_yaml_inputs['Atmospheric_inputs']['Height ground']
                                      )

else:    
    # pdb.set_trace()
    Atmospheric_Scenario = atmosphere(name             = Qlunc_yaml_inputs['Atmospheric_inputs']['Name'],
                                      temperature_dev  = Qlunc_yaml_inputs['Atmospheric_inputs']['Temperature'],
                                      temperature_meas = None,
                                      
                                      wind_direction   = (np.linspace(Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'][0],Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'][1],Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'][2])),
                                      availability     = None,

                                      Vref             = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']*np.ones(len(np.linspace(Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'][0],Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'][1],Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'][2]))),
                                      Vref2            = None,
                                      date             = None,
                                      PL_exp           = [Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][i]*np.ones(len(np.linspace(Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'][0],Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'][1],Qlunc_yaml_inputs['Atmospheric_inputs']['Wind direction'][2]))) for i in range(len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))],
                                      wind_tilt        = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind tilt'],
                                      Hg               = Qlunc_yaml_inputs['Atmospheric_inputs']['Height ground'],
                                      )
# pdb.set_trace()

#%% Run Qlunc for different values of tilt angle
########################################################################
# def get_points(radius, center, number_of_points):
#     radians_between_each_point = 2*np.pi/number_of_points
#     list_of_points = []
#     for p in range(0, number_of_points):
#         list_of_points.append( (center[0]+radius*np.cos(p*radians_between_each_point),center[1]+radius*np.sin(p*radians_between_each_point)) )
#     return list_of_points

# rho = 500
# elevation_angle = np.radians( 13.65)
# center = [rho*np.cos(elevation_angle),0]
# A=get_points(rho*np.cos(elevation_angle),center, 12)

# x = [A[ii][0] for ii in range(len(A))]
# y = [A[ii][1] for ii in range(len(A))]

# x1 = [A[ii][0] for ii in range(1,6)]
# y1 = [A[ii][1] for ii in range(1,6)]


# x2 = [A[ii][0] for ii in range(7,12)]
# x2 = x2[::-1]
# y2 = [A[ii][1] for ii in range(7,12)]
# y2 = y2[::-1]


# z=np.zeros(len(x))

# Lidar2_pos=list(zip(x1,y1,z))
# Lidar3_pos=list(zip(x2,y2,z))
# Lidar1_pos=list(zip(z,z,z))

# plt.plot(x,y,'ob')
# plt.gca().set_aspect('equal')

# plt.plot(x1[0],y1[0],'k^')
# plt.plot(x1[-1],y1[-1],'k^')


# plt.plot(x2[0],y2[0],'ro')
# plt.plot(x2[-1],y2[-1],'ro')

#%% Run Qlunc   ############################################################################################
QluncData = Lidar.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)

# for i_position in range(len(Lidar2_pos)):
#     Lidar.optics.scanner.origin = [Lidar1_pos[i_position],Lidar2_pos[i_position],Lidar3_pos[i_position]]
#     # pdb.set_trace()
#     Atmospheric_Scenario.wind_tilt = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind tilt']
#     Atmospheric_Scenario.Vref      = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']
# for i_tilt in np.linspace(Atmospheric_Scenario.wind_tilt[0],Atmospheric_Scenario.wind_tilt[1],Atmospheric_Scenario.wind_tilt[2]):
#     for i_Vref in np.linspace(Atmospheric_Scenario.Vref[0],Atmospheric_Scenario.Vref[1],Atmospheric_Scenario.Vref[2]):
#         Atmospheric_Scenario.wind_tilt = i_tilt
#         Atmospheric_Scenario.Vref = i_Vref
#         QluncData = Lidar.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)