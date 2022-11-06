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
        2) Instantiate the different components
        3) Instantiate modules including corresponding components
        4) Instantiate class `atmosphere` --> atmospheric conditions
        5) Instantiate class `lidar_gral_inp` --> lidar general inputs
        6) Instantiate class `lidar` by including modules, lidar general inputs 
           and atmospheric conditions
           
"""
import os
os.chdir('../')
# importing  uncertainty functions
import UQ_Functions.UQ_Vhorizontal_Classes as uVhc, UQ_Functions.UQ_Photonics_Classes as uphc,UQ_Functions.UQ_Optics_Classes as uopc, UQ_Functions.UQ_Power_Classes as upwc,UQ_Functions.UQ_Lidar_Classes as ulc, UQ_Functions.UQ_ProbeVolume_Classes as upbc,UQ_Functions.UQ_Data_processing_Classes as uprm, UQ_Functions.UQ_SignalProcessor_Classes as uspc
from Utils.Qlunc_ImportModules import *

#%% Running Qlunc_Classes.py:
with open (r'./Main/Qlunc_inputs.yml') as file: # WHere the yaml file is in order to get the input data
    Qlunc_yaml_inputs={}
    docs = yaml.load_all(file, Loader=yaml.FullLoader)
    for doc in docs:      
        for k, v in doc.items():           
            Qlunc_yaml_inputs.setdefault(k,v)  # save a dictionary with the data coming from yaml file 

# Execute Qlunc_Classes.py (creating classes for lidar 'objects')
exec(open(Qlunc_yaml_inputs['Main_directory']+'/Main/Qlunc_Classes.py').read())   
#%%%%%%%%%%%%%%%%% INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Optics components and Module: ##############################################

# Here we create optics components and optics module. User can create as many components as he/she want and combine them to create different module types
# Each module/component is a python object with their own technical characteristics and can be flexible combined to assess different use cases. 

# Scanner:
# pdb.set_trace()
Scanner           = scanner(name            = Qlunc_yaml_inputs['Components']['Scanner']['Name'],           # Introduce your scanner name.
                            scanner_type    = Qlunc_yaml_inputs['Components']['Scanner']['Type'],
                            origin          = Qlunc_yaml_inputs['Components']['Scanner']['Origin'],         # Origin (coordinates of the lidar deployment).
                            N_MC            = Qlunc_yaml_inputs['Components']['Scanner']['N_MC'],
                            N_Points        = Qlunc_yaml_inputs['Components']['Scanner']['N_Points'],
                            pattern         = Qlunc_yaml_inputs['Components']['Scanner']['Pattern'],
                            lissajous_param = Qlunc_yaml_inputs['Components']['Scanner']['Lissajous parameters'],
                            sample_rate     = Qlunc_yaml_inputs['Components']['Scanner']['Sample rate'],    # for now introduce it in [degrees].
                            time_pattern    = Qlunc_yaml_inputs['Components']['Scanner']['Pattern time'],
                            time_point      = Qlunc_yaml_inputs['Components']['Scanner']['Single point measuring time'], 
                            Href            = Qlunc_yaml_inputs['Components']['Scanner']['Href'],
                           # This values for focus distance, cone_angle and azimuth define a typical VAD scanning sequence:
                               # I changed azimuth calculations because with "np.arange" we do not capture the last point in the pattern. "np.arange does not include the last point"; np.linspace capture all the points.
                               # Furthermore, once the time of the pattern is included in the pattern, we will do calculations based on the n° of points yielded by the ratio: time_pattern[sec]/time_point[sec/point]
                               # HAve to decide if wnat np.arange or np.linspace here (azimuth). If 360° is chosen for azimuth, np.arange works, but np.linspace doesn't
                            # azimuth         = np.array(np.arange(Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][0],                                                  
                            #                                       Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][1],
                            #                                       math.floor((Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][1]-Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][0])/(Qlunc_yaml_inputs['Components']['Scanner']['Pattern time']/Qlunc_yaml_inputs['Components']['Scanner']['Single point measuring time'])))), # Azimuth angle in [degrees].
                            
                            # # azimuth         = np.array(np.linspace(Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][0],                                                  
                            # #                                         Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][1],
                            # #                                         math.floor(Qlunc_yaml_inputs['Components']['Scanner']['Pattern time']/Qlunc_yaml_inputs['Components']['Scanner']['Single point measuring time']))), # Azimuth angle in [degrees].                                  
                            # focus_dist      = np.tile(Qlunc_yaml_inputs['Components']['Scanner']['Focus distance'],(1,len(np.linspace(Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][0],                                                  
                            #                                        Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][1],
                            #                                        math.floor(Qlunc_yaml_inputs['Components']['Scanner']['Pattern time']/Qlunc_yaml_inputs['Components']['Scanner']['Single point measuring time'])))))[0],   # Focus distance in [meters]                                                                                                
                            # cone_angle      = np.tile(Qlunc_yaml_inputs['Components']['Scanner']['Cone angle'],(1,len(np.linspace(Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][0],                                                  
                            #                                        Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][1],
                            #                                        math.floor(Qlunc_yaml_inputs['Components']['Scanner']['Pattern time']/Qlunc_yaml_inputs['Components']['Scanner']['Single point measuring time'])))))[0],   # Cone angle in [degrees].
                            
                            azimuth         = Qlunc_yaml_inputs['Components']['Scanner']['Azimuth'][0],
                            focus_dist      = Qlunc_yaml_inputs['Components']['Scanner']['Focus distance'][0],   # Focus distance in [meters]                                                                                                
                            cone_angle      = Qlunc_yaml_inputs['Components']['Scanner']['Cone angle'][0],   # Cone angle in [degrees].
                            
                        
                          
                            stdv_location   = Qlunc_yaml_inputs['Components']['Scanner']['Error origin'],
                            stdv_focus_dist = Qlunc_yaml_inputs['Components']['Scanner']['stdv focus distance'],                 # Focus distance standard deviation in [meters].
                            stdv_cone_angle = Qlunc_yaml_inputs['Components']['Scanner']['stdv Cone angle'],                 # Cone angle standard deviation in [degrees].
                            stdv_azimuth    = Qlunc_yaml_inputs['Components']['Scanner']['stdv Azimuth'],                 # Azimuth angle standard deviation in [degrees].
                            correlations    = Qlunc_yaml_inputs['Components']['Scanner']['correlations'],
                            unc_func        = uopc.UQ_Scanner) #eval(Qlunc_yaml_inputs['Components']['Scanner']['Uncertainty function']) )    # here you put the function describing your scanner uncertainty. 
# Measurement points
# Meas_points = measurement_points (coord_meas_point = Qlunc_yaml_inputs['Components']['Scanner']['Measuring point'])

#Optical Circulator:
Optical_circulator = optical_circulator (name           = Qlunc_yaml_inputs['Components']['Optical Circulator']['Name'],       # Introduce your Optical circulator name.
                                         insertion_loss = Qlunc_yaml_inputs['Components']['Optical Circulator']['Insertion loss'],                        # In [dB]. Insertion loss parameters.
                                         SNR            = Qlunc_yaml_inputs['Components']['Optical Circulator']['SNR'], # [dB] SNR optical circulator
                                         unc_func       = uopc.UQ_OpticalCirculator) #eval(Qlunc_yaml_inputs['Components']['Optical Circulator']['Uncertainty function']))  # Function describing your scanner uncertainty.  Further informaion in "UQ_Optics_Classes.py" comments.


Telescope = telescope (name                       = Qlunc_yaml_inputs['Components']['Telescope']['Name'],
                       stdv_aperture              = Qlunc_yaml_inputs['Components']['Telescope']['Stdv Aperture'],
                       aperture                   = Qlunc_yaml_inputs['Components']['Telescope']['Aperture'],                       
                       focal_length               = Qlunc_yaml_inputs['Components']['Telescope']['Focal length'],
                       fiber_lens_d               = Qlunc_yaml_inputs['Components']['Telescope']['Fiber-lens distance'],
                       fiber_lens_offset          = Qlunc_yaml_inputs['Components']['Telescope']['Fiber-lens offset'],
                       effective_radius_telescope = Qlunc_yaml_inputs['Components']['Telescope']['Effective radius telescope'],
                       output_beam_radius         = Qlunc_yaml_inputs['Components']['Telescope']['Output beam radius'],
                       stdv_focal_length          = Qlunc_yaml_inputs['Components']['Telescope']['stdv Focal length'],
                       stdv_fiber_lens_d          = Qlunc_yaml_inputs['Components']['Telescope']['stdv Fiber-lens distance'],
                       stdv_fiber_lens_offset     = Qlunc_yaml_inputs['Components']['Telescope']['stdv Fiber-lens offset'], 
                       stdv_eff_radius_telescope  = Qlunc_yaml_inputs['Components']['Telescope']['stdv Effective radius telescope'],
                       tau                        = Qlunc_yaml_inputs['Components']['Telescope']['Pulse shape'],
                       tau_meas                   = Qlunc_yaml_inputs['Components']['Telescope']['Gate length'], 
                       stdv_tau                   = Qlunc_yaml_inputs['Components']['Telescope']['stdv Pulse shape'],
                       stdv_tau_meas              = Qlunc_yaml_inputs['Components']['Telescope']['stdv Gate length'], 
                       unc_func                   = uopc.UQ_Telescope)


Probe_Volume = probe_volume (name                       = Qlunc_yaml_inputs['Probe Volume']['Name'],
                             extinction_coef            = Qlunc_yaml_inputs['Probe Volume']['Extinction coeficient'],
                             unc_func                   = upbc.UQ_Probe_volume)


# Optics Module:
Optics_Module =  optics (name               = Qlunc_yaml_inputs['Modules']['Optics Module']['Name'],     # Introduce your Optics Module name.
                         scanner            = eval(Qlunc_yaml_inputs['Modules']['Optics Module']['Scanner']),             # Scanner instance (in this example "Scanner") or "None". "None" means that you don´t want to include Scanner in Optics Module, either in uncertainty calculations.
                         optical_circulator = eval(Qlunc_yaml_inputs['Modules']['Optics Module']['Optical circulator']),  # Optical Circulator instance (in this example "Optical_circulator") or "None". "None" means that you don´t want to include Optical circulator in Optics Module, either in uncertainty calculations.
                         telescope          = eval(Qlunc_yaml_inputs['Modules']['Optics Module']['Telescope']), #Telescope,#
                         probe_volume       = Probe_Volume,#None,#
                         unc_func           = uopc.sum_unc_optics) #eval(Qlunc_yaml_inputs['Modules']['Optics Module']['Uncertainty function']))


## Photonics components and Module: ###########################################
# Here we create photonics components and photonics module. Users can create as many components as they want and combine them to create different module types.

AOM = acousto_optic_modulator (name           = Qlunc_yaml_inputs['Components']['AOM']['Name'],
                               insertion_loss = Qlunc_yaml_inputs['Components']['AOM']['Insertion loss'],
                               unc_func        = uphc.UQ_AOM)

Optical_Amplifier = optical_amplifier(name             = Qlunc_yaml_inputs['Components']['Optical Amplifier']['Name'],        # Introduce your scanner name.
                                      NoiseFig         = Qlunc_yaml_inputs['Components']['Optical Amplifier']['Optical amplifier noise figure'],          # In [dB]. Can introduce it as a table from manufactures (in this example the data is taken from Thorlabs.com, in section EDFA\Graps) or introduce a single well-known value
                                      OA_Gain          = Qlunc_yaml_inputs['Components']['Optical Amplifier']['Optical amplifier gain'],                         # In [dB]. (in this example the data is taken from Thorlabs.com, in section EDFA\Specs)
                                      OA_BW            = Qlunc_yaml_inputs['Components']['Optical Amplifier']['Optical amplifier BW'],
                                      Power_interval   = np.array(np.arange(Qlunc_yaml_inputs['Components']['Optical Amplifier']['Power interval'][0],
                                                                            Qlunc_yaml_inputs['Components']['Optical Amplifier']['Power interval'][1],
                                                                            Qlunc_yaml_inputs['Components']['Optical Amplifier']['Power interval'][2])),
                                      unc_func         = uphc.UQ_Optical_amplifier) #eval(Qlunc_yaml_inputs['Components']['Optical Amplifier']['Uncertainty function']))  # Function describing Optical Amplifier uncertainty. Further informaion in "UQ_Photonics_Classes.py" comments.

Photodetector    = photodetector(name             = Qlunc_yaml_inputs['Components']['Photodetector']['Name'],               # Introduce your photodetector name.
                                 Photo_BandWidth  = Qlunc_yaml_inputs['Components']['Photodetector']['Photodetector BandWidth'],                  # In[]. Photodetector bandwidth
                                 Load_Resistor    = Qlunc_yaml_inputs['Components']['Photodetector']['Load resistor'],                     # In [ohms]
                                 Photo_efficiency = Qlunc_yaml_inputs['Components']['Photodetector']['Photodetector efficiency'],                    # Photodetector efficiency [-]
                                 Dark_Current     = Qlunc_yaml_inputs['Components']['Photodetector']['Dark current'],                   #  In [A]. Dark current in the photodetector.
                                 Photo_SignalP    = Qlunc_yaml_inputs['Components']['Photodetector']['Photodetector signalP'],
                                 Power_interval   = np.array(np.arange(Qlunc_yaml_inputs['Components']['Photodetector']['Power interval'][0],
                                                                       Qlunc_yaml_inputs['Components']['Photodetector']['Power interval'][1],
                                                                       Qlunc_yaml_inputs['Components']['Photodetector']['Power interval'][2])),#np.arange(Qlunc_yaml_inputs['Components']['Photodetector']['Power interval']), # In [w]. Power interval for the photodetector domain in photodetector SNR plot. 
                                 Active_Surf      = Qlunc_yaml_inputs['Components']['Photodetector']['Active Surface'],
                                 Gain_TIA         = Qlunc_yaml_inputs['Components']['Photodetector']['Gain TIA'],                    # In [dB]. If there is a transimpedance amplifier.
                                 V_Noise_TIA      = Qlunc_yaml_inputs['Components']['Photodetector']['V Noise TIA'],                 # In [V]. If there is a transimpedance amplifier.
                                 
                                 unc_func         = uphc.UQ_Photodetector) #eval(Qlunc_yaml_inputs['Components']['Photodetector']['Uncertainty function']))  # Function describing Photodetector uncertainty. Further informaion in "UQ_Photonics_Classes.py" comments.

Laser           = laser(name              = Qlunc_yaml_inputs['Components']['Laser']['Name'],
                        Wavelength        = Qlunc_yaml_inputs['Components']['Laser']['Wavelength'],
                        stdv_wavelength   = Qlunc_yaml_inputs['Components']['Laser']['stdv Wavelength'],
                        conf_int          = Qlunc_yaml_inputs['Components']['Laser']['Confidence interval'],
                        Output_power      = Qlunc_yaml_inputs['Components']['Laser']['Output power'],
                        Laser_Bandwidth   = Qlunc_yaml_inputs['Components']['Laser']['Bandwidth'],
                        RIN               = Qlunc_yaml_inputs['Components']['Laser']['RIN'],
                        unc_func          = uphc.UQ_Laser)
# Module:
Photonics_Module = photonics(name                    = Qlunc_yaml_inputs['Modules']['Photonics Module']['Name'],        # Introduce your Photonics module name
                             photodetector           = eval(Qlunc_yaml_inputs['Modules']['Photonics Module']['Photodetector']),             # Photodetector instance (in this example "Photodetector") or "None". "None" means that you don´t want to include photodetector in Photonics Module, either in uncertainty calculations.
                             optical_amplifier       = eval(Qlunc_yaml_inputs['Modules']['Photonics Module']['Optical amplifier']),         # Scanner instance (in this example "OpticalAmplifier") or "None". "None" means that you don´t want to include Optical Amplifier in Photonics Module, either in uncertainty calculations.
                             laser                   = eval(Qlunc_yaml_inputs['Modules']['Photonics Module']['Laser']),#'None', #Laser,
                             acousto_optic_modulator = eval(Qlunc_yaml_inputs['Modules']['Photonics Module']['AOM']),
                             unc_func                = uphc.sum_unc_photonics) #eval(Qlunc_yaml_inputs['Modules']['Photonics Module']['Uncertainty function']))

## Signal processor components and module: ###########################################################

ADC = analog2digital_converter (name     = Qlunc_yaml_inputs['Components']['ADC']['Name'],
                                nbits    = Qlunc_yaml_inputs['Components']['ADC']['N bits'],
                                vref     = Qlunc_yaml_inputs['Components']['ADC']['Reference voltage'],
                                vground  = Qlunc_yaml_inputs['Components']['ADC']['Ground voltage'],
                                q_error  = Qlunc_yaml_inputs['Components']['ADC']['Quantization error'],
                                ADC_bandwidth = Qlunc_yaml_inputs['Components']['ADC']['ADC Bandwidth'],
                                unc_func = uspc.UQ_ADC)

Signal_processor_Module = signal_processor(name                     = Qlunc_yaml_inputs['Modules']['Signal processor Module']['Name'],
                                           analog2digital_converter = eval(Qlunc_yaml_inputs['Modules']['Signal processor Module']['ADC']),
                                           # f_analyser             = Qlunc_yaml_inputs['Modules']['Signal processor Module']['Frequency analyser'],
                                           unc_func                 = uspc.sum_unc_signal_processor)



## Lidar general inputs: ######################################################
Lidar_inputs     = lidar_gral_inp(name        = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Name'],      # Introduce the name of your lidar data folder.
                                  wave        = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Wavelength'],                    # In [m]. Lidar wavelength.
                                  ltype       = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Type'],
                                  yaw_error   = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Yaw error'],                          # In [°]. Degrees of rotation around z axis because of inclinometer errors
                                  pitch_error = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Pitch error'],                          # In [°]. Degrees of rotation around y axis
                                  roll_error  = Qlunc_yaml_inputs['Components']['Lidar general inputs']['Roll error'],                        # In [°]. Degrees of rotation around z axis.
                                  dataframe   = { })  # Final dataframe


#%% Data processing methods

# Wind field reconstruction model
WFR_M = wfr (name                 = Qlunc_yaml_inputs['WFR model']['Name'],
              reconstruction_model = Qlunc_yaml_inputs['WFR model']['Model'],
              unc_func             = uprm.UQ_WFR)
# pdb.set_trace()
# Data filtering method
Filt_M = filtering_method (name        = Qlunc_yaml_inputs['Filtering method']['Name'],
                           filt_method = Qlunc_yaml_inputs['Filtering method']['Method'],
                           unc_func    = 'uprm.UQ_WFR')

#%% LIDAR device

Lidar = lidar(name           = Qlunc_yaml_inputs['Lidar']['Name'],                       # Introduce the name of your lidar device.
              photonics      = Photonics_Module, #eval(Qlunc_yaml_inputs['Lidar']['Photonics module']),     # Introduce the name of your photonics module.
              optics         = Optics_Module, #eval(Qlunc_yaml_inputs['Lidar']['Optics module']),        # Introduce the name of your optics module.
              power          = None, #eval(Qlunc_yaml_inputs['Lidar']['Power module']),         # Introduce the name of your power module. NOT IMPLEMENTED YET!
              signal_processor = Signal_processor_Module,
              wfr_model      = WFR_M,
              filt_method    = None,
              probe_volume   = Probe_Volume, 
              lidar_inputs   = Lidar_inputs, #eval(Qlunc_yaml_inputs['Lidar']['Lidar inputs']),         # Introduce lidar general inputs
              unc_func       = ulc.sum_unc_lidar,
              unc_Vh         = uVhc.UQ_Vh) #eval(Qlunc_yaml_inputs['Lidar']['Uncertainty function'])) # Function estimating lidar global uncertainty


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
                                      PL_exp         = Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp'],
                                      Vref           = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref'],
                                      wind_direction = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind_direction'],
                                      wind_tilt      = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind_tilt'],
                                      Hg             = Qlunc_yaml_inputs['Atmospheric_inputs']['Height_ground'])

else:    

    Atmospheric_Scenario = atmosphere(name           = Qlunc_yaml_inputs['Atmospheric_inputs']['Name'],
                                      temperature    = Qlunc_yaml_inputs['Atmospheric_inputs']['Temperature'],
                                      PL_exp         = Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp'],
                                      Vref           = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref'],
                                      wind_direction = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind_direction'],
                                      wind_tilt     = Qlunc_yaml_inputs['Atmospheric_inputs']['Wind_tilt'],
                                      Hg             = Qlunc_yaml_inputs['Atmospheric_inputs']['Height_ground'])

