# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:26:23 2020

@author: fcosta
"""
#%% Plot Flags

flags.flag_plot_pointing_accuracy_unc    = False   # Plot flags:
flags.flag_plot_measuring_points_pattern = True
flags.flag_plot_photodetector_noise      = True



#%% ################################################################
###################### __ Instances __ #################################

'''
Here the lidar device is made up with respective modules and components.
Instances of each component are created to build up the corresponding module. Once we have the components
we create the modules, adding up the components. In the same way, once we have the different modules that we
want to include in our lidar, we can create it adding up the modules we have been created.
Different lidars, atmospheric scenarios, components and modules can be created on paralel and can be combined
easily.
Example: ....
'''
#############  Optics ###################

# Components:

#Scanner1          = scanner(name           = 'Scan1',
#                           origin          = [0,0,0], #Origin
#                           focus_dist      = np.array([500,600,750,950,1250,1550,2100, 2950,3700]),
#                           sample_rate     = 10,
#                           cone_angle           = np.array([0]*9),
#                           azimuth             = np.array([0]*9) ,
#                           stdv_focus_dist = .1,
#                           stdv_cone_angle      = .1,
#                           stdv_azimuth        = .1,
#                           unc_func        = uopc.UQ_Scanner)       
Scanner1          = scanner(name           = 'Scan1',
                           origin          = [1,22,1], #Origin
                           focus_dist      = np.array([40]*24),
                           sample_rate     = 10,
                           cone_angle           = np.array([7]*24), #np.linspace(0,125,24),#
                           azimuth             = np.arange(0,360,15),
                           stdv_focus_dist = 0,
                           stdv_cone_angle      = 0,
                           stdv_azimuth        = 0,
                           unc_func        = uopc.UQ_Scanner)  
Scanner2          = scanner(name           = 'Scan2',
                           origin          = [1,22,1], #Origin
                           focus_dist      = np.array([80]*24),
                           sample_rate     = 10,
                           cone_angle           = np.array([7]*24),
                           azimuth             = np.arange(0,360,15),
                           stdv_focus_dist = 0,
                           stdv_cone_angle      = 0,
                           stdv_azimuth        = 0,
                           unc_func        = uopc.UQ_Scanner)       

Scanner3          = scanner(name           = 'Scan3',
                           origin          = [1,22,1], #Origin
                           focus_dist      = np.array([120]*24),
                           sample_rate     = 10,
                           cone_angle           = np.array([7]*24),# np.linspace(0,180,24),#
                           azimuth             = np.arange(0,360,15) ,#np.array([0]*24),#
                           stdv_focus_dist = 0,
                           stdv_cone_angle      = 0,
                           stdv_azimuth        = 0,
                           unc_func        = uopc.UQ_Scanner)       

Optical_circulator1 = optical_circulator (name = 'OptCirc1',
                                          insertion_loss = -100,
                                          unc_func = uopc.UQ_OpticalCirculator) 
# Module:

Optics_Module1 =  optics (name                = 'OptMod1',
                         scanner              = Scanner1,#None,#
                         optical_circulator   = Optical_circulator1,
                         laser                = None,
                         unc_func             = uopc.sum_unc_optics) # here you put the function describing your uncertainty
Optics_Module2 =  optics (name                = 'OptMod2',
                         scanner              = Scanner2,
                         optical_circulator   = Optical_circulator1,
                         laser                = None,
                         unc_func             = uopc.sum_unc_optics)
Optics_Module3 =  optics (name                = 'OptMod3',
                         scanner              = Scanner3,
                         optical_circulator   = Optical_circulator1,
                         laser                = None,
                         unc_func             = uopc.sum_unc_optics)

#############  Photonics ###################

# Components:

OpticalAmplifier = optical_amplifier(name     = 'OA1',
                                     OA_NF    = 50 ,#'NoiseFigure.csv',#
                                     OA_Gain  = 30,
                                     unc_func = uphc.UQ_Optical_amplifier)

Photodetector    = photodetector(name             = 'Photo1',
                                 Photo_BandWidth  = 380e6,
                                 Load_Resistor    = 50,
                                 Photo_efficiency = .85,
                                 Dark_Current     = 5e-9,
                                 Photo_SignalP    = 1e-3,
                                 Power_interval   = np.arange(0,1000,.001), # power interval for the photodetector interval
                                 Gain_TIA         = 5e3,
                                 V_Noise_TIA      = 160e-6,
                                 unc_func         = uphc.UQ_Photodetector)

# Module:

Photonics_Module = photonics(name              = 'PhotoMod1',
                             photodetector     = Photodetector, # or None
                             optical_amplifier = OpticalAmplifier,# None,#
                             unc_func          = uphc.sum_unc_photonics)

#############  Power #########################################

# Components:

PowerSource      = power_source(name      = 'P_Source1',
                                Inp_power = 1,
                                Out_power = 2e-3,
                                unc_func  = upwc.UQ_PowerSource)

Converter        = converter(name      = 'Conv1',
                             frequency = 100,
                             Conv_BW   = 1e5,
                             Infinit   = .8,
                             unc_func  = upwc.UQ_Converter)

# Module:

Power_Module     = power(name         = 'PowerMod1',
                         power_source = PowerSource,
                         converter    = Converter,
                         unc_func     = upwc.sum_unc_power)



########## Lidar general inputs #########################:
Lidar_inputs     = lidar_gral_inp(name        = 'Gral_inp1', 
                                  wave        = 1550e-9, 
                                  sample_rate = 2,       # Hz
                                  yaw_error   = 0,       # Degreesof rotation around z axis because of inclinometer errors
                                  pitch_error = 90,       # Degrees of rotation around y axis
                                  roll_error  = 0)       # Degrees of rotation around z axis


##########  LIDAR  #####################################

Lidar1 = lidar(name         = 'Caixa1',
               photonics    = Photonics_Module,
               optics       = Optics_Module1,
               power        = None,
               lidar_inputs = Lidar_inputs,
               unc_func     = ulc.sum_unc_lidar)
Lidar2 = lidar(name         = 'Caixa2',
               photonics    = Photonics_Module,
               optics       = Optics_Module2,
               power        = Power_Module,
               lidar_inputs = Lidar_inputs,
               unc_func     = ulc.sum_unc_lidar)

Lidar3 = lidar(name         = 'Caixa3',
               photonics    = Photonics_Module,
               optics       = Optics_Module3,
               power        = Power_Module,
               lidar_inputs = Lidar_inputs,
               unc_func     = ulc.sum_unc_lidar)

#%% Creating atmospheric scenarios:
Atmospheric_TimeSeries = True # This defines whether we are using a time series (True) or single values (False) to describe the atmosphere (T, H, rain and fog) 
                               # If so we obtain a time series describing the noise implemented in the measurement.
if Atmospheric_TimeSeries:
    Atmos_TS_FILE           = '../metadata/AtmosphericData/AtmosphericScenarios.csv'
    AtmosphericScenarios_TS = pd.read_csv(Atmos_TS_FILE,delimiter=';',decimal=',')
    Atmospheric_inputs={
                        'temperature' : list(AtmosphericScenarios_TS.loc[:,'T']),# [K]
                        'humidity'    : list(AtmosphericScenarios_TS.loc[:,'H']),# [%]
                        'rain'        : list(AtmosphericScenarios_TS.loc[:,'rain']),
                        'fog'         : list(AtmosphericScenarios_TS.loc[:,'fog']),
                        'time'        : list(AtmosphericScenarios_TS.loc[:,'t'])#for rain and fog intensity intervals might be introduced [none,low, medium high]
                        } 
    Atmospheric_Scenario=atmosphere(name        = 'Atmosphere1',
                                    temperature = Atmospheric_inputs['temperature'])
else:    

    Atmospheric_Scenario=atmosphere(name        = 'Atmosphere1',
                                    temperature = [1])


