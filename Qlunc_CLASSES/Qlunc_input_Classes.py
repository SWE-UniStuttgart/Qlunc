# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:26:23 2020

@author: fcosta
"""

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

Scanner1          = scanner(name           = 'Scan1',
                           origin          = [0,0,0], #Origin
                           focus_dist      = 80,
                           sample_rate     = 10,
                           theta           = np.array([5]),
                           phi             = np.array([np.arange(0,360,10)]) ,
                           stdv_focus_dist = 0.0,
                           stdv_theta      = 0.0,
                           stdv_phi        = 0.0,
                           unc_func        = uopc.UQ_Scanner)       

Scanner2          = scanner(name            = 'Scan2',
                           origin          = [0,0,0], #Origin
                           focus_dist      = 80,
                           sample_rate     = 10,
                           theta           = np.array([5]),
                           phi             = np.array([np.arange(0,360,10)]) ,
                           stdv_focus_dist = 0.9,
                           stdv_theta      = 0.1,
                           stdv_phi        = 0.1,
                           unc_func        = uopc.UQ_Scanner)       

Scanner3          = scanner(name           = 'Scan3',
                           origin          = [0,0,0], #Origin
                           focus_dist      = 80,
                           sample_rate     = 10,
                           theta           = np.array([5]),
                           phi             = np.array([np.arange(0,360,10)]) ,
                           stdv_focus_dist = 1.4,
                           stdv_theta      = 0.8,
                           stdv_phi        = 0.87,
                           unc_func        = uopc.UQ_Scanner)       


# Module:

Optics_Module1 =  optics (name     = 'OptMod1',
                         scanner  = Scanner1,
                         unc_func = uopc.UQ_Scanner) # here you put the function describing your uncertainty
Optics_Module2 =  optics (name     = 'OptMod2',
                         scanner  = Scanner2,
                         unc_func = uopc.UQ_Scanner)
Optics_Module3 =  optics (name     = 'OptMod3',
                         scanner  = Scanner3,
                         unc_func = uopc.UQ_Scanner)

#############  Photonics ###################

# Components:

OpticalAmplifier = optical_amplifier(name     = 'OA1',
                                     OA_NF    = 'NoiseFigure.csv',
                                     OA_Gain  = 30,
                                     unc_func = uphc.UQ_Optical_amplifier)

Photodetector    = photodetector(name        = 'Photo1',
                                 Photo_BW    = 1e9,
                                 RL          = 50,
                                 n           = .85,
                                 DC          = 5e-9,
                                 Photo_SP    = 1e-3,
                                 G_TIA       = 5e3,
                                 V_noise_TIA = 160e-6,
                                 unc_func    = uphc.UQ_Photodetector)

# Module:

Photonics_Module = photonics(name              = 'PhotoMod1',
                             photodetector     = Photodetector, # or None
                             optical_amplifier = OpticalAmplifier,
                             unc_func          = uphc.sum_unc_photonics)

#############  Power #########################################

# Components:

PowerSource      = power_source(name      = 'P_Source1',
                                Inp_power = 1,
                                Out_power = 2,
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
                                  sample_rate = 2)


##########  LIDAR  #####################################

Lidar1 = lidar(name         = 'Caixa1',
               photonics    = Photonics_Module,
               optics       = Optics_Module1,
               power        = Power_Module,
               lidar_inputs = Lidar_inputs)
#              unc_func     = UQ_Lidar)
Lidar2 = lidar(name         = 'Caixa2',
               photonics    = Photonics_Module,
               optics       = Optics_Module2,
               power        = Power_Module,
               lidar_inputs = Lidar_inputs)
#              unc_func     = UQ_Lidar)

Lidar3 = lidar(name         = 'Caixa3',
               photonics    = Photonics_Module,
               optics       = Optics_Module3,
               power        = Power_Module,
               lidar_inputs = Lidar_inputs)
#              unc_func     = UQ_Lidar)

#%% Creating atmospheric scenarios:
TimeSeries=True  # This defines whether we are using a time series (True) or single values (False) to describe the atmosphere (T, H, rain and fog) 
                  # If so we obtain a time series describing the noise implemented in the measurement.
if TimeSeries:
    Atmos_TS_FILE           = 'AtmosphericScenarios.csv'
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
                                    temperature = [300])