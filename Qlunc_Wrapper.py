# -*- coding: utf-8 -*-
"""
Created on Thu May 21 00:11:03 2020

@author: fcosta
"""

from Qlunc_ImportModules import *
import Qlunc_Help_standAlone as SA
import Qlunc_inputs

#%%Naming data to make it easier:
#Atmosphere:
temp        = inputs.atm_inp.Atmospheric_inputs['temperature']
hum         = inputs.atm_inp.Atmospheric_inputs['humidity']
wave        = inputs.lidar_inp.Lidar_inputs['Wavelength']
#Power
conv_noise  = inputs.power_inp.Converter_uncertainty_inputs['converter_noise']
conv_oc     = inputs.power_inp.Converter_uncertainty_inputs['converter_OtherChanges']
conv_losses = inputs.power_inp.Converter_uncertainty_inputs['converter_losses']
ps_noise    = inputs.power_inp.PowerSource_uncertainty_inputs['power_source_noise']
ps_oc       = inputs.power_inp.PowerSource_uncertainty_inputs['power_source_OtherChanges']
#photonics
amp_noise   = inputs.photonics_inp.Optical_amplifier_uncertainty_inputs['Optical_amplifier_noise']
amp_oc      = inputs.photonics_inp.Optical_amplifier_uncertainty_inputs['Optical_amplifier_OtherChanges']
#amp_noise_figure = Qlunc_UQ_Photonics_func.FigNoise(inputs,direct)
ls_noise    = inputs.photonics_inp.LaserSource_uncertainty_inputs['laser_source_noise']
ls_oc       = inputs.photonics_inp.LaserSource_uncertainty_inputs['laser_source_OtherChanges']
photo_noise = inputs.photonics_inp.Photodetector_uncertainty_inputs['photodetector_noise']
photo_oc    = inputs.photonics_inp.Photodetector_uncertainty_inputs['photodetector_OtherChanges']
#telescope
tele_cl     = inputs.optics_inp.Telescope_uncertainty_inputs['telescope_curvature_lens']
tele_oc     = inputs.optics_inp.Telescope_uncertainty_inputs['telescope_OtherChanges']
tele_aberr  = inputs.optics_inp.Telescope_uncertainty_inputs['telescope_aberration']
tele_losses = inputs.optics_inp.Telescope_uncertainty_inputs['telescope_losses']


# %%Getting scenarios:
def Get_Scenarios():
    # Initialazing variables:
    global Scenarios
    Scenarios=dict()
    Temperature=[]       
    type_noise = [[wave]]
    Val        = ()
    Names_Val=['VAL_WAVE'] # initialize with wave because is a variable common to many component noises.
    
    # We need to create the cases we want to loop over. For that are the next steps to create 'type_noise' and 'Val'.
    # First create a dictionary (add_typeN) to identify type of noise with variables we want to use to loop over.
    
    add_typeN={'power_source_noise'  :[[[ps_noise],'VAL_NOISE_POWER_SOURCE',[inputs.VAL.VAL_NOISE_POWER_SOURCE]],   [[ps_oc],'VAL_OC_POWER_SOURCE',[inputs.VAL.VAL_OC_POWER_SOURCE] ]]  ,
                                                                     
               'converter_losses'    :[[[conv_losses],'VAL_CONVERTER_LOSSES',[inputs.VAL.VAL_CONVERTER_LOSSES] ] ] ,           
               'converter_noise'     :[[[conv_noise],'VAL_NOISE_CONVERTER',[inputs.VAL.VAL_NOISE_CONVERTER]],[[conv_oc],'VAL_OC_CONVERTER',[inputs.VAL.VAL_OC_CONVERTER]]],
                                                                   
               'photodetector_noise' :[[[photo_noise],'VAL_NOISE_PHOTO',[inputs.VAL.VAL_NOISE_PHOTO]],[[photo_oc], 'VAL_OC_PHOTO',[inputs.VAL.VAL_OC_PHOTO]  ] ],
                                                                   
#               'Optical_amplifier_fignoise'  :[[[amp_noise_figure], 'VAL_NOISE_FIG',[inputs.VAL.VAL_NOISE_FIG]  ]  ],
               'Optical_amplifier_noise'     :[[[amp_oc],  'VAL_OC_AMPLI',[inputs.VAL.VAL_OC_AMPLI]],   [[amp_noise],'VAL_NOISE_AMPLI',[inputs.VAL.VAL_NOISE_AMPLI]] ],  
                                                                   
               'laser_source_noise'  :[[[ls_noise],'VAL_NOISE_LASER_SOURCE',[inputs.VAL.VAL_NOISE_LASER_SOURCE]], [[ls_oc],'VAL_OC_LASER_SOURCE',[inputs.VAL.VAL_OC_LASER_SOURCE] ]      ],
                                                                    
               'telescope_noise'     :[[[tele_aberr],'VAL_ABERRATION_TELESCOPE',[inputs.VAL.VAL_ABERRATION_TELESCOPE]],[[tele_oc],'VAL_OC_TELESCOPE',[inputs.VAL.VAL_OC_TELESCOPE]],
                                       [[tele_cl], 'VAL_CURVE_LENS_TELESCOPE',[inputs.VAL.VAL_CURVE_LENS_TELESCOPE]]],
                                                                     
                                                                    
               'telescope_losses'    :[[[tele_losses],'VAL_LOSSES_TELESCOPE',[inputs.VAL.VAL_LOSSES_TELESCOPE]]]}              

    
 
#    type_noise = [wave,conv_noise,conv_oc,conv_losses,ps_noise,ps_oc,amp_noise,amp_oc,amp_noise_figure,ls_noise,ls_oc,photo_noise,photo_oc,
#                  tele_cl,tele_oc,tele_aberr,tele_losses]
    #%%Loop to create typeN and Val depending on the user modules/components inputs):

    for user_typeN in list(itertools.chain(*user_inputs.user_itype_noise)): # For user selection noises
        for i in range(len(add_typeN[user_typeN])):
            type_noise.append(((add_typeN[user_typeN][i][0]))) # obtaining the values we want to loop over
            Names_Val.append(((add_typeN[user_typeN][i][1])))  # obtaining the names to create the dictionary Scenarios, to pass as **Scenarios (in this way we can dinamically vary the variables of the functions)
            Val=Val+((tuple(add_typeN[user_typeN][i][2])))     # obtaining the number of none´s we need to run the loop
            

    # %%main loop to go over all variables to create the cases:############### 
    # FIGURE NOISE IS NOT INCLUDED IN THE SCENARIOS BECAUSE IS NOT NEEDED FOR ANY CALCULATION. IT IS JUST A 'NOISE' MORE TO ADD AT THE END, 
    # WHEN BUILDING THE DATA FRAME  
    
    
    for i in list(SA.flatten('VAL_T','VAL_H',Names_Val)):
        Scenarios[i]=[]
    
    for inputs.VAL.VAL_T,inputs.VAL.VAL_H in zip (temp, hum):
            for Val in itertools.product(*list(itertools.chain(*type_noise))):
                for k,v in zip (Scenarios.keys(), (inputs.VAL.VAL_T,)+(inputs.VAL.VAL_H,)+Val): # for loop to build up the dictionary 'Scenarios'. If user includes some variability (dependency on wavelength e.g. of any variable)              
                    Scenarios[k].append(v)
#                Scenarios.append(list(flatten(inputs.VAL.VAL_T,inputs.VAL.VAL_H,inputs.VAL.VAL_WAVE,inputs.VAL.VAL_NOISE_FIG,Val))) #
                Temperature.append([inputs.VAL.VAL_T])
#    pdb.set_trace()
    return Scenarios,Temperature
#%% Running the different cases. If user has included it, the case is evaluated: Can I do this in a loop?????????
   
def Get_Noise(module,Scenarios):    
    METHODS={}
    if module=='power':
        if 'power_source_noise' in list(SA.flatten(user_inputs.user_itype_noise)):
            METHODS.setdefault('power_source_noise',Qlunc_UQ_Power_func.UQ_PowerSource(**Scenarios))   # 'Setdefault' is just like append but it's used when no element is yet included in the dictionary     
        if 'converter_noise' in list(SA.flatten(user_inputs.user_itype_noise)):     
            METHODS.setdefault('converter_noise',Qlunc_UQ_Power_func.UQ_Converter(**Scenarios))        
        if 'converter_losses' in list(SA.flatten(user_inputs.user_itype_noise)):        
            METHODS.setdefault('converter_losses',Qlunc_UQ_Power_func.Losses_Converter(**Scenarios))  
            
    if module=='photonics':
        if 'laser_source_noise' in list(SA.flatten(user_inputs.user_itype_noise)):        
            METHODS.setdefault('laser_source_noise',Qlunc_UQ_Photonics_func.UQ_LaserSource(**Scenarios))        
        if 'photodetector_noise' in list(SA.flatten(user_inputs.user_itype_noise)):        
            METHODS.setdefault('photodetector_noise',Qlunc_UQ_Photonics_func.UQ_Photodetector(**Scenarios))               
        if 'Optical_amplifier_noise' in list(SA.flatten(user_inputs.user_itype_noise)):        
            METHODS.setdefault('Optical_amplifier_noise',Qlunc_UQ_Photonics_func.UQ_Optical_amplifier(**Scenarios))              
        if 'Optical_amplifier' in list(SA.flatten(user_inputs.user_icomponents)):        
            METHODS.setdefault('Optical_amplifier_fignoise',Qlunc_UQ_Photonics_func.FigNoise(inputs,direct,**Scenarios))
    
    if module=='optics':          
        if 'telescope_noise' in list(SA.flatten(user_inputs.user_itype_noise)):        
            METHODS.setdefault('telescope_noise',Qlunc_UQ_Optics_func.UQ_Telescope(**Scenarios))               
        if 'telescope_losses' in list(SA.flatten(user_inputs.user_itype_noise)):        
            METHODS.setdefault('telescope_losses',Qlunc_UQ_Optics_func.Losses_Telescope(**Scenarios))        

    return METHODS