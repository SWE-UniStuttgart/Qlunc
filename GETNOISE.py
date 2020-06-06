# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:08:57 2020

@author: fcosta
"""

def Get_Noise(module,Scenarios):    
    METHODS={}
    if module == 'Power':
        Func = {'Power_source_noise'  : Qlunc_UQ_Power_func.UQ_PowerSource,
                'Converter_noise'     : Qlunc_UQ_Power_func.UQ_Converter,
                'Converter_losses'    : Qlunc_UQ_Power_func.Losses_Converter,
                }
    
    elif module== 'Photonics':
        Func = {'Laser_source_noise'      : Qlunc_UQ_Photonics_func.UQ_LaserSource,
                'Photodetector_noise'     : Qlunc_UQ_Photonics_func.UQ_Photodetector,
                'Optical_amplifier_noise' : Qlunc_UQ_Photonics_func.UQ_Optical_amplifier, 
                }
        if 'Optical_amplifier' in list(SA.flatten(user_inputs.user_icomponents)): 
            # For methods that we want them to appear in estimations although they´re not in the 'user_inputs.user_itype_noise'(user options) list, like the optical amplifier noise figure
            # wich is estimated automatically when introducing the optical amplifier as a component and it is not involved in any calculations:
            METHODS.setdefault('Optical_amplifier_fignoise',Qlunc_UQ_Photonics_func.FigNoise(user_inputs,inputs,direct,**Scenarios)) 
    
    elif module=='Optics':
        Func = {'Telescope_noise'     : Qlunc_UQ_Optics_func.UQ_Telescope,
                'Telescope_losses'    : Qlunc_UQ_Optics_func.Losses_Telescope
                }
           
    for k,v in Func.items():
        if k in list(SA.flatten(user_inputs.user_itype_noise)):  
            METHODS.setdefault(k,list(SA.flatten(Func[k](user_inputs,inputs,cts,**Scenarios))))
#    pdb.set_trace()
    return METHODS