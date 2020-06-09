# -*- coding: utf-8 -*-
"""
Created on Thu May 21 00:11:03 2020

@author: fcosta
"""

from Qlunc_ImportModules import *
from  Qlunc_Help_standAlone import flatten
import Qlunc_inputs

#%%Naming data to make it easier:


class user_inputs():


    user_imodules=list(inputs.modules.keys())
    user_icomponents=[list(reduce(getitem,[i],inputs.modules).keys()) for i in inputs.modules.keys()]
    user_itype_noise= [list(inputs.modules[module].get(components,{})) for module in inputs.modules.keys() for components in inputs.modules[module].keys()]

input_values_LOOP=[]
input_values_LOOP2={}
# Find the data want to loop over inside classes and nested classes:  
#This if is because have to calculate the figure noise to pass it as a int parameter instead a string
if 'Optical_amplifier_noise' in list(flatten(user_inputs.user_itype_noise)):
    inputs.photonics_inp.Optical_amplifier_inputs['Optical_amplifier_noise']['Optical_amplifier_NF']=Qlunc_UQ_Photonics_func.FigNoise(inputs,direct)
    
    
inputs_attributes=[atr for atr in dir(inputs) if inspect.getmembers(getattr(inputs,atr))]
inputs_attributes=list([a for a in inputs_attributes if not(a.startswith('__') and a.endswith('__'))]) # obtaining attributes from the class inputs 
inputs_attributes=inputs_attributes[3:] # Only take component values, not modules, atmospheric or general values
res2={}
for ind_ATR in inputs_attributes:
    fd=eval('inputs.'+ind_ATR)
    res=inspect.getmembers(fd,lambda a:not(inspect.isroutine(a)))
    res2.setdefault(ind_ATR,list([a for a in res if not(a[0].startswith('__') and a[0].endswith('__')) ]))
input_values=list(flatten(list(res2.values())))

# Extract from input classes the info we need to loop over (values of the components and names included by the user)
LOOP_inputs_dict=[]
Values2loop=[]
Names2loop=[]
Val=()
for index_dict in range(len(input_values)):  
    if isinstance (input_values[index_dict],dict) :
        LOOP_inputs_dict.append(input_values[index_dict]) # Just to keep the dictionary objects inside the different classes disregarding atmospheric and genetal lidar inputs
for index_loop0 in range(len(LOOP_inputs_dict)):
    for index_loop1 in LOOP_inputs_dict[index_loop0].keys():
        if index_loop1 in list(flatten(user_inputs.user_itype_noise)):
            Values2loop.append(list(LOOP_inputs_dict[index_loop0][index_loop1].values()))
            Names2loop.append(list(LOOP_inputs_dict[index_loop0][index_loop1].keys()))
Val=[None]*len(list(flatten(Names2loop))) # We need this to loop over without having to put all variables in the loop
Names2loop=list(flatten(Names2loop))

#print(Values2loop)
#print(Names2loop)
#print(Val)

#Atmosphere:
Temp        = inputs.atm_inp.Atmospheric_inputs['temperature']
Hum         = inputs.atm_inp.Atmospheric_inputs['humidity']



# %%Getting scenarios:
def Get_Scenarios():
    # Initialazing variables:
    global Scenarios
    Scenarios=dict()
    TempCol=[]       

#    Temperature=None # initialize this values in None, as well as the values in 'add_typeN' to pass it as empty values to fill them in the loop when getting Scenarios!!!!!
#    Humidity=None

    #%%Loop to create typeN and Val depending on the user modules/components inputs):

    # Main loop to go over all variables to create the cases:############### 
    # FIGURE NOISE IS NOT INCLUDED IN THE SCENARIOS BECAUSE IS NOT NEEDED FOR ANY CALCULATION. IT IS JUST A 'NOISE' MORE TO ADD AT THE END, 
    # WHEN BUILDING THE DATA FRAME  
#    pdb.set_trace()
    
    for i in list(flatten('Temperature','Humidity',Names2loop)):
        Scenarios[i]=[]
    
    for VAL_Temp,VAL_Hum in zip (Temp, Hum): # This for lop is apart of the others because of the zip, since T and H shouldn´t be mixed to obtain different scenarions --> T and H are paired for each scenario
#      for VAL_WAVE,VAL_NOISE_FIG in zip(wave,amp_noise_figure):
        for Val in list(itertools.product(*list(itertools.chain(*Values2loop)))): # This makes all possible combinations among user inputs
                
            
            
            for k,v in zip (Scenarios.keys(), (VAL_Temp,)+(VAL_Hum,)+Val): # for loop to build up the dictionary 'Scenarios'. If user includes some variability (dependency on wavelength e.g. of any variable)              
                Scenarios[k].append(v)
#                Scenarios.append(list(flatten(inputs.VAL.VAL_T,inputs.VAL.VAL_H,inputs.VAL.VAL_WAVE,inputs.VAL.VAL_NOISE_FIG,Val))) #
            TempCol.append(Temp)
#    pdb.set_trace()
    return Scenarios,TempCol
#%% Running the different cases. If user has included it, the case is evaluated: Can I do this in a loop??????
    # User have to include here the module, component and estimation method
   
#def Get_Noise(module,Scenarios):    
#    METHODS={}
#    if module=='power':
#        if 'power_source_noise' in list(SA.flatten(user_inputs.user_itype_noise)):
#            METHODS.setdefault('power_source_noise',Qlunc_UQ_Power_func.UQ_PowerSource(**Scenarios))   # 'Setdefault' is just like append but it's used when no element is yet included in the dictionary     
#        if 'converter_noise' in list(SA.flatten(user_inputs.user_itype_noise)):     
#            METHODS.setdefault('converter_noise',Qlunc_UQ_Power_func.UQ_Converter(**Scenarios))        
#        if 'converter_losses' in list(SA.flatten(user_inputs.user_itype_noise)):        
#            METHODS.setdefault('converter_losses',Qlunc_UQ_Power_func.Losses_Converter(**Scenarios))  
#            
#    if module=='photonics':
#        if 'laser_source_noise' in list(SA.flatten(user_inputs.user_itype_noise)):        
#            METHODS.setdefault('laser_source_noise',Qlunc_UQ_Photonics_func.UQ_LaserSource(**Scenarios))        
#        if 'photodetector_noise' in list(SA.flatten(user_inputs.user_itype_noise)):        
#            METHODS.setdefault('photodetector_noise',Qlunc_UQ_Photonics_func.UQ_Photodetector(user_inputs,inputs,cts,**Scenarios))               
#        if 'Optical_amplifier_noise' in list(SA.flatten(user_inputs.user_itype_noise)):        
#            METHODS.setdefault('Optical_amplifier_noise',Qlunc_UQ_Photonics_func.UQ_Optical_amplifier(**Scenarios))              
#        if 'Optical_amplifier' in list(SA.flatten(user_inputs.user_icomponents)):        
#            METHODS.setdefault('Optical_amplifier_fignoise',Qlunc_UQ_Photonics_func.FigNoise(inputs,direct,**Scenarios))
#    
#    if module=='optics':          
#        if 'telescope_noise' in list(SA.flatten(user_inputs.user_itype_noise)):        
#            METHODS.setdefault('telescope_noise',Qlunc_UQ_Optics_func.UQ_Telescope(**Scenarios))               
#        if 'telescope_losses' in list(SA.flatten(user_inputs.user_itype_noise)):        
#            METHODS.setdefault('telescope_losses',Qlunc_UQ_Optics_func.Losses_Telescope(**Scenarios))        
#
#    return METHODS
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
                'Optical_amplifier_noise' : Qlunc_UQ_Photonics_func.UQ_Optical_amplifier 
                }
#        if 'Optical_amplifier' in list(SA.flatten(user_inputs.user_icomponents)): 
#            # For methods that we want them to appear in estimations although they´re not in the 'user_inputs.user_itype_noise'(user options) list, like the optical amplifier noise figure
#            # wich is estimated automatically when introducing the optical amplifier as a component and it is not involved in any calculations:
#            METHODS.setdefault('Optical_amplifier_fignoise',Qlunc_UQ_Photonics_func.FigNoise(user_inputs,inputs,direct,**Scenarios)) 
#    
    elif module=='Optics':
        Func = {'Telescope_noise'     : Qlunc_UQ_Optics_func.UQ_Telescope,
                'Telescope_losses'    : Qlunc_UQ_Optics_func.Losses_Telescope
                }
           
    for k,v in Func.items():
        if k in list(SA.flatten(user_inputs.user_itype_noise)):  
            METHODS.setdefault(k,list(SA.flatten(Func[k](user_inputs,inputs,cts,direct,**Scenarios))))
#    pdb.set_trace()
    return METHODS