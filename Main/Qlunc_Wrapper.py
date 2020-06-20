# -*- coding: utf-8 -*-
"""
Created on Thu May 21 00:11:03 2020

@author: fcosta
"""


from   Utils.Qlunc_ImportModules import *
from   Utils.Qlunc_Help_standAlone import flatten
from   Main.Qlunc_inputs import *
import UQ_Functions.Qlunc_UQ_Photonics_func
import UQ_Functions.Qlunc_UQ_Power_func
import UQ_Functions.Qlunc_UQ_Optics_func
import Utils.Qlunc_Help_standAlone as SA
#%% From the initial classes want to take the dictionaries and their content to loop over the input method names and their values

#input_values_LOOP=[]
#input_values_LOOP2={}
# Find the data want to loop over inside classes and nested classes:  
#This if is because have to calculate the figure noise befeore passing it as a int parameter instead a string ()
if 'Optical_amplifier_noise' in list(flatten(user_inputs.user_itype_noise)) and isinstance(inputs.photonics_inp.Optical_amplifier_inputs['Optical_amplifier_noise']['Optical_amplifier_NF'],str):
    inputs.photonics_inp.Optical_amplifier_inputs['Optical_amplifier_noise']['Optical_amplifier_NF']=UQ_Functions.Qlunc_UQ_Photonics_func.FigNoise(inputs,direct)
#pdb.set_trace()  
    
inputs_attributes=[atr for atr in dir(inputs) if inspect.getmembers(getattr(inputs,atr))]
inputs_attributes=list([a for a in inputs_attributes if not(a.startswith('__') and a.endswith('__'))]) # obtaining attributes from the class inputs 
inputs_attributes=inputs_attributes[3:] # Only take component values, not modules, atmospheric nor lidar general values
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
            Values2loop.append(list(LOOP_inputs_dict[index_loop0][index_loop1].values())) # Values names we want to loop over
            Names2loop.append(list(LOOP_inputs_dict[index_loop0][index_loop1].keys())) # Method names we want to loop over
Val=[None]*len(list(flatten(Names2loop))) # We need this to loop over without having to put all variables in the loop
Names2loop=list(flatten(Names2loop))


#Atmosphere and general data needed inputs we have to include in calculations but either dont want to loop over or the loop needs to be done on parallel (temperature and humidity)
Temp        = inputs.atm_inp.Atmospheric_inputs['temperature']
Hum         = inputs.atm_inp.Atmospheric_inputs['humidity']
Wave        = inputs.lidar_inp.Lidar_inputs['Wavelength']


# %%Getting scenarios:
def Get_Scenarios():
    # Initialazing variables:
    global Scenarios
    Scenarios=dict()
    TempCol=[]       
    Wavelength=[]
#    Temperature=None # initialize this values in None, as well as the values in 'add_typeN' to pass it as empty values to fill them in the loop when getting Scenarios!!!!!
#    Humidity=None

    #%%Loop to create typeN and Val depending on the user modules/components inputs):

    # Main loop to go over all variables to create the cases:############### 
    # FIGURE NOISE IS NOT INCLUDED IN THE SCENARIOS BECAUSE IS NOT NEEDED FOR ANY CALCULATION. IT IS JUST A 'NOISE' MORE TO ADD AT THE END, 
    # WHEN BUILDING THE DATA FRAME  

    
    for i in list(flatten('Temperature','Humidity',Names2loop)):
        Scenarios[i]=[]
    
    for VAL_Temp,VAL_Hum in zip (Temp, Hum): # This for lop is apart of the others because of the zip, since T and H shouldn´t be mixed to obtain different scenarions --> T and H are paired for each scenario
#      for VAL_WAVE,VAL_NOISE_FIG in zip(wave,amp_noise_figure):
        countt=0
        for Val in list(itertools.product(*list(itertools.chain(*Values2loop)))): # This makes all possible combinations among user inputs
                
            
           
            for k,v in zip (Scenarios.keys(), (VAL_Temp,)+(VAL_Hum,)+Val): # for loop to build up the dictionary 'Scenarios'. If user includes some variability (dependency on wavelength e.g. of any variable)              
                Scenarios[k].append(v)
            Wavelength.append(Wave[countt])
            countt+=1
#                Scenarios.append(list(flatten(inputs.VAL.VAL_T,inputs.VAL.VAL_H,inputs.VAL.VAL_WAVE,inputs.VAL.VAL_NOISE_FIG,Val))) #
            TempCol.append(VAL_Temp)
#    pdb.set_trace()
    return Scenarios,TempCol,Wavelength
#%% Running the different cases. If user has included it, the case is evaluated: Can I do this in a loop??????
    # User have to include here the module, component and estimation method
   
def Get_Noise(module,Wavelength,Scenarios):    
    METHODS={}
    if module == 'Power':
        Func = {'Power_source_noise' : Qlunc_UQ_Power_func.UQ_PowerSource,
                'Converter_noise'    : Qlunc_UQ_Power_func.UQ_Converter,
                'Converter_losses'   : Qlunc_UQ_Power_func.Losses_Converter,
                }
    
    elif module== 'Photonics':
        Func = {'Laser_source_noise'      : UQ_Functions.Qlunc_UQ_Photonics_func.UQ_LaserSource,
                'Photodetector_noise'     : UQ_Functions.Qlunc_UQ_Photonics_func.UQ_Photodetector,
                'Optical_amplifier_noise' : UQ_Functions.Qlunc_UQ_Photonics_func.UQ_Optical_amplifier 
                }
#        if 'Optical_amplifier' in list(SA.flatten(user_inputs.user_icomponents)): 
#            # For methods that we want them to appear in estimations although they´re not in the 'user_inputs.user_itype_noise'(user options) list, like the optical amplifier noise figure
#            # wich is estimated automatically when introducing the optical amplifier as a component and it is not involved in any calculations:
#            METHODS.setdefault('Optical_amplifier_fignoise',Qlunc_UQ_Photonics_func.FigNoise(user_inputs,inputs,direct,**Scenarios)) 
#    
    elif module=='Optics':
        Func = {'Telescope_noise'  : Qlunc_UQ_Optics_func.UQ_Telescope,
                'Telescope_losses' : Qlunc_UQ_Optics_func.Losses_Telescope
                }
           
    for k,v in Func.items(): # Loop to call functions whose module and method is included in user inputs
        if k in list(SA.flatten(user_inputs.user_itype_noise)):  
            METHODS.setdefault(k,list(SA.flatten(Func[k](user_inputs,inputs,cts,direct,Wavelength,**Scenarios)))) # 'Setdefault' is just like append but it's used when no element is yet included in the dictionary     
#    pdb.set_trace()
    return METHODS