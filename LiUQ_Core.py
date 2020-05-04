# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:44:25 2020

@author: fcosta
"""
import pandas as pd
import UQ_Hardware  # script with all calculations of hardware unc are done
import UQ_Data_processing # script with all calculations of data processing methods unc are done
import numpy as np
import pdb
import pickle	

with open('I_D.pickle', 'rb') as c_data:
    ImpDATA = pickle.load(c_data)
    modules=ImpDATA[0]
    DP=ImpDATA[1]
    temperature=ImpDATA[2]
    humidity=ImpDATA[3]
    noise_amp=ImpDATA[4]
    o_c_amp=ImpDATA[5]
    o_c_photo=ImpDATA[6]
    noise_photo=ImpDATA[7]
    curvature_lens=ImpDATA[8]
    o_c_tele=ImpDATA[9]
    aberration=ImpDATA[10]


modules = [each_string.lower() for each_string in modules] #lower case
DP      = [each_string.lower() for each_string in DP] #lower case


#Definig the lists we fill later on within the process;
H_UQ=[]
DP_UQ=[]

#%% Hardware:
for module in modules:
    def Hardware_U(module=module):  # creating a function to call each different module. HAve to add an if for a new module
        if module=='amplifier': 
            UQ_Amp= UQ_Hardware.UQ_Amplifier(temperature,humidity,noise_amp,o_c_amp) #function calculating amplifier uncertainties ((UQ_Amplifier.py))
            return UQ_Amp            
        elif module== 'photodetector':
            UQ_Photo=UQ_Hardware.UQ_Photodetector(temperature,humidity,noise_photo,o_c_photo)#function calculating amplifier uncertainties ((UQ_Photodetector.py))
            return UQ_Photo           
        elif module=='telescope':
            UQ_Tele=UQ_Hardware.UQ_Telescope(temperature,humidity,curvature_lens,aberration,o_c_tele)#function calculating amplifier uncertainties ((UQ_Telescope.py))
            return UQ_Tele
    H_UQ.append(Hardware_U(module=module))
    
#%% Data processing:
for method in DP:
    def Data_Processing_U(method=method):
        if method=='los': 
            UQ_LineOfSight= UQ_Data_processing.UQ_LOS() #function calculating amplifier uncertainties ((UQ_Amplifier.py))
            return UQ_LineOfSight
        elif method=='filtering_methods': 
            UQ_Filtering= UQ_Data_processing.UQ_FilterMethods() #function calculating amplifier uncertainties ((UQ_Amplifier.py))
            return UQ_Filtering
    DP_UQ.append(Data_Processing_U(method=method))

#%% Create a complete data frame (Hardware+data processing uncertainties): 

df_H_UQ=pd.DataFrame(H_UQ,columns=['Hardware (dB)'],index=modules) # Create a data frame with the calcutlated hardware uncertainties 
df_DP_UQ=pd.DataFrame(DP_UQ,columns=['Data Processing (dB)'], index=DP)
df_UQ=df_H_UQ.append(df_DP_UQ,ignore_index=False,sort=True)#concatenate df´s

#Sum af decibels:
in_dB=0
for valrows in range(df_UQ.shape[0]):
    in_dB+=(10**(df_UQ.iloc[valrows,1]/10))
Sum_decibels=10*np.log10(in_dB)

#Transforming into watts
df_UQ.loc['Total UQ','Hardware (dB)']= Sum_decibels# for now sum the uncertainties. Uncertainty expansion.
df_UQ['Hardware (w)']=(10**(df_UQ['Hardware (dB)']/10)) #convert uncertainties from dB to watts.
with open ('DF.pickle','wb') as DATAFRAME:
    pickle.dump([df_H_UQ,df_DP_UQ,df_UQ],DATAFRAME)
