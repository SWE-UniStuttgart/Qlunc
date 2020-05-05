# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:44:25 2020

@author: fcosta
"""
#Header:
#04272020 - Francisco Costa
#SWE - Stuttgart

#Calculation Hardware uncertainties:
#This code calculates uncertainties related with hardware in lidar devices, classifying them in different classes which are the different hardware modules the lidar is divided in.

#Definition of classes:
#Amplifier: figure noise.

#%% Modules to import: 
import pandas as pd
import UQ_Hardware  # script with all calculations of hardware unc are done
import UQ_Data_processing # script with all calculations of data processing methods unc are done
import numpy as np
import pdb
import pickle	

#%% Read data from the GUI script:#######################
#with open('I_D.pickle', 'rb') as c_data:
#    ImpDATA = pickle.load(c_data)
#    modules=ImpDATA[0]
#    DP=ImpDATA[1]
#    temperature=ImpDATA[2]
#    humidity=ImpDATA[3]
#    noise_amp=ImpDATA[4]
#    o_c_amp=ImpDATA[5]
#    o_c_photo=ImpDATA[6]
#    noise_photo=ImpDATA[7]
#    curvature_lens=ImpDATA[8]
#    o_c_tele=ImpDATA[9]
#    aberration=ImpDATA[10]
#########################################################

modules = [each_string.lower() for each_string in modules] #lower case
DP      = [each_string.lower() for each_string in DP] #lower case


#Definig the lists we fill later on within the process;
H_UQ=[]
DP_UQ=[]

#%% Hardware:

class Hardware_U():  # creating a function to call each different module. HAve to add an if for a new module
        if 'amplifier' in modules:
            class Amplifier():
                def Amp_noise(temperature,humidity,noise_amp,o_c_amp) :
                    UQ_Amp= UQ_Hardware.UQ_Amplifier(temperature,humidity,noise_amp,o_c_amp) #function calculating amplifier uncertainties ((UQ_Amplifier.py))
                    return UQ_Amp
        if 'photodetector' in modules:
            class Photodetector():
                def Photo_noise(temperature,humidity,noise_photo,o_c_photo):
                    UQ_Photo=UQ_Hardware.UQ_Photodetector(temperature,humidity,noise_photo,o_c_photo)#function calculating amplifier uncertainties ((UQ_Photodetector.py))
                    return UQ_Photo                        
        if 'telescope' in modules:
            class Telescope():
                def Tele_noise(temperature,humidity,curvature_lens,aberration,o_c_tele):
                    UQ_Tele=UQ_Hardware.UQ_Telescope(temperature,humidity,curvature_lens,aberration,o_c_tele)#function calculating amplifier uncertainties ((UQ_Telescope.py))
                    return UQ_Tele

Amp=Hardware_U.Amplifier#create an amplifier object in hardware class
#Create H_UQ list of values:
H_UQ.append(Hardware_U.Amplifier.Amp_noise(temperature,humidity,noise_amp,o_c_amp))
H_UQ.append(Hardware_U.Telescope.Tele_noise(temperature,humidity,curvature_lens,aberration,o_c_tele))
H_UQ.append(Hardware_U.Photodetector.Photo_noise(temperature,humidity,noise_photo,o_c_photo))
    
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
df_UQ.loc['Total UQ','Hardware (dB)']= Sum_decibels# for now sum the uncertainties. Here have to apply Uncertainty expansion.
df_UQ['Hardware (w)']=(10**(df_UQ['Hardware (dB)']/10)) #convert uncertainties from dB to watts.

## GUI Stuff ############################################
#with open ('DF.pickle','wb') as DATAFRAME:
#    pickle.dump([df_H_UQ,df_DP_UQ,df_UQ],DATAFRAME)
#######################################################
#%% Plotting:

#if flag_plot_signal_noise==True: Introduce this flag in the gui
    
#    #Create original received power signal in watts (for now this is necessary):
#    t           = np.linspace(0,100,1000)
#    O_signal_W  = (10*np.sin(t/(2*np.pi)))**2 #original in w (**2 because is supoused original signal are volts)
#    O_signal_dB = 10*np.log10(O_signal_W) # original in dB
#
#    #adding Hardware noise
#    
#    noise_H_dB=df_H_UQ.loc['Total UQ','HarDwaRE (dB)'] # in dB
#    noise_H_W=10**(noise_H_dB/10) #convert into watts    
#    mean_noise=0
#    stdv=np.sqrt(noise_H_W)
#    noise_W=np.random.normal(mean_noise,stdv,len(O_signal_W)) #add normal noise centered in 0 and stdv
#    Noisy_signal_W=O_signal_W+noise_W
#    Noisy_signal_dB=O_signal_dB+10*np.log10(Noisy_signal_W)
#    
#    #Plotting:
#    
#    #original (w)
#    plt.subplot(2,2,1)
#    plt.plot(t,O_signal_W)
#    plt.title('Signal')
#    plt.ylabel('power intensity [w]')
#    plt.show()    
#    #original (dB)
#    plt.subplot(2,2,3)
#    plt.plot(t,O_signal_dB)
#    plt.ylabel('power intensity [dB]')
#    plt.xlabel('time [s]')
#    plt.show()
#    #original + noise (w)
#    plt.subplot(2,2,2)
#    plt.plot(t,Noisy_signal_W)
#    plt.title('Signal+noise')
#    plt.show() 
#    #original + noise (dB)    
#    plt.subplot(2,2,4)
#    plt.plot(t,Noisy_signal_dB)
#    plt.xlabel('time [s]')   
#    plt.show()  
