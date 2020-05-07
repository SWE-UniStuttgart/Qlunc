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
H_UQ={}
DP_UQ=[]

#%% Hardware:

class Hardware_U():  # creating a function to call each different module. HAve to add an if for a new module
    Res={}
    if 'amplifier' in modules:
            class amplifier(): # Create class amplifier
                def Amp_noise(self,temperature,humidity,noise_amp,o_c_amp): # Calculation of losses in amplifier
                    self.UQ_Amp= UQ_Hardware.UQ_Amplifier(temperature,humidity,noise_amp,o_c_amp) #function calculating amplifier uncertainties ((UQ_Amplifier.py))
                    return self.UQ_Amp
                def Amp_losses(self): # Calculation of losses in amplifier
                    self.amp_losses=0
                    return self.amp_losses
            Obj=amplifier()#Create instance of object amplifier
            Res['amplifier']=({'noise':Obj.Amp_noise(temperature,humidity,noise_amp,o_c_amp),'losses':Obj.Amp_losses()})# Creating a nested dictionary
    if 'photodetector' in modules:
            class photodetector():
                def Photo_noise(self,temperature,humidity,noise_photo,o_c_photo):
                    UQ_Photo=UQ_Hardware.UQ_Photodetector(temperature,humidity,noise_photo,o_c_photo)#function calculating amplifier uncertainties ((UQ_Photodetector.py))
                    return UQ_Photo   
                def Photo_losses(self):
                    self.photo_losses=0
                    return self.photo_losses
            Obj=photodetector()
            Res['photodetector']=({'noise':Obj.Photo_noise(temperature,humidity,noise_photo,o_c_photo),'losses':Obj.Photo_losses()})
                                         
    if 'telescope' in modules:
            class telescope():
                def Tele_noise(self,temperature,humidity,curvature_lens,aberration,o_c_tele):
                    UQ_Tele=UQ_Hardware.UQ_Telescope(temperature,humidity,curvature_lens,aberration,o_c_tele)#function calculating amplifier uncertainties ((UQ_Telescope.py))
                    return UQ_Tele
                def Tele_losses(self):
                    self.tele_losses=0
                    return self.tele_losses
                def Tele_others(self):
                    self.tele_others=0
                    return self.tele_others
            Obj=telescope()
            Res['telescope']=({'noise':Obj.Tele_noise(temperature,humidity,curvature_lens,aberration,o_c_tele),'losses':Obj.Tele_losses(),'others':Obj.Tele_others()})

#Create H_UQ dictionary of values: 

H_Obj=Hardware_U()
for i in modules:       
    H_UQ[i]=(H_Obj.Res[i])
#    count_index+=1
#If want to make fast calculations can apply: Hardware_U().amplifier().Amp_noise(25,20,5,.005)
    
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
Values_modules=list(H_UQ.values())
Keys_modules=list(H_UQ.keys())
Values_errors=[]
Keys_errors=[]
#Generate list of keys and values to loop over
for i in range(len(Values_modules)):
    Values_errors.append(list(Values_modules[i].values()))
    Keys_errors.append(list(Values_modules[i].keys()))
df_UQ= pd.DataFrame()
keyCounter=0    
#Loop over keys and values to generate the data frame:
for key in Keys_modules:
    subindices=[]
    #Generate indexes of the data frame:     
    for index in Keys_errors[keyCounter]:
        subindices.append(key + ' ' + index)
    #Generate the Data frame appending data frames from each module:
    df_UQ=df_UQ.append(pd.DataFrame(Values_errors[keyCounter],columns=['Hardware (dB)'],index=subindices))
    keyCounter+=1   

in_dB=0
#Sum af decibels:
for valrows in range(df_UQ.shape[0]):
    in_dB+=(10**(df_UQ.iloc[valrows,0]/10))
Sum_decibels=10*np.log10(in_dB)

#Transforming into watts
df_UQ.loc['Total UQ','Hardware (dB)']= Sum_decibels# for now sum the uncertainties. Here have to apply Uncertainty expansion.

#transform in watts. We supose that raw data is in dB:
df_UQ['Hardware (w)']=(10**(df_UQ['Hardware (dB)']/10))



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
