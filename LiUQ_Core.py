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
import itertools	

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
    Res={}# Dictionary outcome stored in Res
    if 'amplifier' in modules:
            class amplifier(): # Create class amplifier
                def Amp_noise(self,Atmospheric_inputs,Amplifier_uncertainty_inputs): # Calculation of losses in amplifier
                    UQ_Amp=UQ_Hardware.UQ_Amplifier(Atmospheric_inputs,Amplifier_uncertainty_inputs)
                    return UQ_Amp
                def Amp_losses(self): # Calculation of losses in amplifier                   
                    self.amp_losses=[5,95]
                    return self.amp_losses
                def Amp_others(self):                    
                    self.amp_others=[6]
                    return self.amp_others
                def Amp_Failures(self):
                    self.ampli_failures=[23]
                    return self.ampli_failures    
            Obj=amplifier()#Create instance of object amplifier
            # Every calculation method ("def whatever...") included in "class amplifier()" should be added also in "RES['amplifier']" as a new dictionary key:value pair
            Res['amplifier']=({'Ampli_noise':Obj.Amp_noise(Atmospheric_inputs,Amplifier_uncertainty_inputs),'Ampli_losses':Obj.Amp_losses(),'Ampli_DELTA':Obj.Amp_others(),'Ampli_Failures':Obj.Amp_Failures()})# Creating a nested dictionary
    if 'photodetector' in modules:
            class photodetector():
                def Photo_noise(self,Atmospheric_inputs,Photodetector_uncertainty_inputs):
                    UQ_Photo=UQ_Hardware.UQ_Photodetector(Atmospheric_inputs,Photodetector_uncertainty_inputs)#function calculating amplifier uncertainties ((UQ_Photodetector.py))
                    return UQ_Photo   
                def Photo_losses(self):                   
                    self.photo_losses=[7]
                    return self.photo_losses
                def Photo_Failures(self):
                    self.photo_failures=[3]
                    return self.photo_failures               
            Obj=photodetector()
            Res['photodetector']=({'Photo_noise':Obj.Photo_noise(Atmospheric_inputs,Photodetector_uncertainty_inputs),'Photo_losses':Obj.Photo_losses(),'Photo_Failures':Obj.Photo_Failures()})                                         
    if 'telescope' in modules:
            class telescope():
                def Tele_noise(self,Atmospheric_inputs,Telescope_uncertainty_inputs):
                    UQ_Tele=UQ_Hardware.UQ_Telescope(Atmospheric_inputs,Telescope_uncertainty_inputs)#function calculating amplifier uncertainties ((UQ_Telescope.py))
                    return UQ_Tele
                def Tele_losses(self):                   
                    self.tele_losses=[31]
                    return self.tele_losses
                def Tele_others(self):                    
                    self.tele_others=[33]
                    return self.tele_others
                def Tele_Failures(self):                    
                    self.tele_failures=[234,2677]
                    return self.tele_failures   
            Obj=telescope()
            Res['telescope']=({'Tele_noise':Obj.Tele_noise(Atmospheric_inputs,Telescope_uncertainty_inputs),'Tele_losses':Obj.Tele_losses(),'Tele_DELTA':Obj.Tele_others(),'Tele_Failures':Obj.Tele_Failures()})

#Create H_UQ dictionary of values: 

H_Obj=Hardware_U()# HArdware instance
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

#Generate list of keys and values to loop over
Values_errors = [list(itertools.chain((H_UQ[ind_error_val].values()))) for ind_error_val in H_UQ.keys()]
Keys_errors   = [list(itertools.chain((H_UQ[ind_error_key].keys()))) for ind_error_key in H_UQ.keys()]
subindices=[]
subcolumns=[]
#Generate indexes of the data frame:
subindices=list((itertools.chain(*Keys_errors)))

subcolumns0=list(itertools.product(*Atmospheric_inputs.values(),Hardware_U().amplifier().Amp_losses(),Hardware_U().amplifier().Amp_others(),Hardware_U().amplifier().Amp_Failures(),
                                   Hardware_U().photodetector().Photo_losses(), Hardware_U().photodetector().Photo_Failures(),Hardware_U().telescope().Tele_losses(),Hardware_U().telescope().Tele_Failures(),
                                   Hardware_U().telescope().Tele_others()))
subcolumns=[str(subcolumns0[ind_str]) for ind_str in range(len(subcolumns0))]

#Flattening the error values not including noise errors because noise errors are not repeated for all the scenarios
Values_errors_removed=[list(itertools.product(*Values_errors[i][1:])) for i in range (len (Values_errors))] # values of the rest of errors (not related with atmospheric conditions) 
fl_Values_errors_removed=list(itertools.product(*Values_errors_removed))
fl2_Values_errors_removed=[list(itertools.chain(*fl_Values_errors_removed[Flat_ind])) for Flat_ind in range(len(fl_Values_errors_removed))]

#extract noise errors. Is [0] hardcoded because noise errors are always the first position of "Values_errors" list
Values_errors_noise=[Values_errors[i][0] for i in range (len (Values_errors))]
Values_errors_noise_DEF=list(map(list,list(zip(*Values_errors_noise))))
fl_Values_errors_removed=list(map(list,fl_Values_errors_removed))
ListFinalScen=[]
Final_Scenarios=[]
for indc in range(len(Values_errors_noise_DEF)):
    ListFinalScen.append(([list(zip(Values_errors_noise_DEF[indc],fl_Values_errors_removed[indc2])) for indc2 in range(len(fl_Values_errors_removed))]))
ListFinalScen=list(itertools.chain.from_iterable(ListFinalScen))
for indIter in range(len(ListFinalScen)):
    Final_Scenarios.append(list(itertools.chain(*(i if isinstance(i, tuple) else (i,) for i in list(itertools.chain(*(i if isinstance(i, tuple) else (i,) for i in ListFinalScen[indIter])))))))

df_UQ=pd.DataFrame(np.transpose(Final_Scenarios),index=subindices,columns=subcolumns)    



#in_dB=0
##Sum af decibels:
#for valrows in range(0,df_UQ.shape[0]):
#    in_dB+=(10**(df_UQ.iloc[valrows,0]/10))
#Sum_decibels=10*np.log10(in_dB)
#
##Transforming into watts
#df_UQ.loc['Total UQ','Hardware (dB)']= Sum_decibels# for now sum the uncertainties. Here have to apply Uncertainty expansion.
#
##transform in watts. We supose that raw data is in dB:
#df_UQ['Hardware (w)']=(10**(df_UQ['Hardware (dB)']/10))



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
