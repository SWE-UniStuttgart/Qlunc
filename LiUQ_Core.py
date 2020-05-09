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
                    global amp_losses
                    self.amp_losses=[5,6]
                    return self.amp_losses
                def Amp_others(self):
                    global amp_others
                    self.amp_others=[7,8]
                    return self.amp_others
            Obj=amplifier()#Create instance of object amplifier
            Res['amplifier']=({'amplifier_noise':Obj.Amp_noise(Atmospheric_inputs,Amplifier_uncertainty_inputs),'amplifier_losses':Obj.Amp_losses(),'amplifier_DELTA':Obj.Amp_others()})# Creating a nested dictionary
    if 'photodetector' in modules:
            class photodetector():
                def Photo_noise(self,Atmospheric_inputs,Photodetector_uncertainty_inputs):
                    UQ_Photo=UQ_Hardware.UQ_Photodetector(Atmospheric_inputs,Photodetector_uncertainty_inputs)#function calculating amplifier uncertainties ((UQ_Photodetector.py))
                    return UQ_Photo   
                def Photo_losses(self):
                    global photo_losses
                    self.photo_losses=[9]
                    return self.photo_losses
                def Photo_PACOEffects(self):
                    global photo_PacoEffects
                    self.photo_PacoEffects=[10]
                    return self.photo_PacoEffects                
            Obj=photodetector()
            Res['photodetector']=({'photodetector_noise':Obj.Photo_noise(Atmospheric_inputs,Photodetector_uncertainty_inputs),'photodetector_losses':Obj.Photo_losses(),'photodetector_PAcoEffects':Obj.Photo_PACOEffects()})
                                         
    if 'telescope' in modules:
            class telescope():
                def Tele_noise(self,Atmospheric_inputs,Telescope_uncertainty_inputs):
                    UQ_Tele=UQ_Hardware.UQ_Telescope(Atmospheric_inputs,Telescope_uncertainty_inputs)#function calculating amplifier uncertainties ((UQ_Telescope.py))
                    return UQ_Tele
                def Tele_losses(self):
                    global tele_losses
                    self.tele_losses=[1]
                    return self.tele_losses
                def Tele_others(self):
                    global tele_others
                    self.tele_others=[33]
                    return self.tele_others
            Obj=telescope()
            Res['telescope']=({'telescope_noise':Obj.Tele_noise(Atmospheric_inputs,Telescope_uncertainty_inputs),'telescope_losses':Obj.Tele_losses(),'telescope_DELTA':Obj.Tele_others()})

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
Values_errors = [list(itertools.chain((H_UQ[i].values()))) for i in H_UQ.keys()]
Keys_errors   = [list(itertools.chain((H_UQ[i].keys()))) for i in H_UQ.keys()]
subindices=[]
subcolumns=[]
#Generate indexes of the data frame:
subindices=list((itertools.chain(*Keys_errors)))
# Generating columns of the data frame:
subcolumns=[('T='+str(temp)+'; H='+str(hum)+'; Rain:'+str(rain)+'; Fog:'+str(fog) +'; error1='+str(error1)+'; error2='+ str(error2)+'; error3='+ str(error3)) for temp in Atmospheric_inputs['temperature'] for hum in Atmospheric_inputs['humidity'] for rain in Atmospheric_inputs['rain'] for fog in Atmospheric_inputs['fog'] for error1 in H_UQ['amplifier']['amplifier_losses'] for error2 in H_UQ['amplifier']['amplifier_DELTA'] for error3 in H_UQ['photodetector']['photodetector_losses'] ]

Values_errors_removed=[list(itertools.product(*Values_errors[i][1:])) for i in range (len (Values_errors))] # values of the rest of errors (not related with atmospheric conditions) 
fl_Values_errors_removed=list(itertools.product(*Values_errors_removed))
fl2_Values_errors_removed=[list(itertools.chain(*fl_Values_errors_removed[Flat_ind])) for Flat_ind in range(len(fl_Values_errors_removed))]
Values_errors_noise=[Values_errors[i][0] for i in range (len (Values_errors))]
funcScenarios=lambda x:[x[i1][i2] for i1 in range(len(x))] 
atmospheric_scenarios=[]
for i2 in range(len(Values_errors_noise[0])):
    atmospheric_scenarios.append(funcScenarios(Values_errors_noise))
FinalScenarios0=list((itertools.product(atmospheric_scenarios,fl2_Values_errors_removed))) # Different scenarios for the different inputs (atmospheric-)
FinalScenarios=[list(itertools.chain(*FinalScenarios0[ind_Scenario])) for ind_Scenario in range(len(FinalScenarios0)) ]




#finalScenariostotal=list(itertools.chain(finalScenarios))

df_UQ=pd.DataFrame(np.transpose(FinalScenarios),index=subindices,columns=subcolumns)    








































#Split_Val_err_mod=[]
#for ind_val_err in range(len(Values_errors)):
#    Val_err_mod=Values_errors[ind_val_err]
#    Split_Val_err_mod.append(list(itertools.product(*Val_err_mod))) # Split the module errors of the atmospheric scenarios
#    
#df_UQ=pd.DataFrame(index=subindices,columns=subcolumns)    
#
#datacount=0
#count2=0
#
#t=[len(H_UQ[i]) for i in H_UQ.keys()] # numer of key elements for the different modules (info about how many noise elements do we have per lidar module)
#
#
#for ind3 in range(t[count2]):
#    for ind2 in range(len(Split_Val_err_mod[ind3][0])):
#        for ind1 in range(len(Split_Val_err_mod[ind3])):
#            df_UQ.iloc[datacount,ind1]=(Split_Val_err_mod[ind3][ind1][ind2]) #allocate each value
#        datacount+=1
#    count2+=1







#Generate the Data frame appending data frames from each module:
#countt=0
#for ind in range(len(subindices)):
#    for ind2 in range(len(subcols)):
#        df_UQ.iloc[ind, ind2]=Values_errors[0][0][0]
#        countt+=1
#
#
#
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
