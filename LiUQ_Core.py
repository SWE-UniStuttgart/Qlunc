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


#%% Modules to import: 
from ImportModules import *

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

#inputs.modules() = [each_string.lower() for each_string in modules] #lower case
#DP      = [each_string.lower() for each_string in DP] #lower case


#Definig the lists we fill later on within the process;
H_UQ={}
DP_UQ=[] # To work on in the future

#subcolumns=[]# columns of the dataframe
subcolumnsComb=[]
subcolumns_NoneComb=[]
#%% Hardware:

Mod_Comp_Meth = {'power'     : {'power_source':UQ_Power_func.UQ_PowerSource,'converter':UQ_Power_func.UQ_Converter},
                 'photonics' : {'laser_source':UQ_Photonics_func.UQ_LaserSource,'photodetector':UQ_Photonics_func.UQ_Photodetector,
                                'amplifier'   :{'Amp_Noise':UQ_Photonics_func.UQ_Amplifier, 'FigNoise':UQ_Photonics_func.FigNoise}},
                 'optics'    : {'telescope':UQ_Optics_func.UQ_Telescope}}
MOD=list(Mod_Comp_Meth.keys())
class Hardware_U():  # creating a function to call each different module. HAve to add an if for a new module
    H_UQ={}# Dictionary outcome stored in Res
             
    if 'power' in MOD:
        class Power(): # Create class amplifier                   
            power_dic={k:v for k,v in Mod_Comp_Meth['power'].items()}                     
#            for i in range(len(MOD)):
            PowerMod_Methods = type('power',(),power_dic)    
        power        = Power.PowerMod_Methods
        H_UQ['power'] = {'power_source':power.power_source(inputs),
                        'converter'   :power.converter(inputs)}
    if 'photonics' in MOD:
        class Photonics(): # Create class amplifier
            photonics_dic={k:v for k,v in Mod_Comp_Meth['photonics'].items()}                     
#            for i in range(len(MOD)):
            PhotonicsMod_Methods = type('photonics',(),photonics_dic)   
        photonics = Photonics.PhotonicsMod_Methods
        H_UQ['photonics'] = {'laser_source'      : photonics.laser_source(inputs),
                            'photodetector'      : photonics.photodetector(inputs),
                            'amplifier_noise'    : photonics.amplifier['Amp_Noise'](inputs),
                            'amplifier_fignoise' : photonics.amplifier['FigNoise'](inputs)}
    if 'optics' in MOD:
        class Optics(): # Create class amplifier
            optics_dic={k:v for k,v in Mod_Comp_Meth['optics'].items()}                     
#            for i in range(len(MOD)):
            OpticsMod_Methods = type('optics',(),optics_dic)   
        optics = Optics.OpticsMod_Methods
        H_UQ['optics'] = {'telescope'      : optics.telescope(inputs)}
        


                    
#                    
#                    
#                    
#                    
#                    
#                    
#                    
#                    
#                    
#                
#                    
#                    
#                    def Amp_Noise(self,Atmospheric_inputs,Amplifier_uncertainty_inputs,Wavelength): # Run noise in amplifier device calculations
#                        self.NoiseFigure_VALUE=UQ_Hardware.FigNoise(Wavelength)
#                        self.UQ_Amp=UQ_Hardware.UQ_Amplifier(Atmospheric_inputs,Amplifier_uncertainty_inputs)
#                        return self.NoiseFigure_VALUE,self.UQ_Amp                                                                            
#                    def Amp_losses(self): # Calculation of losses in amplifier                   
#                        self.amp_losses=[0.6,0.8]
#                        return self.amp_losses
#                    def Amp_others(self):                    
#                        self.amp_others=[0]
#                        return self.amp_others
#                    def Amp_Failures(self):
#                        self.ampli_failures=[0.01]
#                        return self.ampli_failures 
#                Obj=amplifier()#Create instance of object amplifier
#                # Every calculation method ("def whatever...") included in "class amplifier()" should be added also in "RES['amplifier']" as a new dictionary key:value pair
#                # If the function (e.g. "Amp_Noise") contains different outcomes (e.g. "NoiseFigure_VALUE" and "UQ_Amp") we should classify them as combinatory or none combinatory elements.
#                # Atmosphere variations of different modules for different T, H, ... cannot be combined so should exist a none combinatory subcolumn
#                CombStuff,NoneCombStuff=Obj.Amp_Noise(Atmospheric_inputs,Amplifier_uncertainty_inputs,Wavelength) # Whether in a function there are combinatory and none combinatory elements
#                Res['amplifier']=({'Ampli_FN':CombStuff,
#                                   'Ampli_noise':NoneCombStuff,
#                                   'Ampli_losses':Obj.Amp_losses(),
#                                   'Ampli_DELTA':Obj.Amp_others(),
#                                   'Ampli_Failures':Obj.Amp_Failures()})# Creating a nested dictionary
#                subcolumnsComb.append([CombStuff, Res['amplifier']['Ampli_losses'],Res['amplifier']['Ampli_DELTA'],Res['amplifier']['Ampli_Failures']])
#                subcolumns_NoneComb.append([NoneCombStuff]) #variables can combine
#        if 'photodetector' in modules:
#                class photodetector():
#                    def Photo_noise(self,Atmospheric_inputs,Photodetector_uncertainty_inputs):
#                        UQ_Photo=UQ_Hardware.UQ_Photodetector(Atmospheric_inputs,Photodetector_uncertainty_inputs)#function calculating amplifier uncertainties ((UQ_Photodetector.py))
#                        return UQ_Photo   
#                    def Photo_losses(self):                   
#                        self.photo_losses=[1.1]
#                        return self.photo_losses
#                    def Photo_Failures(self):
#                        self.photo_failures=[0]
#                        return self.photo_failures               
#                Obj=photodetector()
#                Res['photodetector']=({'Photo_noise':Obj.Photo_noise(Atmospheric_inputs,Photodetector_uncertainty_inputs),
#                                       'Photo_losses':Obj.Photo_losses(),
#                                       'Photo_Failures':Obj.Photo_Failures()})                                         
#                subcolumnsComb.append([Res['photodetector']['Photo_losses'], 
#                                       Res['photodetector']['Photo_Failures']])
#                subcolumns_NoneComb.append([Res['photodetector']['Photo_noise']])
#        if 'telescope' in modules:
#                class telescope():
#                    def Tele_noise(self,Atmospheric_inputs,Telescope_uncertainty_inputs):
#                        UQ_Tele=UQ_Hardware.UQ_Telescope(Atmospheric_inputs,Telescope_uncertainty_inputs)#function calculating amplifier uncertainties ((UQ_Telescope.py))
#                        return UQ_Tele
#                    def Tele_losses(self):                   
#                        self.tele_losses=[0.8]
#                        return self.tele_losses
#                    def Tele_others(self):                    
#                        self.tele_others=[0.3]
#                        return self.tele_others
#                    def Tele_Failures(self):                    
#                        self.tele_failures=[1.2]
#                        return self.tele_failures   
#                Obj=telescope()
#                Res['telescope']=({'Tele_noise':Obj.Tele_noise(Atmospheric_inputs,Telescope_uncertainty_inputs),
#                                   'Tele_losses':Obj.Tele_losses(),
#                                   'Tele_DELTA':Obj.Tele_others(),
#                                   'Tele_Failures':Obj.Tele_Failures()})
#                subcolumnsComb.append([Res['telescope']['Tele_losses'],
#                                       Res['telescope']['Tele_Failures'],
#                                       Res['telescope']['Tele_DELTA']])
#                subcolumns_NoneComb.append([Res['telescope']['Tele_noise']])
#Create H_UQ dictionary of values: 
H_UQ=Hardware_U.H_UQ# HArdware instance


#If want to make fast calculations can apply: Hardware_U().amplifier().Amp_noise(25,20,5,.005)
    
#%% Data processing:
#for method in DP:
#    def Data_Processing_U(method=method):
#        if method=='los': 
#            UQ_LineOfSight= UQ_Data_processing.UQ_LOS() #function calculating amplifier uncertainties ((UQ_Amplifier.py))
#            return UQ_LineOfSight
#        elif method=='filtering_methods': 
#            UQ_Filtering= UQ_Data_processing.UQ_FilterMethods() #function calculating amplifier uncertainties ((UQ_Amplifier.py))
#            return UQ_Filtering
#    DP_UQ.append(Data_Processing_U(method=method))

#%% Create a complete data frame (Hardware+data processing uncertainties): 

#Generate list of keys and values to loop over
Values_errors = [list(itertools.chain((H_UQ[ind_error_val].values()))) for ind_error_val in H_UQ.keys()]
Keys_errors   = [list(itertools.chain((H_UQ[ind_error_key].keys()))) for ind_error_key in H_UQ.keys()]

#Generate indexes and columns of the data frame:
subindices = list((itertools.chain(*Keys_errors)))
subcolumns_NoneComb=list(zip(*list(itertools.chain(*subcolumns_NoneComb))))
subcolumnsComb=list(itertools.product(*list(itertools.chain(*subcolumnsComb))))


Scenarios=[subcolumns_NoneComb]+[subcolumnsComb]
FinalScenarios=list(itertools.product(*Scenarios))
FinalScenarios=[list(itertools.chain(*FinalScenarios[i])) for i in range(len(FinalScenarios))]
#Columns=[str(list(itertools.chain(*subcolumns[i]))) for i in range(len(subcolumns))]




df_UQ=pd.DataFrame(np.transpose(FinalScenarios), index=subindices)


#subcolumns = list(itertools.chain(itertools.product(list(itertools.product(*list(itertools.chain(*list([Atmospheric_inputs.values()]))))),list(itertools.product(*list(itertools.chain(*subcolumns)))))))
#subcolumns = [str(list(itertools.chain(*subcolumns [indSub]))) for indSub in range(len(subcolumns))]

#Flattening the error values not including noise errors because noise errors are not repeated for all the scenarios
#Values_errors_removed    = [list(itertools.product(*Values_errors[i][1:])) for i in range (len (Values_errors))] # values of the rest of errors (not related with atmospheric conditions) 
#fl_Values_errors_removed = list(itertools.product(*Values_errors_removed)) #Flatted values errors removed
#
##extract noise errors. Is [0] hardcoded because noise errors are always the first position of "Values_errors" list
#Values_errors_noise      = [Values_errors[i][0:1] for i in range (len (Values_errors))]
#Values_errors_noise_DEF  = list(map(list,list(zip(*Values_errors_noise))))
#fl_Values_errors_removed = list(map(list,fl_Values_errors_removed))
#List_Scenarios           = []
#Final_Scenarios          = []
#for indc in range(len(Values_errors_noise_DEF)):
#    List_Scenarios.append(([list(zip(Values_errors_noise_DEF[indc],fl_Values_errors_removed[indc2])) for indc2 in range(len(fl_Values_errors_removed))]))
#List_Scenarios = list(itertools.chain.from_iterable(List_Scenarios))
#for indIter in range(len(List_Scenarios)):
#    Final_Scenarios.append(list(itertools.chain(*(i if isinstance(i, tuple) else (i,) for i in list(itertools.chain(*(i if isinstance(i, tuple) else (i,) for i in List_Scenarios[indIter])))))))
#
#df_UQ=pd.DataFrame(np.transpose(Final_Scenarios),index=subindices,columns=subcolumns)    

#Sum af decibels:
in_dB=0
Sum_decibels= []
for valcols in range(0,df_UQ.shape[1]):
    Sum_in_dB     = sum([(10**(df_UQ.iloc[valrows,valcols]/10)) for valrows in range(0,df_UQ.shape[0])])
#    Sum_in_dB = sum(in_dB)
    Sum_decibels.append(10*np.log10(Sum_in_dB) )


df_UQ.loc['Total UQ']= Sum_decibels# for now sum the uncertainties. Here have to apply Uncertainty expansion.

#transform in watts. We supose that raw data is in dB:
#df_UQ['Hardware (w)']=(10**(df_UQ['Hardware (dB)']/10))



## GUI Stuff ############################################
#with open ('DF.pickle','wb') as DATAFRAME:
#    pickle.dump([df_H_UQ,df_DP_UQ,df_UQ],DATAFRAME)
#######################################################

#%% Plotting:
flag_plot_signal_noise=True
if flag_plot_signal_noise==True: #Introduce this flag in the gui    
    #Create original received power signal in watts (for now this is necessary as far as we dont have outgoing signal from lidar):
    t           = np.linspace(0,100,1000)
    O_signal_W  = (6*np.sin(t/(2*np.pi)))**2 #original in w (**2 because is supoused original signal are volts)
    O_signal_dB = 10*np.log10(O_signal_W) # original in dB
    
    
    #adding Hardware noise for all different scenarios
    ind_Scen=2 # Chooosing scenario (arranged as columns in the dataframe)
    noise_H_dB=[df_UQ.iloc[ind_Param,ind_Scen] for ind_Param in range(np.shape(df_UQ)[0]-1)] # in dB. Here we can change scenario changing the index 'ind_Scen'. This goes throw the columns of the df_UQ
    noise_H_W=[10**(noise_H_dB[i]/10)  for i in range (len(noise_H_dB)) ]#convert into watts    
    mean_noise=0
    stdv=np.sqrt(noise_H_W)
    noise_W=[np.random.normal(mean_noise,stdv[ind_stdv],len(O_signal_W)) for ind_stdv in range(len(stdv)) ]#create normal noise signal centered in 0 and stdv
    Noisy_signal_W=[O_signal_W+noise_W[ind_fociflama] for ind_fociflama in range(len(noise_W))] # Add noise to the original signal [w]
    Noisy_signal_dB=[10*np.log10(Noisy_signal_W[in_noise]) for in_noise in range(len(Noisy_signal_W))]# Converting noise in watts to dB
    
    # Total noise: 
    stdvTotal=np.sqrt(noise_H_W)
    noise_W_Total=np.random.normal(mean_noise,df_UQ.iloc[-1,ind_Scen],len(O_signal_W))#create normal noise signal centered in 0 and stdv
    Total_Noise_signal_W=O_signal_W+noise_W_Total# Add noise to the original signal [w]
    Total_Noise_signal_dB=10*np.log10(Total_Noise_signal_W)# Converting noise in watts to dB
    
    
    #Plotting:
    plt.figure()
    plt.plot(t,O_signal_W)
    plt.plot(t,Total_Noise_signal_W,'r-') 
    plt.title('Signal + noise')
    plt.gcf().canvas.set_window_title('Signal_Watts')
    plt.xlabel('time [s]')
    plt.ylabel('power intensity [w]')
    
    plt.show() #original + noise (w) 
    for ind_plot_W in [3]:
        plt.plot(t,Noisy_signal_W[ind_plot_W],'g--') 
        plt.legend(['original','Total Noise','Figure_noise'])#,'Total error [w]'])
    
    plt.show() 
        
    #Plotting original signal in  (dB)
    plt.figure()
    plt.plot(t,O_signal_dB)
    plt.title('Signal + noise')
    plt.gcf().canvas.set_window_title('Signal_dB')
    plt.ylabel('power intensity [dB]')
    plt.xlabel('time [s]')
    plt.plot(t,Total_Noise_signal_dB,'r-') 

    # original + noise (dB)        
    for ind_plot_dB in [3]:
        plt.plot(t,Noisy_signal_dB[ind_plot_dB],'g--')        
        
        plt.legend(['original','Total Noise','Figure_noise'])#,'Total error [w]'])
    
    plt.show()  
