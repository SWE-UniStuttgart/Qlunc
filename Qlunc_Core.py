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
from Qlunc_ImportModules import *
import Qlunc_Wrapper as QW
import time
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

t=time.time()
#Definig the lists we fill later on within the process;
H_UQ_Power={}
H_UQ_Photonics={}
H_UQ_Optics={}
DP_UQ={} # To work on in the future

#subcolumns=[]# columns of the dataframe
subcolumnsComb=[]
subcolumns_NoneComb=[]
#%% Hardware:

#Defining classes:
class Hardware_U():  # creating a function to call each different module. HAve to add an if for a new module    
     # outcome dictionary
    H_UQ=pd.DataFrame()            
    if 'Power' in inputs.modules:             
        class Power(): # Create class optical_amplifier
            if 'Scenarios' not in globals():  # This 'if' decreases drastically computational time              
                Scenarios,DF_columns = QW.Get_Scenarios()# Temperature is calculated once and it is to get columns of dataframe
                H_UQ_Power           = QW.Get_Noise('Power',Scenarios)
            else:
                 H_UQ_Power          = QW.Get_Noise('Power',Scenarios)
        H_UQ_POWER     = SA.Get_DataFrame(Power.H_UQ_Power,Power.DF_columns).T
        H_UQ=H_UQ.append(H_UQ_POWER)
    
    if 'Photonics' in inputs.modules:
        class Photonics(): # Create class Photonics 
            
            if 'Scenarios' not in globals():                    
                Scenarios,DF_columns     = QW.Get_Scenarios()# Temperature is calculated once and it is to get columns of dataframe
                H_UQ_Photonics           = QW.Get_Noise('Photonics',Scenarios)
            else:
                 H_UQ_Photonics          = QW.Get_Noise('Photonics',Scenarios)
        H_UQ_PHOTONICS = SA.Get_DataFrame(Photonics.H_UQ_Photonics,Photonics.DF_columns).T
        H_UQ=H_UQ.append(H_UQ_PHOTONICS)
    
    if 'Optics' in inputs.modules:
       class Optics(): # Create class Optics            
            if 'Scenarios' not in globals():                    
                Scenarios,DF_columns  = QW.Get_Scenarios()# Temperature is calculated once and it is to get columns of dataframe
                H_UQ_Optics           = QW.Get_Noise('Optics',Scenarios)
            else:
                 H_UQ_Optics          = QW.Get_Noise('Optics',Scenarios)
       H_UQ_OPTICS    = SA.Get_DataFrame(Optics.H_UQ_Optics,Optics.DF_columns).T
       H_UQ=H_UQ.append(H_UQ_OPTICS)

# Creating th dataframe  
H_UQ=Hardware_U.H_UQ                        
#H_UQ=   H_UQ_POWER.append([H_UQ_PHOTONICS,H_UQ_OPTICS])# Total DataFrame                       

elapsed_time=time.time()-t
print('Elapsed time =', elapsed_time, 's...')

## GUI Stuff ############################################
#with open ('DF.pickle','wb') as DATAFRAME:
#    pickle.dump([df_H_UQ,df_DP_UQ,df_UQ],DATAFRAME)
#######################################################

#%% Plotting:
flag_plot_signal_noise=False
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
    
#    'Plotting time series'
    plt.figure()
    plt.plot(inputs.atm_inp.Atmospheric_inputs['time'],df_UQ.loc['Total UQ',:])
    plt.plot(inputs.atm_inp.Atmospheric_inputs['time'],inputs.atm_inp.Atmospheric_inputs['temperature'])
    plt.plot(inputs.atm_inp.Atmospheric_inputs['time'],inputs.atm_inp.Atmospheric_inputs['humidity'])
    
    
    
    plt.show()  
