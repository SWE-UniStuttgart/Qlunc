# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:51:36 2020

@author: fcosta
"""

from Utils.Qlunc_ImportModules import *
import Main.Qlunc_inputs
import Utils.Qlunc_Help_standAlone as SA
#%% PHOTODETECTOR
def UQ_Photodetector(user_inputs,inputs,cts,direct,Wavelength,**Scenarios):
    UQ_Photo=[]
    global Photodetector_SNR_thermal_noise,Photodetector_Thermal_noise,Photodetector_Shot_noise,Photodetector_SNR_Shot_noise,Photodetector_Dark_current_noise,Photodetector_SNR_DarkCurrent,Photodetector_TIA_noise,Photodetector_SNR_TIA
    UQ_Photodetector.Photodetector_SNR_thermal_noise=[]
    Photodetector_Thermal_noise=[]
    Photodetector_Shot_noise=[]
    UQ_Photodetector.Photodetector_SNR_Shot_noise=[]
    Photodetector_Dark_current_noise=[]
    UQ_Photodetector.Photodetector_SNR_DarkCurrent=[]
    Photodetector_TIA_noise=[]
    UQ_Photodetector.Photodetector_SNR_TIA=[]
    UQ_Photodetector.Responsivity=[]
    for ind_UQ_PHOTO in range(len(Scenarios.get('Temperature'))): # To loop over all Scenarios
#        UQ_photodetector.append(Scenarios.get('VAL_T')[i]*0.4+Scenarios.get('VAL_H')[i]*0.1+Scenarios.get('VAL_NOISE_PHOTO')[i]+Scenarios.get('VAL_OC_PHOTO')[i]+Scenarios.get('VAL_WAVE')[i]/1000)
        R = Scenarios.get('Photodetector_Efficiency')[ind_UQ_PHOTO]*cts.e*Wavelength[ind_UQ_PHOTO]/(cts.h*cts.c)  #[W/A]  Responsivity
        UQ_Photodetector.Responsivity.append(R) # this notation allows me to get Responsivity from outside of the function 
        # Photodetector Thermal noise
        Photodetector_Thermal_noise.append((10*np.log10(4*cts.k*Scenarios.get('Temperature')[ind_UQ_PHOTO]/Scenarios.get('Photodetector_RL')[ind_UQ_PHOTO]*Scenarios.get('Photodetector_Bandwidth')[ind_UQ_PHOTO]))) #[dBm]
        UQ_Photodetector.Photodetector_SNR_thermal_noise.append(10*np.log10(((R**2)/(4*cts.k*Scenarios.get('Temperature')[ind_UQ_PHOTO]*Scenarios.get('Photodetector_Bandwidth')[ind_UQ_PHOTO]/Scenarios.get('Photodetector_RL')[ind_UQ_PHOTO]))*(Scenarios.get('Photodetector_Signal_power')[ind_UQ_PHOTO]/1000)**2))
    
        #Photodetector shot noise:
        Photodetector_Shot_noise.append(10*np.log10(2*cts.e*R*Scenarios.get('Photodetector_Bandwidth')[ind_UQ_PHOTO]*Scenarios.get('Photodetector_Signal_power')[ind_UQ_PHOTO]))
        UQ_Photodetector.Photodetector_SNR_Shot_noise.append(10*np.log10(((R**2)/(2*cts.e*R*Scenarios.get('Photodetector_Bandwidth')[ind_UQ_PHOTO]))*(Scenarios.get('Photodetector_Signal_power')[ind_UQ_PHOTO])/1000))

        #Photodetector dark current noise
        Photodetector_Dark_current_noise.append(10*np.log10(2*cts.e*Scenarios.get('Photodetector_DarkCurrent')[ind_UQ_PHOTO]*Scenarios.get('Photodetector_Bandwidth')[ind_UQ_PHOTO]*Scenarios.get('Photodetector_Signal_power')[ind_UQ_PHOTO]))
        UQ_Photodetector.Photodetector_SNR_DarkCurrent.append(10*np.log10(((R**2)/(2*cts.e*Scenarios.get('Photodetector_DarkCurrent')[ind_UQ_PHOTO]*Scenarios.get('Photodetector_Bandwidth')[ind_UQ_PHOTO]))*((Scenarios.get('Photodetector_Signal_power')[ind_UQ_PHOTO]/1000)**2) ))

        if 'TIA_noise' in list(SA.flatten(user_inputs.user_itype_noise)):     # If TIA is included in the components have to include in the noise calculations:
            # Photodetector TIA noise
            Photodetector_TIA_noise.append( 10*np.log10(Scenarios.get('V_noise_TIA')[ind_UQ_PHOTO]**2/Scenarios.get('Gain_TIA')[ind_UQ_PHOTO]**2))
            UQ_Photodetector.Photodetector_SNR_TIA.append(10*np.log10(((R**2)/(Scenarios.get('V_noise_TIA')[ind_UQ_PHOTO]**2/Scenarios.get('Gain_TIA')[ind_UQ_PHOTO]**2))*(Scenarios.get('Photodetector_Signal_power')[ind_UQ_PHOTO]/1000)**2))
            UQ_Photo.append(SA.Sum_dB([Photodetector_Thermal_noise[ind_UQ_PHOTO],Photodetector_Shot_noise[ind_UQ_PHOTO],Photodetector_Dark_current_noise[ind_UQ_PHOTO],Photodetector_TIA_noise[ind_UQ_PHOTO]]))
        else:
             UQ_Photo.append(SA.Sum_dB([Photodetector_Thermal_noise[ind_UQ_PHOTO],Photodetector_Shot_noise[ind_UQ_PHOTO],Photodetector_Dark_current_noise[ind_UQ_PHOTO]]))
    
#    for nT in range(len(Photodetector_Thermal_noise)):
#        UQ_Photodetector.append(SA.Sum_dB([Photodetector_Thermal_noise[nT],Photodetector_Shot_noise[nT],Photodetector_Dark_current_noise[nT],Photodetector_TIA_noise[nT]]))
#    UQ_photodetector=[round(UQ_photodetector[i_dec],3) for i_dec in range(len(UQ_photodetector))] # 3 decimals
    return UQ_Photo



#%% OPTICAL AMPLIFIER
def UQ_Optical_amplifier(user_inputs,inputs,cts,direct,Wavelength,**Scenarios): # Calculating ASE - Amplified Spontaneous Emission definition ((**Optics and Photonics) Bishnu P. Pal - Guided Wave Optical Components and Devices_ Basics, Technology, and Applications -Academic Press (2005))
    UQ_Optical_amplifier=[]
    
    for i_UQ_OA in range(len(Scenarios.get('Temperature'))):
            FigureNoise=Scenarios['Optical_amplifier_NF'][i_UQ_OA] #dB
#            pdb.set_trace()

            UQ_Optical_amplifier.append([10*np.log10((10**(FigureNoise/10))*cts.h*(cts.c/Wavelength[i_UQ_OA])*10**(Scenarios['Optical_amplifier_Gain'][i_UQ_OA]/10))]) # ASE
#    pdb.set_trace()
    return (UQ_Optical_amplifier) # convert to dB


def FigNoise(inputs,direct): # This is the Figure noise of the optical amplifier. We use it to calculate the UQ_Optical_amplifier
    if isinstance (inputs.photonics_inp.Optical_amplifier_inputs['Optical_amplifier_noise']['Optical_amplifier_NF'], numbers.Number): #If user introduces a number or a table of values
        FigureNoise=inputs.photonics_inp.Optical_amplifier_inputs['Optical_amplifier_noise']['Optical_amplifier_NF']
    else:
        NoiseFigure_DATA = pd.read_csv(direct.Inputs_dir+inputs.photonics_inp.Optical_amplifier_inputs['Optical_amplifier_noise']['Optical_amplifier_NF'],delimiter=';',decimal=',') #read from an excel file variation of dB with wavelength(for now just with wavelegth)
        FigureNoise = []
        for i_FN in range(len(inputs.lidar_inp.Lidar_inputs['Wavelength'])):
            figure_noise_INT  = itp.interp1d(NoiseFigure_DATA.iloc[:,0],NoiseFigure_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
            NoiseFigure_VALUE = figure_noise_INT(inputs.lidar_inp.Lidar_inputs['Wavelength'][i_FN]) # in dB
            FigureNoise.append(NoiseFigure_VALUE.tolist())
#    FigureNoise=[round(FigureNoise[i_dec],3) for i_dec in range(len(FigureNoise))]
    return FigureNoise


#%% LASER SOURCE
def UQ_LaserSource(user_inputs,inputs,cts,direct,Wavelength,**Scenarios):
    UQ_laser_source=[]
    for i_UQ_LaserSource in range(len(Scenarios.get('Temperature'))):
        UQ_laser_source.append(Scenarios.get('Temperature')[i_UQ_LaserSource]*1+Scenarios.get('VAL_H')[i_UQ_LaserSource]*0.1+Scenarios.get('VAL_WAVE')[i_UQ_LaserSource]/1200+Scenarios.get('VAL_NOISE_LASER_SOURCE')[i_UQ_LaserSource]+
                               Scenarios.get('VAL_OC_LASER_SOURCE')[i_UQ_LaserSource])
#    UQ_laser_source=[round(UQ_laser_source[i_dec],3) for i_dec in range(len(UQ_laser_source))]
    return UQ_laser_source






#def Photodetector_Noise(inputs):
#    Photodetector_noise_DATA=pd.read_excel(direct.Main_directory+photodetector_Noise_FILE) #read from an excel file variation of dB with Wavelength(for now just with wavelegth)
#    Photodetector_noise_INT=itp.interp1d(Photodetector_noise_DATA.iloc[:,0],Photodetector_noise_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column Noise in dB
##    NEP_Lambda=NEP_min*(RespMAX/RespLambda) #NoiseEquivalentPower
#    Pmin=NEP_Lambda+np.sqrt(BW)#Minimum detectable signal power BW is teh band width
#    Photodetector_noise_VALUE=Photodetector_noise_INT(freq) # in dB
#    Photodetector_N=Photodetector_noise_VALUE.tolist()
#    return Photodetector_N