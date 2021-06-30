# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:15:24 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c)

Here we calculate the uncertainties related with components in the `photonics`
module. 
    
   - noise definintions: 
       - Photodetector: H. Rongqing, “Introduction to Fiber-Optic Communications,” Elsevier, vol. 4, pp. 125–154, 2019, doi: https://doi.org/10.1016/B978-0-12-805345-4.00004-4.
       - Optical amplifier:  
           - Calculating ASE - Amplified Spontaneous Emission definition ((**Optics and Photonics) Bishnu P. Pal - Guided Wave Optical Components and Devices_ Basics, Technology, and Applications -Academic Press (2005))
           - EDFA Testing with interpolation techniques - Product note 71452-1
"""

from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
from Utils import Qlunc_Plotting as QPlot
import os
import pdb
#%% PHOTODETECTOR:
def UQ_Photodetector(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    """
    Photodetector uncertainty estimation. Location: ./UQ_Functions/UQ_Photonics_Classes.py
    
    Parameters
    ----------
    
    * Lidar
        data...
    * Atmospheric_Scenario
        Atmospheric data. Integer or Time series
    * cts
        Physical constants
    * Qlunc_yaml_inputs
        Lidar parameters data
        
    Returns
    -------
    
    list
    
    """    
    
    UQ_Photodetector.Thermal_noise      = []
    UQ_Photodetector.SNR_thermal        = []
    UQ_Photodetector.Shot_noise         = []
    UQ_Photodetector.SNR_shot           = []
    UQ_Photodetector.Dark_current_noise = []
    UQ_Photodetector.SNR_DarkCurrent    = []
    UQ_Photodetector.UQ_Photo           = []
    UQ_Photodetector.SNR_TIA            = []
    UQ_Photodetector.TIA_noise          = []
    
    R = Lidar.photonics.photodetector.Efficiency*cts.e*Lidar.lidar_inputs.Wavelength/(cts.h*cts.c)  #[A/W]  Responsivity
    UQ_Photodetector.Responsivity = (R) # this notation allows me to get Responsivity from outside of the function 
    # pdb.set_trace()
    # SNR calculations:
    # SNR in watts
    UQ_Photodetector.SNR_thermal_w      = [((R**2)/(4*cts.k*Atmospheric_Scenario.temperature[0]*Lidar.photonics.photodetector.BandWidth/Lidar.photonics.photodetector.Load_Resistor))*(Lidar.photonics.photodetector.Power_interval/1000)**2]
    UQ_Photodetector.SNR_shot_w         = [((R**2)/(2*cts.e*R*Lidar.photonics.photodetector.BandWidth))*(Lidar.photonics.photodetector.Power_interval)/1000]
    UQ_Photodetector.SNR_DarkCurrent_w  = [((R**2)/(2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth))*(Lidar.photonics.photodetector.Power_interval/1000)**2]
    # SNR in dBW
    UQ_Photodetector.SNR_thermal     = [10*np.log10(UQ_Photodetector.SNR_thermal_w)][0]
    UQ_Photodetector.SNR_shot        = [10*np.log10(UQ_Photodetector.SNR_shot_w )][0]
    UQ_Photodetector.SNR_DarkCurrent = [10*np.log10(UQ_Photodetector.SNR_DarkCurrent_w)][0]
    
    
    # Photodetector Thermal noise:
    for i in range(len(Atmospheric_Scenario.temperature)):
        UQ_Photodetector.Thermal_noise.append(4*cts.k*Atmospheric_Scenario.temperature[i]*Lidar.photonics.photodetector.BandWidth/Lidar.photonics.photodetector.Load_Resistor)         
    
    # Photodetector shot noise:
    UQ_Photodetector.Shot_noise     = [(2*cts.e*R*Lidar.photonics.photodetector.BandWidth*Lidar.photonics.photodetector.SignalPower)]*len(Atmospheric_Scenario.temperature)     
    
    # Photodetector dark current noise:
    UQ_Photodetector.Dark_current_noise = [(2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth)]*len(Atmospheric_Scenario.temperature) 
    
      
    if any(TIA_val == 'None' for TIA_val in [Lidar.photonics.photodetector.Gain_TIA,Lidar.photonics.photodetector.V_Noise_TIA]): # If any value of TIA is None dont include TIA noise in estimations :
        UQ_Photodetector.UQ_Photo    = [(SA.unc_comb(10*np.log10([UQ_Photodetector.Thermal_noise,UQ_Photodetector.Shot_noise,UQ_Photodetector.Dark_current_noise])))]        
        UQ_Photodetector.SNR_total_w = [((R**2)/((2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth)+ \
                                                 (4*cts.k*Atmospheric_Scenario.temperature[0]*Lidar.photonics.photodetector.BandWidth/Lidar.photonics.photodetector.Load_Resistor)+ \
                                                 (2*cts.e*R*Lidar.photonics.photodetector.BandWidth*(Lidar.photonics.photodetector.Power_interval/1000))))*(Lidar.photonics.photodetector.Power_interval/1000)**2]
        UQ_Photodetector.Total_SNR   = [10*np.log10(UQ_Photodetector.SNR_total_w)][0]
        SNR_data={'SNR_Shot':UQ_Photodetector.SNR_shot,'SNR_Thermal':UQ_Photodetector.SNR_thermal,'SNR_Dark_Current':UQ_Photodetector.SNR_DarkCurrent,'Total_SNR':UQ_Photodetector.Total_SNR}
        print('There is NO TIA component in the photodetector')
    else:       
        # Photodetector TIA noise:
        UQ_Photodetector.TIA_noise   = [(Lidar.photonics.photodetector.V_Noise_TIA**2/Lidar.photonics.photodetector.Gain_TIA**2)]*len(Atmospheric_Scenario.temperature)
        UQ_Photodetector.SNR_TIA     = [10*np.log10(((R**2)/(Lidar.photonics.photodetector.V_Noise_TIA**2/Lidar.photonics.photodetector.Gain_TIA**2))*(Lidar.photonics.photodetector.Power_interval/1000)**2)]       
        
        UQ_Photodetector.SNR_total_w = [((R**2)/((Lidar.photonics.photodetector.V_Noise_TIA**2/Lidar.photonics.photodetector.Gain_TIA**2)+ \
                                                 (2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth)+ \
                                                 (4*cts.k*Atmospheric_Scenario.temperature[0]*Lidar.photonics.photodetector.BandWidth/Lidar.photonics.photodetector.Load_Resistor)+ \
                                                 (2*cts.e*R*Lidar.photonics.photodetector.BandWidth*(Lidar.photonics.photodetector.Power_interval/1000))))*(Lidar.photonics.photodetector.Power_interval/1000)**2]
        UQ_Photodetector.Total_SNR   = [10*np.log10(UQ_Photodetector.SNR_total_w)][0]
        SNR_data={'SNR_Shot':UQ_Photodetector.SNR_shot,'SNR_Thermal':UQ_Photodetector.SNR_thermal,'SNR_Dark_Current':UQ_Photodetector.SNR_DarkCurrent,'Total_SNR':UQ_Photodetector.Total_SNR,'SNR_TIA':UQ_Photodetector.SNR_TIA}
        UQ_Photodetector.UQ_Photo  = SA.unc_comb(10*np.log10([UQ_Photodetector.Thermal_noise,UQ_Photodetector.Shot_noise,UQ_Photodetector.Dark_current_noise,UQ_Photodetector.TIA_noise]))
        print('There is a TIA component in the photodetector')

    UQ_Photodetector.UQ_Photo_total=list(SA.flatten(UQ_Photodetector.UQ_Photo))
    Final_Output_UQ_Photo={'Uncertainty_Photodetector':UQ_Photodetector.UQ_Photo_total,'SNR_data_photodetector':SNR_data}      
    Lidar.lidar_inputs.dataframe['Photodetector']=Final_Output_UQ_Photo['Uncertainty_Photodetector'][0]
    pdb.set_trace()
    # Plotting:
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_Photo,False,Qlunc_yaml_inputs['Flags']['Photodetector noise'])
    return Final_Output_UQ_Photo,Lidar.lidar_inputs.dataframe



#%% OPTICAL AMPLIFIER:
def UQ_Optical_amplifier(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs): 
    """
    Optical amplifier uncertainty estimation. Location: ./UQ_Functions/UQ_Photonics_Classes.py
    
    Parameters
    ----------
    
    * Lidar
        data...
    * Atmospheric_Scenario
        Atmospheric data. Integer or Time series
    * cts
        Physical constants
    * Qlunc_yaml_inputs
        Lidar parameters data
        
    Returns
    -------
    
    list
    
    """ 
    # Obtain SNR from figure noise or pass directly numerical value:
    if isinstance (Lidar.photonics.optical_amplifier.NoiseFig, numbers.Number): #If user introduces a number or a table of values
        FigureNoise=[(Lidar.photonics.optical_amplifier.NoiseFig)]*len(Atmospheric_Scenario.temperature) #Figure noise vector        
        # pdb.set_trace()
        # ASE noise:
        UQ_Optical_amplifier  = [np.array([10*np.log10((10**(FigureNoise[0]/10))*cts.h*(cts.c/Lidar.lidar_inputs.Wavelength)*10**(Lidar.photonics.optical_amplifier.OA_Gain/10))]*len(Atmospheric_Scenario.temperature))] 
        
    else:
        NoiseFigure_DATA  = pd.read_csv(Lidar.photonics.optical_amplifier.NoiseFig,delimiter=';',decimal=',') #read from a .csv file variation of dB with wavelength (for now just with wavelength)    
        # HERE THERE IS AN ERROR PRODUCED BY THE INTERPOLATION --> DATAPROCESSING UNCERTAINTIES
        figure_noise_INT  = itp.interp1d(NoiseFigure_DATA.iloc[:,0],NoiseFigure_DATA.iloc[:,1],kind='cubic',fill_value="extrapolate")# First column wavelength,second column SNR in dB
        NoiseFigure_VALUE = figure_noise_INT(Lidar.lidar_inputs.Wavelength) # in dB
        FigureNoise       = (NoiseFigure_VALUE.tolist())
        
        # ASE noise:
        UQ_Optical_amplifier = [np.array([10*np.log10((10**(FigureNoise/10))*cts.h*(cts.c/Lidar.lidar_inputs.Wavelength)*10**(Lidar.photonics.optical_amplifier.OA_Gain/10))]*len(Atmospheric_Scenario.temperature)) ]
    
    Final_Output_UQ_Optical_Amplifier = {'Uncertainty_OpticalAmp':UQ_Optical_amplifier}
    Lidar.lidar_inputs.dataframe['Optical Amplifier'] = Final_Output_UQ_Optical_Amplifier['Uncertainty_OpticalAmp'][0]
    # pdb.set_trace()

    return Final_Output_UQ_Optical_Amplifier,Lidar.lidar_inputs.dataframe


#%% LASER SOURCE
def UQ_Laser(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    """
    Laser uncertainty estimation. Location: ./UQ_Functions/UQ_Photonics_Classes.py
    
    Parameters
    ----------
    
    * Lidar
        data...
    * Atmospheric_Scenario
        Atmospheric data. Integer or Time series
    * cts
        Physical constants
    * Qlunc_yaml_inputs
        Lidar parameters data
        
    Returns
    -------
    
    list
    
    """ 
    # STDV of the laser due to the wavelength error and the confidence interval, assuming a normal distribution. "JCGM 100:2008 GUM 1995 with minor changes", Anxex G, Table G1

        
    if Lidar.photonics.laser.conf_int==1:
        u_fact = 1
    elif Lidar.photonics.laser.conf_int==2:
        u_fact = 1.645
    elif Lidar.photonics.laser.conf_int==3:
        u_fact = 1.96
    elif Lidar.photonics.laser.conf_int==4:
        u_fact = 2
    elif Lidar.photonics.laser.conf_int==5:
        u_fact = 2.576
    elif Lidar.photonics.laser.conf_int==6:
        u_fact = 3        
        
    # UQ_Laser = np.array([( Lidar.photonics.laser.stdv_wavelength/u_fact)])    
    UQ_Laser = 10*np.log10(10**(Lidar.photonics.laser.RIN/10)*Lidar.photonics.laser.BandWidth*Lidar.photonics.laser.Output_power)
    
    
    Final_Output_UQ_Laser = {'Uncertainty_Laser':UQ_Laser}
    # pdb.set_trace()
    Lidar.lidar_inputs.dataframe['Laser'] = Final_Output_UQ_Laser['Uncertainty_Laser']
    # pdb.set_trace()

    return Final_Output_UQ_Laser,Lidar.lidar_inputs.dataframe

#%% Acousto-optic-modulator

def UQ_AOM(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    """
    AOM uncertainty estimation. Location: ./UQ_Functions/UQ_Photonics_Classes.py
    
    Parameters
    ----------
    
    * Lidar
        data...
    * Atmospheric_Scenario
        Atmospheric data. Integer or Time series
    * cts
        Physical constants
    * Qlunc_yaml_inputs
        Lidar parameters
        
    Returns
    -------
    
    AOM losses
    
    """ 
    if Lidar.lidar_inputs.LidarType=='Pulsed':
        UQ_AOM = Lidar.photonics.acousto_optic_modulator.insertion_loss # in dB
        P_il=Pt/(10**(Lidar.photonics.acousto_optic_modulator.insertion_loss/10))
    return UQ_AOM
#%% Sum of uncertainties in photonics module: 
def sum_unc_photonics(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs): 
    List_Unc_photonics = []
    try: # each try/except evaluates wether the component is included in the module, therefore in the calculations
        Photodetector_Uncertainty,DataFrame = Lidar.photonics.photodetector.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
        List_Unc_photonics.append(Photodetector_Uncertainty['Uncertainty_Photodetector'])
        
    except:
        Photodetector_Uncertainty=None
        print('No photodetector in calculations!')
    try:
        Optical_Amplifier_Uncertainty,DataFrame = Lidar.photonics.optical_amplifier.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
        List_Unc_photonics.append(Optical_Amplifier_Uncertainty['Uncertainty_OpticalAmp'])
    except:
        Optical_Amplifier_Uncertainty=None
        print('No optical amplifier in calculations!')
    # try:
    #     # pdb.set_trace()
    #     Laser_Uncertainty,DataFrame = Lidar.photonics.laser.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
    #     List_Unc_photonics.append(Laser_Uncertainty['Uncertainty_Laser'])
        
    # except:
    #     Laser_Uncertainty=None
    #     print('No laser in calculations!')
    
    # pdb.set_trace()
    Uncertainty_Photonics_Module                     = SA.unc_comb(List_Unc_photonics)
    Final_Output_UQ_Photonics                        = {'Uncertainty_Photonics':Uncertainty_Photonics_Module}
    Lidar.lidar_inputs.dataframe['Photonics Module'] = Final_Output_UQ_Photonics['Uncertainty_Photonics'][0]
    return Final_Output_UQ_Photonics,Lidar.lidar_inputs.dataframe
