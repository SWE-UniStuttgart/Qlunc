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
           - Optik 126 (2015) 3492–3495Contents lists available at ScienceDirectOptikjo ur nal homepage: www.elsevier.de/ijleoStudy of ASE noise power, noise figure and quantum conversion efficiency for wide-band EDFA
"""

from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
from Utils import Qlunc_Plotting as QPlot
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
    
    P_int = Lidar.photonics.photodetector.Power_interval*Lidar.photonics.photodetector.Active_Surf
    R     = Lidar.photonics.photodetector.Efficiency*cts.e*Qlunc_yaml_inputs['Components']['Laser']['Wavelength']/(cts.h*cts.c)  #[A/W]  Responsivity
    UQ_Photodetector.Responsivity = (R) # this notation allows me to get Responsivity from outside of the function 

    # SNR calculations:
    
    # SNR in watts
    UQ_Photodetector.SNR_thermal_w      = [((P_int**2*R**2) / (4*cts.k*Atmospheric_Scenario.temperature[0]*Lidar.photonics.photodetector.BandWidth/Lidar.photonics.photodetector.Load_Resistor))]
    UQ_Photodetector.SNR_shot_w         = [((P_int*R**2)    / (2*cts.e*R*Lidar.photonics.photodetector.BandWidth))]
    UQ_Photodetector.SNR_DarkCurrent_w  = [((P_int**2*R**2) / (2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth))]
    
    # SNR in dBW
    UQ_Photodetector.SNR_thermal     = [10*np.log10(UQ_Photodetector.SNR_thermal_w)][0]
    UQ_Photodetector.SNR_shot        = [10*np.log10(UQ_Photodetector.SNR_shot_w )][0]
    UQ_Photodetector.SNR_DarkCurrent = [10*np.log10(UQ_Photodetector.SNR_DarkCurrent_w)][0]
    
    # Noise calculations
    # Photodetector Thermal noise:
    for i in range(len(Atmospheric_Scenario.temperature)):
        UQ_Photodetector.Thermal_noise.append(4*cts.k*Atmospheric_Scenario.temperature[i]*Lidar.photonics.photodetector.BandWidth/Lidar.photonics.photodetector.Load_Resistor)         

    # Photodetector shot noise:
    UQ_Photodetector.Shot_noise     = [(2*cts.e*(R*Lidar.photonics.photodetector.SignalPower)*Lidar.photonics.photodetector.BandWidth*Lidar.photonics.photodetector.Active_Surf)]*len(Atmospheric_Scenario.temperature)     
    
    # Photodetector dark current noise:
    UQ_Photodetector.Dark_current_noise = [(2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth)]*len(Atmospheric_Scenario.temperature) 
      
    if any(TIA_val == 'None' for TIA_val in [Lidar.photonics.photodetector.Gain_TIA,Lidar.photonics.photodetector.V_Noise_TIA]): # If any value of TIA is None dont include TIA noise in estimations :
        UQ_Photodetector.UQ_Photo    = [ np.array([10*np.log10(np.sum([UQ_Photodetector.Thermal_noise,UQ_Photodetector.Shot_noise,UQ_Photodetector.Dark_current_noise]))])]
        # UQ_Photodetector.UQ_Photo    = [SA.unc_comb(10*np.log10([UQ_Photodetector.Thermal_noise,UQ_Photodetector.Shot_noise,UQ_Photodetector.Dark_current_noise]))]        

        UQ_Photodetector.SNR_total_w = [((R**2)/((2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth)+ \
                                                 (4*cts.k*Atmospheric_Scenario.temperature[0]*Lidar.photonics.photodetector.BandWidth/Lidar.photonics.photodetector.Load_Resistor)+ \
                                                 (2*cts.e*R*Lidar.photonics.photodetector.BandWidth*(P_int))))*(P_int)**2]
        UQ_Photodetector.Total_SNR   = [10*np.log10(UQ_Photodetector.SNR_total_w)][0]
        SNR_data={'SNR_Shot':UQ_Photodetector.SNR_shot,'SNR_Thermal':UQ_Photodetector.SNR_thermal,'SNR_Dark_Current':UQ_Photodetector.SNR_DarkCurrent,'Total_SNR':UQ_Photodetector.Total_SNR}
        print(colored('There is NO TIA component in the photodetector','cyan', attrs=['bold']))
    else:       
        # Photodetector TIA noise:
        UQ_Photodetector.TIA_noise   = [(Lidar.photonics.photodetector.V_Noise_TIA**2/Lidar.photonics.photodetector.Gain_TIA**2)]*len(Atmospheric_Scenario.temperature)
        UQ_Photodetector.SNR_TIA     = [10*np.log10(((R**2)/(Lidar.photonics.photodetector.V_Noise_TIA**2/Lidar.photonics.photodetector.Gain_TIA**2))*(P_int)**2)]       
        
        UQ_Photodetector.SNR_total_w = [(P_int**2*R**2)/((Lidar.photonics.photodetector.V_Noise_TIA**2/Lidar.photonics.photodetector.Gain_TIA**2)+ \
                                                 (2*cts.e*Lidar.photonics.photodetector.DarkCurrent*Lidar.photonics.photodetector.BandWidth)+ \
                                                 (4*cts.k*Atmospheric_Scenario.temperature[0]*Lidar.photonics.photodetector.BandWidth/Lidar.photonics.photodetector.Load_Resistor)+ \
                                                 (2*cts.e*R*Lidar.photonics.photodetector.BandWidth*P_int))]
        UQ_Photodetector.Total_SNR   = [10*np.log10(UQ_Photodetector.SNR_total_w)][0]
        SNR_data={'SNR_Shot':UQ_Photodetector.SNR_shot,'SNR_Thermal':UQ_Photodetector.SNR_thermal,'SNR_Dark_Current':UQ_Photodetector.SNR_DarkCurrent,'SNR_TIA':UQ_Photodetector.SNR_TIA,'Total_SNR':UQ_Photodetector.Total_SNR}
        UQ_Photodetector.UQ_Photo  = [ np.array([10*np.log10(np.sum([UQ_Photodetector.Thermal_noise,UQ_Photodetector.Shot_noise,UQ_Photodetector.Dark_current_noise,UQ_Photodetector.TIA_noise]))])]
        print(colored('There is a TIA component in the photodetector','cyan', attrs=['bold']))

    UQ_Photodetector.UQ_Photo_total=list(SA.flatten(UQ_Photodetector.UQ_Photo))
    Final_Output_UQ_Photo={'Uncertainty_Photodetector':UQ_Photodetector.UQ_Photo_total,'SNR_data_photodetector':SNR_data, 'Thermal noise':10*np.log10(UQ_Photodetector.Thermal_noise),'Shot noise':10*np.log10(UQ_Photodetector.Shot_noise), 'Dark current noise':10*np.log10(UQ_Photodetector.Dark_current_noise), 'TIA noise':10*np.log10(UQ_Photodetector.TIA_noise)}      
    
    # Add to data frame
    # Lidar.lidar_inputs.dataframe['Photodetector']=Final_Output_UQ_Photo['Uncertainty_Photodetector'][0]
    Lidar.lidar_inputs.dataframe['Thermal noise']=10*np.log10(UQ_Photodetector.Thermal_noise)
    Lidar.lidar_inputs.dataframe['Shot noise']=10*np.log10(UQ_Photodetector.Shot_noise)
    Lidar.lidar_inputs.dataframe['Dark current noise']=10*np.log10(UQ_Photodetector.Dark_current_noise)
    Lidar.lidar_inputs.dataframe['TIA noise']=10*np.log10(UQ_Photodetector.TIA_noise)
    
    # Plotting:
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_Photo,False,Qlunc_yaml_inputs['Flags']['Photodetector noise'],False,False,False,False,False)
    return Final_Output_UQ_Photo,Lidar.lidar_inputs.dataframe



#%% Sum of uncertainties in photonics module: 
def sum_unc_photonics(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs): 
    List_Unc_photonics = []
    
    ###############
    # Photodetector
    ###############
    if Lidar.photonics.photodetector != None:
        try: # each try/except evaluates wether the component is included in the module, therefore in the calculations
            Photodetector_Uncertainty,DataFrame = Lidar.photonics.photodetector.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
            # pdb.set_trace()
            List_Unc_photonics.append(Photodetector_Uncertainty['Thermal noise'])
            List_Unc_photonics.append(Photodetector_Uncertainty['Shot noise'])
            List_Unc_photonics.append(Photodetector_Uncertainty['Dark current noise'])
            List_Unc_photonics.append(Photodetector_Uncertainty['TIA noise'])
            
        except:
            Photodetector_Uncertainty=None
            print(colored('Error in photodetector uncertainty calculations!','cyan', attrs=['bold']))
    else:
        print(colored('You didnt include a photodetector in the lidar, so that photodetector uncertainty contribution is not in lidar uncertainty estimations','cyan', attrs=['bold']))

    Uncertainty_Photonics_Module                     = SA.unc_comb(List_Unc_photonics)
    Final_Output_UQ_Photonics                        = {'Noise Photodetector':Uncertainty_Photonics_Module}
    Lidar.lidar_inputs.dataframe['Total noise photodetector [dB]'] = np.array(Final_Output_UQ_Photonics['Noise Photodetector'])
    return Final_Output_UQ_Photonics,Lidar.lidar_inputs.dataframe
