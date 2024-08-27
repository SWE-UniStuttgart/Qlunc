# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:41:29 2022

@author: fcosta
"""

from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
# from scipy.fft import fft, ifft

#%% Analog to digital converter

def UQ_ADC(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs,DataFrame):
    """
    Analog to digital converter uncertainty estimation. Location: ./UQ_Functions/UQ_SignalProcessor_Classes.py
    
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
    
    DataFrame: Dictionary
    
    """
    
    fd_peak = []
    vlos_MC = []
   
    #%% Inputs
    V_ref                = Atmospheric_Scenario.Vref        
    lidar_wavelength     = Qlunc_yaml_inputs['Components']['Laser']['Wavelength'] # wavelength of the laser source.
    fs_av                = Qlunc_yaml_inputs['Components']['ADC']['Sampling frequency']   # sampling frequency
    L                    = 2**Lidar.signal_processor.analog2digital_converter.nbits    #length of the signal.
    n_fftpoints          = L #2**8      # n° of points for each block (fft points).
    fd                   = [2 * i_V / lidar_wavelength for i_V in V_ref]  # Doppler frequency corresponding to Vref
    n_pulses             = 1        #   % n pulses for averaging the spectra
    N_MC                 = 5000 # n° MC samples to calculate the uncertainty due to bias in sampling frequency and wavelength

    #%% Uncertainty due to hardware noise, signal processing and speckle interference:
    
    # Hardware noise (thermal noise + shot noise + dark current noise + TIA noise):
    if DataFrame['Uncertainty Photodetector']['Total noise photodetector [dB]']==0:
        level_noise_hardware = np.array([0])
    else: 
        level_noise_hardware = 10**(DataFrame['Uncertainty Photodetector']['Total noise photodetector [dB]']/10) # Hardware noise added before signal downmixing
    hardware_noise       = np.random.normal(0 , level_noise_hardware , N_MC)

    
    # Bias in the sampling frequency 
    std_fs_av = Qlunc_yaml_inputs['Components']['ADC']['Uncertainty sampling freq']*fs_av
    # fs        = np.random.normal(fs_av , std_fs_av , N_MC)
    fs        = np.random.uniform(fs_av-std_fs_av , fs_av+std_fs_av , N_MC)

    Ts        = 1/fs

    
    # Bias in the laser wavelength 
    stdv_wavelength   = Qlunc_yaml_inputs['Components']['Laser']['stdv Wavelength'] / np.sqrt(3) #; % m
    wavelength_noise  = np.random.normal(lidar_wavelength , stdv_wavelength , N_MC)  #; % Noisy wavelength vector
    e_perc_wavelength = 100 * wavelength_noise / lidar_wavelength


    
    #%% MC method  to calculate the impact of the uncertainty sources above
    Stdv_fpeak,Stdv_vlos,mean_vlos=[],[],[]
    # pdb.set_trace()
    # Check if all elements in fd are the same to differentiate from the TimeSeries, which will have different fd values, and need to go through all of them:
    if all(x == fd[0] for x in fd): 
        fd0=[fd[0]]
    else:
        fd0=fd
        
    for i_fd in range(len(fd0)):
        fd_peak,vlos_MC=[],[]
        # Speckle multiplicative noise    
        lower             = fd[i_fd]-Lidar.signal_processor.analog2digital_converter.u_speckle*fd[i_fd]
        upper             = fd[i_fd]+Lidar.signal_processor.analog2digital_converter.u_speckle*fd[i_fd]    
        # fd_speckle_noise  = np.random.uniform( lower, upper, N_MC)
        fd_speckle_noise  =  np.random.normal(fd[i_fd] , fd[i_fd]*Lidar.signal_processor.analog2digital_converter.u_speckle/np.sqrt(3) , N_MC)  #; % Noisy wavelength vector
        
        for ind_MCM in range(N_MC):
            

            t = np.array(range(0,n_fftpoints)) * Ts[ind_MCM]
            f = np.linspace(0 , int(fs[ind_MCM] / 2) , int(math.floor(len(t)) / 2 + 1))

            # Signal + Noise:
            #Create signal and add hardware noise and speckle noise:
            S1 = hardware_noise[ind_MCM] + (np.sin(2*np.pi*fd_speckle_noise[ind_MCM]*t))#-.1*np.sin(2*np.pi*abs(np.random.normal(0,1.9))*fd_speckle_noise[ind_MCM]*t) + .2*np.sin(2*np.pi*abs(np.random.normal(0,3))*fd_speckle_noise[ind_MCM]*t)+\
                                    #.2*np.sin(2*np.pi*abs(np.random.normal(0,6))*fd_speckle_noise[ind_MCM]*t) + .3*np.sin(2*np.pi*abs(np.random.normal(0,1))*fd_speckle_noise[ind_MCM]*t) #; % Adding up Signal contributors

            S = S1 / S1.max()
            """Uniform quantization approach
        
            Notebook: C2/C2S2_DigitalSignalQuantization.ipynb
        
            Args:
                S         : Original signal
                quant_min : Minimum quantization level 
                quant_max : Maximum quantization level 
        
            Returns:
                x_quant  : Quantized signal
            """
         
            quant_min                      = S.min()
            quant_max                      = S.max()
            x_normalize                    = (S - quant_min) * (L-1) / (quant_max - quant_min)
            x_normalize[x_normalize > L-1] = L-1
            x_normalize[x_normalize < 0]   = 0
            x_normalize_quant            = np.around(x_normalize)
            x_quant                      = (x_normalize_quant) * (quant_max-quant_min) / (L-1) + quant_min 
            
            # FFT
            P3            = fft(x_quant) # Fourier transform
            P2            = abs(P3) / n_fftpoints
            P1            = 2 * P2[0:int(n_fftpoints / 2)]
            S_fft_quant   = np.ndarray.tolist(P1**2)
            max_fft       = S_fft_quant.index(max(S_fft_quant))
            
            # Frequency peak and Vlos
            fd_peak.append(f[max_fft])
            vlos_MC.append(0.5*wavelength_noise[ind_MCM]*fd_peak[ind_MCM])
        
        # Stdv of the frequency peak and Vlos
        Stdv_fpeak.append( np.std(fd_peak))
        Stdv_vlos.append(np.std(vlos_MC))
        mean_vlos.append(np.mean(vlos_MC))
        
        # Store data
        
    DataFrame['Uncertainty ADC'] = {'Stdv Doppler f_peak [Hz]':np.array(Stdv_fpeak)*np.ones(len(fd)),'Stdv wavelength [m]':stdv_wavelength,'Stdv Vlos [m/s]':Stdv_vlos}
    return DataFrame

#%% Sum of uncertainties in `signal processor` module: 
def sum_unc_signal_processor(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs,DataFrame):
    """
    This function runs from the UQ_Lidar_Classes and calcualtes the uncertainty in the signal processor module.
    So far, the signal processor module only includes the ADC
    
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
    
    DataFrame: Dictionary
    
    """
    if Lidar.signal_processor.analog2digital_converter != None:
        # try:               
            # pdb.set_trace()
        DataFrame = Lidar.signal_processor.analog2digital_converter.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,DataFrame)
        # except:
        #     ADC_Uncertainty=None
        #     print(colored('Error in ADC uncertainty calculations!','cyan', attrs=['bold']))
    else:
        # pdb.set_trace()
        ADC_Uncertainty=None
        DataFrame['Uncertainty ADC'] = {'Stdv Doppler f_peak [Hz]':np.array(0)*np.linspace(1,1,len(Atmospheric_Scenario.temperature_dev)),'Stdv wavelength [m]':0,'Stdv Vlos [m/s]':0}
        print (colored('You didn´t include an ADC in the lidar. The ADC uncertainty contribution is zero in the lidar hardware uncertainty estimations','cyan', attrs=['bold']))       
    
    # Store data
    return DataFrame