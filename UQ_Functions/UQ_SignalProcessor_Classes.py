# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:41:29 2022

@author: fcosta
"""

from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
from Utils import Qlunc_Plotting as QPlot
from scipy.fft import fft, ifft

#%% Analog to digital converter

def UQ_ADC(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
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
    
    list
    
    """   
     
    # n_bits           = Lidar.signal_processor.analog2digital_converter.nbits     # N_MC° bits ADC
    V_ref            = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']      # Reference voltaje ADC
    lidar_wavelength = Qlunc_yaml_inputs['Components']['Laser']['Wavelength'] # wavelength of the laser source.
    fs_av            = Qlunc_yaml_inputs['Components']['ADC']['Sampling frequency']   # sampling frequency
    
    # L                = 2**n_bits    #length of the signal.

    L                = 2**Lidar.signal_processor.analog2digital_converter.nbits    #length of the signal.
    n_fftpoints      = L       # n° of points for each block (fft points).
    fd               = 2*V_ref/lidar_wavelength  # Doppler frequency corresponding to Vref
    level_noise      = 0.008 # Hardware noise added before signal downmixing
    n_pulses         = 1        #   % n pulses for averaging the spectra
    N_MC             = 1000 # n° MC samples to calculate the uncertainty due to bias in sampling frequency and wavelength
    
    # Uncertainty in the signal processing.
    # pdb.set_trace()
    ### Uncertainty in the sampling frequency 
    std_fs_av   = Qlunc_yaml_inputs['Components']['ADC']['Uncertainty sampling freq']*fs_av
    fs=np.random.normal(fs_av,std_fs_av,N_MC)
    Ts          = 1/fs
    # Accepted values
    Tv = 1/fs_av
    tv = np.array(range(0,n_fftpoints))*Tv
    fv = np.linspace(0,fs_av/2,math.floor(len(tv)/2+1))
    vv=0.5*lidar_wavelength*fv
    
    stdv_wavelength  = 1e-9/np.sqrt(3) #; % m
    wavelength_noise = lidar_wavelength+np.random.normal(0,stdv_wavelength,N_MC)  #; % Noisy wavelength vector
    e_perc_wavelength = 100*wavelength_noise/lidar_wavelength
    fd_peak=[]
    vlos_MC=[]
    for ind_pulse in range(N_MC):

        t= np.array(range(0,n_fftpoints))*Ts[ind_pulse]
        f=np.linspace(0,int(fs[ind_pulse]/2),int(math.floor(len(t))/2+1))
        noise=np.random.normal(0,level_noise,len(t))
        S        = noise+(14*np.sin(2*np.pi*fd*t) - 2.1*np.sin(2*np.pi*abs(np.random.normal(0,1.9))*fd*t) + 2*np.sin(2*np.pi*abs(np.random.normal(0,3))*fd*t)+\
                      3.24*np.sin(2*np.pi*abs(np.random.normal(0,6))*fd*t) + 4.7*np.sin(2*np.pi*abs(np.random.normal(0,1))*fd*t) ) #; % Adding up Signal contributors
        # S=S/S.max()
        """Uniform quantization approach
    
        Notebook: C2/C2S2_DigitalSignalQuantization.ipynb
    
        Args:
            S (np.ndarray): Original signal
            quant_min (float): Minimum quantization level (Default value = -1.0)
            quant_max (float): Maximum quantization level (Default value = 1.0)
            quant_level (int): Number of quantization levels (Default value = 5)
    
        Returns:
            x_quant (np.ndarray): Quantized signal
        """
        
        quant_min=S.min()
        quant_max=S.max()
        x_normalize = (S-quant_min) * (L-1) / (quant_max-quant_min)
        x_normalize[x_normalize > L - 1] = L - 1
        x_normalize[x_normalize < 0] = 0
        x_normalize_quant = np.around(x_normalize)
        x_quant = (x_normalize_quant) * (quant_max-quant_min) / (L-1) + quant_min 
        P3            = fft(x_quant) # Fourier transform
        P2            = abs(P3)/n_fftpoints
        P1            = 2*P2[1:int((n_fftpoints/2)+1)]
        # P1            = 2*P1[1:]
        S_fft_quant   = np.ndarray.tolist(P1**2)
        max_fft=S_fft_quant.index(max((S_fft_quant)))
        
        fd_peak.append(f[max_fft])
        vlos_MC.append(0.5*wavelength_noise[ind_pulse]*fd_peak[ind_pulse])

        # RMSE=SA.rmse(S,S_fft_quant)

    Stdv_fpeak=np.std(fd_peak)
    Stdv_vlos=np.std(vlos_MC)
    # pdb.set_trace() 
          # 
    # # UQ_ADC.Thermal_noise =[]
    # # UQ_ADC.ADC_quant_err =[]
    # # UQ_ADC.ADC_resolution_V =[]
    # pdb.set_trace()
    # # Resolution:
    # UQ_ADC.ADC_resolution_V  = (Lidar.signal_processor.analog2digital_converter.vref-Lidar.signal_processor.analog2digital_converter.vground)/(2**Lidar.signal_processor.analog2digital_converter.nbits)# =1LSB
    # ADC_resolution_dB = 10**((Lidar.signal_processor.analog2digital_converter.vref-Lidar.signal_processor.analog2digital_converter.vground)/(2**Lidar.signal_processor.analog2digital_converter.nbits)/20 )# =1LSB
    # # ADC_DR_dB         = 20*np.log10(((2**Lidar.signal_processor.analog2digital_converter.nbits)-1))
    # # ADC_DR_V          = 10**(ADC_DR_dB/20)

    # # ADC_FS = Lidar.signal_processor.analog2digital_converter.vref-ADC_resolution# ADC full scale
    
    # ## Noise added by an ADC 
    # UQ_ADC.ADC_quant_err  = 0.5*UQ_ADC.ADC_resolution_V # Maximum quantization error added by the ADC representing the worst case
    # # UQ_ADC.ADC_quant_err  = UQ_ADC.ADC_resolution_V/np.sqrt(12) # Analog-digital conversion - Marek Gasior - CERN Beam instrumentation Group

    # UQ_ADC.Thermal_noise = cts.k*Atmospheric_Scenario.temperature[0]*Lidar.signal_processor.analog2digital_converter.BandWidth
    # UQ_ADC.UQ_ADC_Total_W =  np.sqrt(UQ_ADC.ADC_quant_err**2+UQ_ADC.Thermal_noise**2)
    # UQ_ADC.UQ_ADC_Total_dB= 10*np.log10(UQ_ADC.UQ_ADC_Total_W)
    Final_Output_UQ_ADC = {'Stdv Doppler f_peak':np.array(Stdv_fpeak),'Stdv Vlos':np.array(Stdv_vlos)}
    # # pdb.set_trace()
    # #Ideal ADC SNR
    # SNR_ideal_dB = 6.02*Lidar.signal_processor.analog2digital_converter.nbits+1.76 # dB --> by definition
    # SNR_ideal_watts = 10**(SNR_ideal_dB/10)
    
    
    Lidar.lidar_inputs.dataframe['Uncertainty frequency peak [Hz]']=Final_Output_UQ_ADC['Stdv Doppler f_peak']*np.linspace(1,1,len(Atmospheric_Scenario.temperature)) # linspace to create the appropriate length for the xarray.
    # pdb.set_trace()
    return Final_Output_UQ_ADC,Lidar.lidar_inputs.dataframe

# #%% Frequency analyser
# def UQ_FrequencyAnalyser(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
#     print('Frequency analyser')
#%% Sum of uncertainties in `signal processor` module: 
def sum_unc_signal_processor(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    # pdb.set_trace()
    List_Unc_signal_processor=[]
    if Lidar.signal_processor.analog2digital_converter != None:
        try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations               
                ADC_Uncertainty,DataFrame=Lidar.signal_processor.analog2digital_converter.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
                List_Unc_signal_processor.append(ADC_Uncertainty['ADC_Noise'])      
        except:
            ADC_Uncertainty=None
            print(colored('Error in ADC uncertainty calculations!','cyan', attrs=['bold']))
    else:
        print (colored('You didn´t include an analog to digital converter in the lidar.','cyan', attrs=['bold']))       
    # pdb.set _trace()
    Uncertainty_SignalProcessor_Module=SA.unc_comb(List_Unc_signal_processor)
    Final_Output_UQ_SignalProcessor = {'Uncertainty_SignalProcessor':Uncertainty_SignalProcessor_Module}
    # 
    Lidar.lidar_inputs.dataframe['SignalProcessor Module']=Final_Output_UQ_SignalProcessor['Uncertainty_SignalProcessor']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))  # linspace to create the appropiate length for the xarray. 
    # pdb.set_trace()
    return Final_Output_UQ_SignalProcessor,Lidar.lidar_inputs.dataframe