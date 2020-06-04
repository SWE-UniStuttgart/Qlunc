# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:56:11 2020

@author: fcosta
"""
import numpy as np
import matplotlib.pyplot as plt
from Qlunc_ImportModules import *
#This is a calculation for a amplified photodetector with fixed gain
#The different photodetector contributions are here calculated, namely: thermal, dark current, shot noise and transimpedance amplifier noise (TIA)


#first lets define some inputs and constants

#Constants:
k = 1.38064852 *10**(-23) # Boltzman constant:[m^2 kg s^-2 K^-1]
h = 6.6207004  *10**(-34) # Plank constant [m^2 kg s^-1]
e = 1.60217662 *10**(-19) # electron charge [C]
c = 2.99792    *10**8 #speed of light [m s^-1]
T = 300 #[K] Temperature

#Photodiode Inputs
BW          = 380*10**6 #1*10**(9) #[Hz] Band width
wavelength  = 1550 *10**(-9) #[m] 
RL          = 50 #[ohms] Load resistor
n           = 0.85 #efficiency of the photodiode:
Id          = 5*10**(-9) #[A] Dark current intensity
R           = inputs.photodetector.n*cts.e*inputs.lidar_inp.Lidar_inputs['Wavelength'][0]/(cts.h*cts.c)  #[W/A]  Responsivity

# Amplifier inputs:
G           = 10 # Vin/Vout[V] Amplifier Gain:
Z_TIA       = 5*10^3  #[ohms] transimpedance gain
V_noise_TIA = 160*10**(-6) #[V] Voltage noise 
#input optical power:
Ps=np.arange(0,1000,.001)
Psax=10*np.log10(Ps)




#%% PHOTODETECTOR:

# thermal noise;
Thermal_noise = (4*cts.k*T/inputs.photodetector.RL)*inputs.photodetector.BW #[W]
SNR_Thermal   = 10*np.log10(((R**2)/(4*cts.k*T*inputs.photodetector.BW/inputs.photodetector.RL))*(Ps/1000)**2)


# Shot noise:
Shot_noise        = 10*np.log10(2*cts.e*R*inputs.photodetector.BW*Ps)
Photo_SNR_Shot_noise    = 10*np.log10(((R**2)/(2*cts.e*R*inputs.photodetector.BW))*Ps/1000)
#0
## Dark current noise
Dark_current_noise  = 10*np.log10(2*cts.e*inputs.photodetector.Id*inputs.photodetector.BW*Ps)
SNR_DarkCurrent     = 10*np.log10(((R**2)/(2*cts.e*inputs.photodetector.Id*inputs.photodetector.BW))*((Ps/1000)**2) )


#%% AMPLIFIER: TIA noise 'input-referred noise':

TIA_noise= (inputs.photodetector.V_noise_TIA**2/inputs.photodetector.Z_TIA**2)
SNR_TIA=10*np.log10(((R**2)/(TIA_noise))*(Ps/1000)**2)


plt.figure()
#plt.xscale('log',basex=10)
#plt.yscale('log',basey=10)

plt.plot(Psax,Photo_SNR_Shot_noise,Psax,SNR_Thermal,Psax,SNR_DarkCurrent,Psax,SNR_TIA)
plt.xlabel('Input Signal optical power (dBm)',fontsize=29)
plt.ylabel('SNR (dB)',fontsize=29)
plt.legend(['Shot Noise','Thermal Noise','Dark current Noise','TIA Noise'],fontsize=16)#,'Total error [w]'])
plt.title('SNR Photodetector',fontsize=35)
plt.grid(axis='both')

#plt.figure()
#plt.plot(Psax,Shot_noise,Psax,Dark_current_noise)
#plt.xlabel('Input Signal optical power (dB)')
#plt.ylabel('Noise (dB)')
#plt.legend(['Shot Noise','Dark current'])#,'Total error [w]'])