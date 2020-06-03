# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:56:11 2020

@author: fcosta
"""
import numpy as np
import matplotlib.pyplot as plt
#This is a calculation for a amplified photodetector with fixed gain
#Lets calculate the different photodetector contributions: thermal, dark current, shot noise and transimpedance amplifier noise (TIA)


#first lets define inputs and constants

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
R           = n*e*wavelength/(h*c)  #[W/A]  Responsivity

# Amplifier inputs:
G           = 10 # Vin/Vout[V] Amplifier Gain:
Z_TIA       = 5*10^3  #[ohms] transimpedance gain
V_noise_TIA = 160*10**(-6) #[V] Voltage noise 
#input optical power:
Ps=np.arange(0,1000,.001)
Psax=10*np.log10(Ps)




#%% PHOTODETECTOR:

# thermal noise;
Thermal_noise = (4*k*T/RL)*BW #[W]
SNR_Thermal   = 10*np.log10(((R**2)/(4*k*T*BW/RL))*(Ps/1000)**2)


# Shot noise:
Shot_noise        = 10*np.log10(2*e*R*BW*Ps)
Photo_SNR_Shot_noise    = 10*np.log10(((R**2)/(2*e*R*BW))*Ps/1000)
#0
## Dark current noise
Dark_current_noise  = 10*np.log10(2*e*Id*BW*Ps)
SNR_DarkCurrent     = 10*np.log10(((R**2)/(2*e*Id*BW))*((Ps/1000)**2) )


#%% AMPLIFIER: TIA noise 'input-referred noise':

TIA_noise= (V_noise_TIA**2/Z_TIA**2)
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