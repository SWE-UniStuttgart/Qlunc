# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:10:40 2020

@author: fcosta
"""
#Trying to recreate the results of Zhang_Noise_Photodetector_current** - Noise with distance 
from scipy import fft
import numpy as np
from numpy.fft import fft,fftfreq
import matplotlib.pyplot as plt

Signal=[]
N=1000000 #numero de mustras
T=1000 # periodo de muestreo

t=np.linspace(0,T,N)
#for i in (50,80):
#    Signal.append(np.cos(i*2*np.pi*t))
#Signal=Signal[0]+ Signal[1]
Signal = np.sin(50.0 * 2.0*np.pi*t/T) + 0.5*np.sin(80.0 * 2.0*np.pi*t/T)
#plt.plot(x,Signal)
Signalfreq=2*(np.abs(fft(Signal)))/N
freq=N*np.fft.fftfreq(N)#np.fft.fftfreq(x.shape[-1])
mask=freq>0

#plt.figure()
#plt.plot(t,Signal)
plt.plot(freq[mask],Signalfreq[mask])


# Number of sample points
N = 600
# sample spacing
T = 1/1000
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
import matplotlib.pyplot as plt
#plt.plot(x,y)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

#########################################################################################################################
n=0.9
E0=2
k=1
dw=1
Dw=0.01
c=3*(10**8) # c in m/s
Dis=50000060#(m)
taud=2*Dis/c
d_ws=1
w=np.linspace(0,100,1000)
ws=60
A=n*(E0**2)

#S=((A**2)*(1+k**2)*DD(w,w))+((A**2)*k*np.exp(-Dw*taud)*DD(w,w-ws))+(0.5*(((A**2)*k*Dw*np.exp(-Dw*taud))/((Dw**2)+(w-ws)**2)))*(np.exp(Dw*taud)-(np.sin(w*taud-ws*taud)/(w-ws))-np.cos(w*taud-ws*taud))
#S=(0.5*(((A**2)*k*Dw*np.exp(-Dw*taud))/((Dw**2)+(w-ws)**2)))*(np.exp(Dw*taud)-(np.sin(w*taud-ws*taud)/(w-ws))-np.cos(w*taud-ws*taud))
#
#Sfft=np.fft.fft(S)
#SdB=10*np.log10(Sfft)
#tauddB=10*np.log10(taud)
#plt.figure()
#plt.plot(w,SdB)
###########################################################################

Signal_res=((A**2)*k*Dw*(np.exp(-w*taud))/(2*((Dw**2)+(w-ws)**2)))*(np.exp(Dw*taud)-(np.sin(w*taud-ws*taud)/(w-ws))-np.cos(w*taud-ws*taud))
Signal_res_dB=10*np.log10(Signal_res)
#plt.figure()
plt.plot(w,Signal_res_dB)