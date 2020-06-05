# -*- coding: utf-8 -*-
"""
Created on Fri May 22 02:22:01 2020

@author: fcosta
"""

import numpy as np
import scipy.interpolate as itp
import scipy.integrate as itg
import scipy.optimize as opt
import pandas as pd
import Qlunc_UQ_Power_func       # script with all calculations of Power module unc are done
import Qlunc_UQ_Photonics_func   # script with all calculations of Photonics module unc are done
import Qlunc_UQ_Optics_func      # script with all calculations of Optics module unc are done
#import Qlunc_UQ_Data_processing  # script with all calculations of data processing methods unc are done
#import LiUQ_inputs
from Qlunc_inputs import *
import pdb
#import pickle # for GUI
import itertools
import functools
import matplotlib.pyplot as plt
import Qlunc_Help_standAlone as salone
from functools import reduce
from operator import getitem
from collections import defaultdict  
import scipy.stats as st
from pylab import *
import numbers
import decimal


    
Func     = {'power_source_noise'  : Qlunc_UQ_Power_func.UQ_PowerSource,
            'converter_noise'     : Qlunc_UQ_Power_func.UQ_Converter,
            'converter_losses'    : Qlunc_UQ_Power_func.Losses_Converter,
            'laser_source_noise'  : Qlunc_UQ_Photonics_func.UQ_LaserSource,
            'photodetector_noise' : Qlunc_UQ_Photonics_func.UQ_Photodetector,
            'amplifier_noise'     : Qlunc_UQ_Photonics_func.UQ_Optical_amplifier, 
            'amplifier_fignoise'  : Qlunc_UQ_Photonics_func.FigNoise,
            'telescope_noise'     : Qlunc_UQ_Optics_func.UQ_Telescope,
            'telescope_losses'    : Qlunc_UQ_Optics_func.Losses_Telescope}


Methods={}

for k,v in Func.items():
    if k in list(salone.flatten(user_inputs.user_itype_noise)):
    
#        print (k)
        Methods.setdefault(k,Func[k](user_inputs,inputs,cts,**Scenarios))




dic={k:v for k,v in power_comp.items()}                     
Power_Mod=type('power',(),photonics_dic)
powere=Power_Mod()
powere.power_source.values()




u=['a','b']
uu=['a','c']

if ii in u and ii in uu:
    print (ii)







#%% fit data to a curve
raw_data= pd.read_csv(direct.Main_directory+inputs.photonics_inp.Photodetector_uncertainty_inputs['photodetector_Noise_FILE'],delimiter=';',decimal=',') 
dataY=raw_data.iloc[:,1]
dataX=raw_data.iloc[:,0]

fitt=np.poly1d(np.polyfit(dataX,dataY,5))
plt.plot(dataX,dataY,dataX,fitt(dataX))
noise=itg.quad(fitt,0,1000)



#%% Delta dirac
def DD(x,a):

    Delta= (np.exp(-(x/a)**2))/(abs(a)*np.sqrt(np.pi))
#    plt.figure
#    plt.plot(x,Delta)
    return Delta

#%% Fourier
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

#%% Normalilzation
import random
import statistics as stt
x=np.random.normal(25,25,1000)
y=np.random.normal(15,0.3,1000)

xmean=np.mean(x)
xSTDV=stt.stdev(x)
ymean=np.mean(y)
ySTDV=stt.stdev(y)


xnorm=(x-xmean)/xSTDV
ynorm=(y-ymean)/ySTDV

xw=(20-xmean)/xSTDV # Normalization. Then the data is comparable
yw=(17-ymean)/ySTDV

plt.figure()
cuenta, cajas, ignorar = plt.hist(xnorm,10)
cuenta, cajas, ignorar = plt.hist(ynorm)

plt.show()