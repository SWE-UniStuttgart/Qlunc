# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:57:04 2020

@author: fcosta
"""
import pandas as pd
import sys,inspect
from functools import reduce
from operator import getitem
import numpy as np
from Qlunc_inputs import cts
#import Qlunc_ImportModules
#import Qlunc_Help_standAlone as SA
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (list,tuple)) else (a,))) 
import pdb



R=.85*cts.e*1550e-9/(cts.h*cts.c)
Temperature=300
Photodetector_RL=50
Photodetector_Bandwidth=1e9
Photodetector_DarkCurrent=5e-9
Photodetector_Signal_Power=.001
#Thermal
Photodetector_Thermal_noise2=(((4*cts.k*Temperature/Photodetector_RL)*Photodetector_Bandwidth)) 
Photodetector_Thermal_noise_dB2=10*np.log10(Photodetector_Thermal_noise2)

#Dark:
Photodetector_Dark_current_noise2=2*cts.e*Photodetector_DarkCurrent*Photodetector_Bandwidth
Photodetector_Dark_current_noise_dB2=10*np.log10(Photodetector_Dark_current_noise2)

#shot
Shot_noise2=2*cts.e*R*Photodetector_Signal_Power*Photodetector_Bandwidth
Shot_noise_dB2=10*np.log10(Shot_noise2)


#Photodetector_SNR_Shot_noise=(R**2)/(2*cts.e*R*Photodetector_Bandwidth)


SUM_noise_dB2=SA.Sum_dB([Photodetector_Thermal_noise_dB2,Photodetector_Dark_current_noise_dB2,Shot_noise_dB2])

def tre(x,r):
    global g
    g=x+r

rr=tre(1,2)
rr.g