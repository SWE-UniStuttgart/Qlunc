# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:15:17 2020

@author: fcosta
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Source:https://www.telescope-optics.net/diffraction_image.htm
Focal_length      = 1 # f [meters]
Aperture_diameter = 1.12 # D [meters]
Focal_ratio       = Focal_length/Aperture_diameter  # F [meters]
Wavelength        = 1550e-9

rAD               = 1.22*Wavelength*Focal_ratio


r=np.arange(-3,3,0.005)#radius of the image pattern



t=np.pi*r/2


I=(1-((t**2)/(math.factorial(1)*math.factorial(2)))+((t**4)/(math.factorial(2)*math.factorial(3)))-
     ((t**6)/(math.factorial(3)*math.factorial(4)))+((t**8)/(math.factorial(4)*math.factorial(5)))-
     ((t**10)/(math.factorial(5)*math.factorial(6)))+((t**12)/(math.factorial(6)*math.factorial(7)))-
     ((t**14)/(math.factorial(7)*math.factorial(8)))+((t**16)/(math.factorial(8)*math.factorial(9)))-
     ((t**18)/(math.factorial(9)*math.factorial(10)))+((t**20)/(math.factorial(10)*math.factorial(11)))-
     ((t**22)/(math.factorial(11)*math.factorial(12)))+((t**24)/(math.factorial(12)*math.factorial(13))))**2

EE=1-((1-((t**2)/(math.factorial(1)**2))+((t**4)/(math.factorial(2)**2))-
     ((t**6)/(math.factorial(3)**2))+((t**8)/(math.factorial(4)**2))-
     ((t**10)/(math.factorial(5)**2))+((t**12)/(math.factorial(6)**2))-
     ((t**14)/(math.factorial(7)**2))+((t**16)/(math.factorial(8)**2))-
     ((t**18)/(math.factorial(9)**2))+((t**20)/(math.factorial(10)**2))-
     ((t**22)/(math.factorial(11)**2))+((t**24)/(math.factorial(12)**2)))**2)-I*t**2

plt.figure()
plt.title('PSF')
plt.ylabel('Normalized Intensity')
plt.xlabel('Image Radius ($\lambda$F)')
plt.yscale('log',basey=10)

plt.plot(r,I,r,EE)
Sorted=np.sort(I)

minima = argrelextrema(I, np.less_equal)     # Min values of I
maxima = argrelextrema(I, np.greater_equal)  # Max values of I



