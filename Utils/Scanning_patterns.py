# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:10:47 2021
Lissajous' pattern
@author: fcosta
"""
import numpy as np
import matplotlib.pyplot as plt

#%% Lissajous psttern
def lissajous_pattern(size_x,size_y,size_z,q,p):
    # parameters
    # (1<p<q)
    t   = np.arange(0,2*np.pi,.01)
    phi = np.linspace(0,2*np.pi,len(t))
    
    # lissajous transformation
    x = size_x*np.sin((p*t+phi))
    y = size_y*np.sin((q*t))
    z = np.array(len(t)*[size_z])
    
    # plotting
    
    # 2D:
    # fig,ax1 = plt.subplots()
    # ax1.plot(x,y)
    # ax1.set_xlim(-30,30)
    # ax1.set_ylim(-30,30)
    
    # 3D
    fig,ax = plt.subplots()
    ax     = plt.axes(projection='3d')
    ax.plot(x,y,z)
    ax.set_xlim3d(-30,30)
    ax.set_ylim3d(-30,30)
    ax.set_zlim3d(-30,30)
  
    return x,y,z