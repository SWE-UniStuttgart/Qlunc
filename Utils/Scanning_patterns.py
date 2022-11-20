# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:10:47 2021
Lissajous' pattern
@author: fcosta
"""

from Utils.Qlunc_ImportModules import *

#%% Lissajous pattern
def lissajous_pattern(Lidar,size_x,size_y,size_z,q,p):
    """
    Plotting. Location: .Utils/Qlunc_plotting.py
    
    Parameters
    ----------
    
    * Lidar
        data...
        
    Returns
    -------
    
    list
    
    """
    # parameters
    # (1<p<q)
    # t   = np.arange(0,2*np.pi,.01)
    t=np.linspace(0,2*np.pi,math.floor(Lidar.optics.scanner.time_pattern/Lidar.optics.scanner.time_point))
    phi = np.linspace(0,2*np.pi,len(t))
    
    # lissajous transformation
    z = size_x*np.sin((p*t+phi))
    y = size_y*np.sin((q*t))
    x = np.array(len(t)*[size_z])
   
    # plotting

    # 2D:
    # fig,ax1 = plt.subplots()
    # ax1.plot(x,y)
    # ax1.set_xlim(-30,30)
    # ax1.set_ylim(-30,30)
    
    # 3D
    # fig,ax = plt.subplots()
    # ax     = plt.axes(projection='3d')
    # ax.plot(x,y,z)
    # ax.set_xlim3d(-30,30)
    # ax.set_ylim3d(-30,30)
    # ax.set_zlim3d(-30,30)
    # pdb.set_trace()
    return x,y,z