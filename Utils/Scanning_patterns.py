# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:10:47 2021
Lissajous' pattern
@author: fcosta
"""

from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
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
    # pdb.set_trace()
    z = size_x*np.sin((p*t+phi))
    y = size_y*np.sin((q*t))
    x = np.array(len(t)*[size_z])
    return x,y,z

def Verticalplane_pattern(x_in,y_in,z_in):
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
    box=np.meshgrid(y_in,z_in)
    
    # Get coordinates of the points on the grid
    box_positions = np.vstack(map(np.ravel, box))
    y_out=box_positions[0]
    z_out=box_positions[1]   
    x_out= np.linspace(x_in,x_in,len(y_out))   
    return x_out,y_out,z_out

