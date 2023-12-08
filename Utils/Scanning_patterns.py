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
    Calculates the coordinates of the points in a lissajous pattern. Location: .Utils/Scanning_patterns.py
    
    Parameters
    ----------
    
    * Lidar
        Dictionary containing lidar info
        
    Returns
    -------
    
    x_out,y_out,z_out arrays
    
    """
    # parameters
    # (1<p<q)
    t   = np.arange(0,2*np.pi,.01)
    # t=np.linspace(0,2*np.pi,math.floor(Lidar.optics.scanner.time_pattern/Lidar.optics.scanner.time_point))
    phi = np.linspace(0,2*np.pi,len(t))
    
    # lissajous transformation
    # pdb.set_trace()
    z = size_x*np.sin((p*t+phi))
    y = size_y*np.sin((q*t))
    x = np.array(len(t)*[size_z])
    return x,y,z

def Verticalplane_pattern(Lidar):
    """
    Calculates the coordinates of the points in a vertical plane. Location: .Utils/Scanning_patterns.py
    
    Parameters
    ----------
    
    * Lidar
        Dictionary containing lidar info
        
    Returns
    -------
    
    x_out,y_out,z_out arrays
    
    """
    x_in = Lidar.optics.scanner.vert_plane[0]
    y_in = np.linspace(Lidar.optics.scanner.vert_plane[1],Lidar.optics.scanner.vert_plane[2],Lidar.optics.scanner.vert_plane[-1])
    z_in = np.linspace(Lidar.optics.scanner.vert_plane[3],Lidar.optics.scanner.vert_plane[4],Lidar.optics.scanner.vert_plane[-1])

    # Get coordinates of the points in the grid
    y_out_M,z_out_M= (np.meshgrid(y_in,z_in))
    y_out          = np.ravel(y_out_M)
    z_out          = np.ravel(z_out_M)
    x_out= np.linspace(x_in,x_in,len(y_out)) 
    return x_out,y_out,z_out

def Horizontalplane_pattern(Lidar):
    """
    Calculates the coordinates of the points in a horizontal plane. Location: .Utils/Scanning_patterns.py
    
    Parameters
    ----------
    
    * Lidar
        Dictionary containing lidar info
        
    Returns
    -------
    
    x_out,y_out,z_out arrays
    
    """
    x_in = np.linspace(Lidar.optics.scanner.hor_plane[0],Lidar.optics.scanner.hor_plane[1],Lidar.optics.scanner.hor_plane[-1]) #Lidar.optics.scanner.vert_plane[0]
    y_in = np.linspace(Lidar.optics.scanner.hor_plane[2],Lidar.optics.scanner.hor_plane[3],Lidar.optics.scanner.hor_plane[-1])
    z_in = Lidar.optics.scanner.vert_plane[4]

    box=np.meshgrid(x_in,y_in)
    # pdb.set_trace()
    # Get coordinates of the points on the grid
    box_positions = np.vstack(map(np.ravel, box))
    y_out=box_positions[0]
    x_out=box_positions[1]   
    z_out= np.linspace(z_in,z_in,len(y_out))   
    return x_out,y_out,z_out