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
    z = size_x*np.sin((p*t+phi))+120
    y = size_y*np.sin((q*t))
    x = np.array(len(t)*[size_z])
    # pdb.set_trace()
    
   
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

def Verticalplane_pattern(x_in,y_in,z_in):
    # pdb.set_trace()
    # xx,yy,zz=SA.sph2cart(Lidar.optics.scanner.focus_dist,Lidar.optics.scanner.cone_angle,Lidar.optics.scanner.azimuth)
    
    x_in = 100
    y_in = np.linspace(-2000,2000,5)
    z_in = np.linspace(1,2300,5)
    box=np.meshgrid(y_in,z_in)
    
    # Get coordinates of the points on the grid
    box_positions = np.vstack(map(np.ravel, box))
    y_out=box_positions[0]
    z_out=box_positions[1]   
    x_out= np.linspace(x_in,x_in,len(y_out))
    return x_out,y_out,z_out

# fig,axs5 = plt.subplots()  
# axs5=plt.axes(projection='3d')
# axs5.plot(xx,yy,zz,'bo')
# axs5.set_xlabel('X',fontsize=25)
# axs5.set_ylabel('Y',fontsize=25)
# axs5.set_zlabel('Z',fontsize=25)