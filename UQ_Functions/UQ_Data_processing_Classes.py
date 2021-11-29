# -*- coding: utf-8 -*-

""".

Created on Mon Nov 29 00:13:52 2021
@author: fcosta

Francisco Costa García
University of Stuttgart(c)

Here we calculate the uncertainties related with lidar data processing methods

    
   - noise definitions (reference in literature)
   
 
"""
from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
from Utils import Qlunc_Plotting as QPlot



#%% Wind Field Reconstruction methods

def UQ_WFR (Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    """.
    
    Wind field reconstruction methods. Location: ./UQ_Functions/UQ_Data_processing_Classes.py   
    Parameters
    ----------
    
    * Lidar
        data...
    * Atmospheric_Scenario
        Atmospheric data. Integer or Time series
    * cts
        Physical constants
    * Qlunc_yaml_inputs
        Lidar parameters data        
    Returns
    -------
    
    Dictionary with information about...
    
    """        
    pdb.set_trace()
    r = 20
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = r*np.outer(np.sin(theta), np.cos(phi))
    y = r*np.outer(np.sin(theta), np.sin(phi))
    z = r*np.outer(np.cos(theta), np.ones_like(phi))
    
    xi, yi, zi = r*sample_spherical(9000)
    
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    ax.scatter(xi, yi, zi, s=10, c='r', zorder=10)