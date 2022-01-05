# -*- coding: utf-8 -*-

""".

Created on Mon Nov 29 00:13:52 2021
@author: fcosta

Francisco Costa García
University of Stuttgart(c)

Here we calculate the uncertainties related with lidar data processing methods

    
   - Wind field reconstruction methods
   - Filtering processes
   
 
"""
from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
from Utils import Qlunc_Plotting as QPlot



#%% Wind Field Reconstruction methods

def UQ_WFR (Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs,Scanner_Uncertainty):
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
    * Scanner uncertainty
        Information about the points to create the sphere
    Returns
    -------
    
    Dictionary with information about...
    
    """        
    xi          = []
    yi          = []
    zi          = []
    rho_point   = []
    theta_point = []
    phi_point   = []
    x1          = []
    y1          = []
    z1          = []
    
    for r_sphere in range (len(Scanner_Uncertainty['Simu_Mean_Distance_Error'])):
        rho   = Scanner_Uncertainty['Simu_Mean_Distance_Error'][r_sphere]
        phi   = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        x = Scanner_Uncertainty['MeasPoint_Coordinates'][0][r_sphere]+rho*np.outer(np.sin(theta), np.cos(phi))
        y = Scanner_Uncertainty['MeasPoint_Coordinates'][1][r_sphere]+rho*np.outer(np.sin(theta), np.sin(phi))
        z = Scanner_Uncertainty['MeasPoint_Coordinates'][2][r_sphere]+rho*np.outer(np.cos(theta), np.ones_like(phi))
        
        xii, yii, zii = SA.sample_spherical(5)
        xi.append(xii+Scanner_Uncertainty['MeasPoint_Coordinates'][0][r_sphere])
        yi.append(yii+Scanner_Uncertainty['MeasPoint_Coordinates'][1][r_sphere])
        zi.append(zii+Scanner_Uncertainty['MeasPoint_Coordinates'][2][r_sphere])
        
        # fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
        # ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
        # ax.scatter(xi, yi, zi, s=10, c='r', zorder=10)
        # ax.set_xlabel('x',fontsize=15)
        # ax.set_ylabel('y',fontsize=15)
        # ax.set_zlabel('z',fontsize=15)
    
    # Coordinates of the points on the sphere surface
    for p_sphere in range(len(zi)):
        xi0 = xi[p_sphere]
        yi0 = yi[p_sphere]
        zi0 = zi[p_sphere]
        rho1,phi1,theta1 = SA.cart2sph(xi0,yi0,zi0)
        
        rho_point.append(rho1)
        phi_point.append(phi1)
        theta_point.append(theta1)
        
        # Just a check to see if the transformation 
        x1,y1,z1 = SA.sph2cart(rho1,phi1,theta1)
        
        
        
    return (xi,yi,zi)