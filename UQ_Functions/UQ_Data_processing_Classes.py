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

def UQ_WFR (Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs,Scan_Unc):
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
    rho_recon   = []
    theta_recon = []
    phi_recon   = []
    recon       = []
    val_transf  = []
    LOS_2_I     = []
    # pdb.set_trace()
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
    ax.set_xlim3d(-200,200)
    ax.set_ylim3d(-200,200)
    ax.set_zlim3d(-200,200)
    ax.set_xlabel('x',fontsize=15)
    ax.set_ylabel('y',fontsize=15)
    ax.set_zlabel('z',fontsize=15)
    for r_sphere in range (len(Scan_Unc['Simu_Mean_Distance_Error'])):
        # Sphere characteristics
        radius   = Scan_Unc['Simu_Mean_Distance_Error'][r_sphere]
        phi      = np.linspace(0, np.pi, 20)
        theta    = np.linspace(0, 2 * np.pi, 40)
        
        
        x = Scan_Unc['MeasPoint_Coordinates'][0][r_sphere]+radius*np.outer(np.sin(theta), np.cos(phi))
        y = Scan_Unc['MeasPoint_Coordinates'][1][r_sphere]+radius*np.outer(np.sin(theta), np.sin(phi))
        z = Scan_Unc['MeasPoint_Coordinates'][2][r_sphere]+radius*np.outer(np.cos(theta), np.ones_like(phi))
        
        xii, yii, zii = SA.sample_sphere(radius,5)
        xi.append(xii+Scan_Unc['MeasPoint_Coordinates'][0][r_sphere])
        yi.append(yii+Scan_Unc['MeasPoint_Coordinates'][1][r_sphere])
        zi.append(zii+Scan_Unc['MeasPoint_Coordinates'][2][r_sphere])
        
        # Plot spheres around the theoretical measuring points
        ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)

    # Coordinates of the points on the sphere surface for each point in the pattern  

    for p_sphere in range(len(zi)):
        xi0 = xi[p_sphere]
        yi0 = yi[p_sphere]
        zi0 = zi[p_sphere]
        
        # Transform coordinates to spherical coordinate system
        rho1,phi1,theta1 = SA.cart2sph(xi0,yi0,zi0)       
        # rho_point.append(rho1)
        # phi_point.append(phi1)
        # theta_point.append(theta1)
        
        #Plot theoretical measuring points and random points on the sphere surface
        ax.plot(Scan_Unc['MeasPoint_Coordinates'][0][p_sphere],Scan_Unc['MeasPoint_Coordinates'][1][p_sphere],Scan_Unc['MeasPoint_Coordinates'][2][p_sphere],'ob',markersize=7.8)
        ax.plot(xi0,yi0,zi0,'or',markersize=4)
        
        # pdb.set_trace()
        ax.plot([Lidar.optics.scanner.origin[0]],[Lidar.optics.scanner.origin[1]],[Lidar.optics.scanner.origin[2]],'ob',label='{} coordinates [{},{},{}]'.format(Lidar.LidarID,Lidar.optics.scanner.origin[0],Lidar.optics.scanner.origin[1],Lidar.optics.scanner.origin[2]),markersize=5)

        # Just a check to see the transformation 
        # x1,y1,z1 = SA.sph2cart(rho1,phi1,theta1)
        
        
        # calculating angles between LOS and I-axes at the theoretical measuring point
        # module_or = np.linalg.norm([Scan_Unc['MeasPoint_Coordinates'][0][0],Scan_Unc['MeasPoint_Coordinates'][1][0],Scan_Unc['MeasPoint_Coordinates'][2][0]])
        # angx_or   = np.rad2deg(math.acos(Scan_Unc['MeasPoint_Coordinates'][0]/module_or))
        # angy_or   = np.rad2deg(math.acos(Scan_Unc['MeasPoint_Coordinates'][1]/module_or))
        # angz_or   = np.rad2deg(math.acos(Scan_Unc['MeasPoint_Coordinates'][2]/module_or))
        angley_or = np.rad2deg(math.atan(Scan_Unc['MeasPoint_Coordinates'][1][0]/Scan_Unc['MeasPoint_Coordinates'][0][0]))
        anglez_or = np.rad2deg(math.atan(Scan_Unc['MeasPoint_Coordinates'][2][0]/Scan_Unc['MeasPoint_Coordinates'][0][0]))
        
        # Rotational matrix
        LOS_2_I_or.append ((np.matrix([[np.cos(np.deg2rad(anglez_or))*np.cos(np.deg2rad(angley_or)), -np.cos(np.deg2rad(anglez_or))*np.sin(np.deg2rad(angley_or)), np.sin(np.deg2rad(anglez_or))   ],\
                                       [                    np.sin(np.deg2rad(angley_or)),                          np.cos(np.deg2rad(angley_or)),                               0                 ],\
                                       [ np.cos(np.deg2rad(anglez_or))*np.sin(np.deg2rad(angley_or)), -np.sin(np.deg2rad(anglez_or))*np.sin(np.deg2rad(angley_or)), np.cos(np.deg2rad(anglez_or))] ]))**-1)
        
        # Value of the transformation for the theoretical measurement point. Here assume v=w=0, that´s why it is sum up just the first row of the transformation matrix. Furthermore, since I am interested in the error of the transformation only, I
        # assume a unity vector to represent the LOS velocity. Therefore: 
        val_or_transf = LOS_2_I_or[0].sum()
        
        # Wind field reconstruction 
        if Lidar.wfr_model.reconstruction_model=='Flat':
            
            # Vectors
            # pdb.set_trace()
            Mes_vector_X =  xi0-Lidar.optics.scanner.origin[0]
            Mes_vector_Y =  yi0-Lidar.optics.scanner.origin[1]
            Mes_vector_Z =  zi0-Lidar.optics.scanner.origin[2]
            for ind_ang in range(len(Mes_vector_X)):
            
                # Angles between vector and inertial axes
                angley = np.rad2deg(math.atan(Mes_vector_Y[ind_ang]/Mes_vector_X[ind_ang]))
                anglez = np.rad2deg(math.atan(Mes_vector_Z[ind_ang]/Mes_vector_X[ind_ang]))
                # Transformation matrix for the points on the sphere surface
                LOS_2_I.append ((np.matrix([[np.cos(np.deg2rad(anglez))*np.cos(np.deg2rad(angley)), -np.cos(np.deg2rad(anglez))*np.sin(np.deg2rad(angley)), np.sin(np.deg2rad(anglez))   ],\
                                      [                    np.sin(np.deg2rad(angley)),                          np.cos(np.deg2rad(angley)),                               0                 ],\
                                      [ np.cos(np.deg2rad(anglez))*np.sin(np.deg2rad(angley)), -np.sin(np.deg2rad(anglez))*np.sin(np.deg2rad(angley)), np.cos(np.deg2rad(anglez))] ]))**-1)
                
                    # Value of the transformation for the points on the sphere surface
                val_transf.append( LOS_2_I[ind_ang][0].sum())
    
        pdb.set_trace()
        
        
    return (xi,yi,zi)