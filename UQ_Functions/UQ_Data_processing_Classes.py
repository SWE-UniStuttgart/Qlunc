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
<<<<<<< Updated upstream
        ax.scatter(xi, yi, zi, s=10, c='r', zorder=10)
        ax.set_xlabel('x',fontsize=15)
        ax.set_ylabel('y',fontsize=15)
        ax.set_zlabel('z',fontsize=15)
        # pdb.set_trace()
    # Coordinates of the points on the sphere surface
=======

    
    # Coordinates of the points on the sphere surface for each point in the pattern  
>>>>>>> Stashed changes
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
        
        module=[]
        angx=[]
        angy=[]
        angz=[]
        # Wind field reconstruction 
        if Lidar.wfr_model.reconstruction_model=='Flat':
            # It is assumed v=w=0
            # recon.append( rho1+phi1+theta1)
            # Vectors
            pdb.set_trace()
            Mes_vector_X =  xi0-Lidar.optics.scanner.origin[0]
            Mes_vector_Y =  yi0-Lidar.optics.scanner.origin[1]
            Mes_vector_Z =  zi0-Lidar.optics.scanner.origin[2]
            for ind_ang in range(len(Mes_vector_X)):
                module.append(np.linalg.norm([Mes_vector_X[ind_ang],Mes_vector_Y[ind_ang],Mes_vector_Z[ind_ang]]))
            
                # Angles between vector and inertial axes
                # angx.append(math.acos(Mes_vector_X[ind_ang]/module[ind_ang]))
                # angy.append(math.acos(Mes_vector_Y[ind_ang]/module[ind_ang]))
                # angz.append(math.acos(Mes_vector_Z[ind_ang]/module[ind_ang]))
                angx.append(np.rad2deg(math.atan(xi0[ind_ang]/module[ind_ang])))

                angy.append(np.rad2deg(math.atan(yi0[ind_ang]/module[ind_ang])))
                angz.append(np.rad2deg(math.atan(zi0[ind_ang]/module[ind_ang])))
                
                angx=np.rad2deg(math.atan(Mes_vector_Z))
            
            
            # In_2_LOS_matrix= [np.cos(anglez(iTra))*cosd(angley(iTra))   -cosd(anglez(iTra))*sind(angley(iTra))   sind(anglez(iTra));
            #                  sind(angley(iTra))                      cosd(angley(iTra))                                 0  ;
            #                  sind(angley(iTra))*cosd(anglez(iTra))   -sind(anglez(iTra))*sind(angley(iTra))   cosd(anglez(iTra)) ]; 
            

    
    pdb.set_trace()
        
        
    return (xi,yi,zi)