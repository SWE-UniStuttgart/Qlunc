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
    xi               = []
    yi               = []
    zi               = []
    # rho_point        = []
    # theta_point      = []
    # phi_point        = []
    # x1               = []
    # y1               = []
    # z1               = []
    # rho_recon        = []
    # theta_recon      = []
    # phi_recon        = []
    # recon            = []
    Total_val_transf = []
    val_transf       = []
    val_or_transf    = []
    LOS_2_I          = []
    LOS_2_I_or       = []
    stdv_sph_points  = []
    Unc_U =[]
    Total_Unc_U=[]
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
        
        # x,y,z define the spheres
        x = Scan_Unc['MeasPoint_Coordinates'][0][r_sphere]+radius*np.outer(np.sin(theta), np.cos(phi))
        y = Scan_Unc['MeasPoint_Coordinates'][1][r_sphere]+radius*np.outer(np.sin(theta), np.sin(phi))
        z = Scan_Unc['MeasPoint_Coordinates'][2][r_sphere]+radius*np.outer(np.cos(theta), np.ones_like(phi))
        #xi, yi,zi define the points on the sphere surface
        xii, yii, zii = SA.sample_sphere(radius,1500)
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
        # rho1,phi1,theta1 = SA.cart2sph(xi0,yi0,zi0)       
        # rho_point.append(rho1)
        # phi_point.append(phi1)
        # theta_point.append(theta1)
        
        #Plot theoretical measuring points and random points on the sphere surface
        ax.plot(Scan_Unc['MeasPoint_Coordinates'][0][p_sphere],Scan_Unc['MeasPoint_Coordinates'][1][p_sphere],Scan_Unc['MeasPoint_Coordinates'][2][p_sphere],'ob',markersize=7.8)
        ax.plot(xi0,yi0,zi0,'or',markersize=4)
        
        
        ax.plot([Lidar.optics.scanner.origin[0]],[Lidar.optics.scanner.origin[1]],[Lidar.optics.scanner.origin[2]],'ob',label='{} coordinates [{},{},{}]'.format(Lidar.LidarID,Lidar.optics.scanner.origin[0],Lidar.optics.scanner.origin[1],Lidar.optics.scanner.origin[2]),markersize=5)

        # Just a check to see the transformation 
        # x1,y1,z1 = SA.sph2cart(rho1,phi1,theta1)
        
        
        # calculating angles between LOS and I-axes at the theoretical measuring point
        # module_or = np.linalg.norm([Scan_Unc['MeasPoint_Coordinates'][0][0],Scan_Unc['MeasPoint_Coordinates'][1][0],Scan_Unc['MeasPoint_Coordinates'][2][0]])
        # angx_or   = np.rad2deg(math.acos(Scan_Unc['MeasPoint_Coordinates'][0]/module_or))
        # angy_or   = np.rad2deg(math.acos(Scan_Unc['MeasPoint_Coordinates'][1]/module_or))
        # angz_or   = np.rad2deg(math.acos(Scan_Unc['MeasPoint_Coordinates'][2]/module_or))
        if Lidar.optics.scanner.scanner_type == 'SCAN':
            angley_or = np.rad2deg(math.atan(Scan_Unc['MeasPoint_Coordinates'][1][p_sphere]/Scan_Unc['MeasPoint_Coordinates'][0][p_sphere]))
            anglez_or = np.rad2deg(math.atan(Scan_Unc['MeasPoint_Coordinates'][2][p_sphere]/Scan_Unc['MeasPoint_Coordinates'][0][p_sphere]))
        elif Lidar.optics.scanner.scanner_type == 'VAD':
            angley_or = np.rad2deg(math.atan(Scan_Unc['MeasPoint_Coordinates'][1][p_sphere]/Scan_Unc['MeasPoint_Coordinates'][2][p_sphere]))
            anglez_or = np.rad2deg(math.atan(Scan_Unc['MeasPoint_Coordinates'][0][p_sphere]/Scan_Unc['MeasPoint_Coordinates'][2][p_sphere]))
        
        # Rotational matrix
        LOS_2_I_or= ((np.matrix([[np.cos(np.deg2rad(anglez_or))*np.cos(np.deg2rad(angley_or)), -np.cos(np.deg2rad(anglez_or))*np.sin(np.deg2rad(angley_or)), np.sin(np.deg2rad(anglez_or))   ],\
                                       [                    np.sin(np.deg2rad(angley_or)),                          np.cos(np.deg2rad(angley_or)),                               0                 ],\
                                       [ np.cos(np.deg2rad(anglez_or))*np.sin(np.deg2rad(angley_or)), -np.sin(np.deg2rad(anglez_or))*np.sin(np.deg2rad(angley_or)), np.cos(np.deg2rad(anglez_or))] ]))**-1)
        # pdb.set_trace()
        # Value of the transformation for the theoretical measurement point. Here assume v=w=0, that´s why it is sum up just the first row of the transformation matrix. Furthermore, since I am interested in the error of the transformation only, I
        # assume a unity vector to represent the LOS velocity. Therefore: 
        val_or_transf .append([LOS_2_I_or[p_sphere][0].sum()])
        
        # Wind field reconstruction 
        if Lidar.wfr_model.reconstruction_model=='Flat':
            val_transf=[]
            # Vectors
            # pdb.set_trace()
            Mes_vector_X =  xi0-Lidar.optics.scanner.origin[0]
            Mes_vector_Y =  yi0-Lidar.optics.scanner.origin[1]
            Mes_vector_Z =  zi0-Lidar.optics.scanner.origin[2]
            # pdb.set_trace()
            # Errors in measurement due to the location error (GPS)
            unc_Mes_vector_X,unc_Mes_vector_Y,unc_Mes_vector_Z = Lidar.optics.scanner.stdv_location
            
            # Find uncertainty for each point on the sphere surface
            for ind_ang in range(len(Mes_vector_X)):
            
                # Angles between vector and inertial axes
                angley = np.rad2deg(math.atan(Mes_vector_Y[ind_ang]/Mes_vector_X[ind_ang]))
                anglez = np.rad2deg(math.atan(Mes_vector_Z[ind_ang]/Mes_vector_X[ind_ang]))
                # Transformation matrix for the points on the sphere surface
                LOS_2_I= (np.matrix([[np.cos(np.deg2rad(anglez))*np.cos(np.deg2rad(angley)), -np.cos(np.deg2rad(anglez))*np.sin(np.deg2rad(angley)), np.sin(np.deg2rad(anglez))    ],\
                                     [                    np.sin(np.deg2rad(angley)),                          np.cos(np.deg2rad(angley)),                               0         ],\
                                     [ np.cos(np.deg2rad(anglez))*np.sin(np.deg2rad(angley)), -np.sin(np.deg2rad(anglez))*np.sin(np.deg2rad(angley)), np.cos(np.deg2rad(anglez))]  ]))**-1
                
                # Value of the transformation for the points on the sphere surface. Error due to pointing accuracy
                val_transf.append(LOS_2_I[0].sum())
                
                # Reconstruction uncertainty: Since final velocity is the result of a linear relation between transformation matrix and LOS velocities it is enough to assess the uncertainty of the reconstruction matrix, avoiding to give a 
                # value for the velocity vector. The uncertainty will be linearly proportional to the velocity vector:
                # Then, assuming v=w=0 only the first row of the reconstruction matrix is needed.
                cosaz  = np.cos(np.deg2rad(anglez))
                cosay  = np.cos(np.deg2rad(angley))
                sinaz  = np.sin(np.deg2rad(anglez))
                sinay  = np.sin(np.deg2rad(angley))                
                cosaz2 = (np.cos(np.deg2rad(anglez)))**2
                cosay2 = (np.cos(np.deg2rad(angley)))**2
                sinaz2 = (np.sin(np.deg2rad(anglez)))**2
                sinay2 = (np.sin(np.deg2rad(angley)))**2
                
                # Partial derivatives
                dU_dangleZ = -(((cosaz*cosay+cosaz2*sinay-cosay*sinaz-sinay*sinaz2)*(-cosay*cosaz2*sinay-2*cosay2*cosaz*sinaz-4*cosaz*sinaz*sinay2+cosay*sinay*sinaz2))/(cosay2*cosaz2+cosaz2*sinay2-cosay*cosaz*sinay*sinaz-sinay2*sinaz2)**2)+((-cosay*cosaz-cosay*sinaz-4*cosaz*sinay*sinaz)/(cosay2*cosaz2+cosaz2*sinay2-cosaz*cosay*sinaz*sinay-sinaz2*sinay2))
                dU_dangleY = -(((cosaz*cosay+cosaz2*sinay-cosay*sinaz-sinay*sinaz2)*(-cosay2*cosaz*sinaz+cosaz*sinay2*sinaz-2*cosay*sinay*sinaz2))/(cosay2*cosaz2+cosaz2*sinay2-cosay*cosaz*sinay*sinaz-sinay2*sinaz2)**2)+((cosay*cosaz2-cosaz*sinay+sinay*sinaz-cosay*sinaz2)/(cosay2*cosaz2+cosaz2*sinay2-cosaz*cosay*sinaz*sinay-sinaz2*sinay2))
                # atan(u)' = u'/(1+u^2)
                dangleY = np.sqrt((((1/Mes_vector_X[ind_ang])/(1+(Mes_vector_Y[ind_ang]/Mes_vector_X[ind_ang])**2))*unc_Mes_vector_Y)**2+(((Mes_vector_Y[ind_ang]/Mes_vector_X[ind_ang]**2)/(1+(Mes_vector_Y[ind_ang]/Mes_vector_X[ind_ang])**2))*unc_Mes_vector_X)**2)
                dangleZ = np.sqrt((((1/Mes_vector_X[ind_ang])/(1+(Mes_vector_Z[ind_ang]/Mes_vector_X[ind_ang])**2))*unc_Mes_vector_Z)**2+(((Mes_vector_Z[ind_ang]/Mes_vector_X[ind_ang]**2)/(1+(Mes_vector_Z[ind_ang]/Mes_vector_X[ind_ang])**2))*unc_Mes_vector_X)**2)
                
                # Unc_U is the error due to the reconstruction process
                Unc_U.append(np.sqrt((dU_dangleY*dangleY)**2+(dU_dangleZ*dangleZ)**2))
        # pdb.set_trace()
        Total_Unc_U.append(np.mean(Unc_U))
        Total_val_transf.append(val_transf)
        pdb.set_trace()
    for ind_std in range(len(Total_val_transf)):
        stdv_sph_points.append(np.std(Total_val_transf[ind_std]))
    
    pdb.set_trace()
        
        
    return (xi,yi,zi)