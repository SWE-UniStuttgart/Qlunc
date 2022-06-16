# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:04:17 2022

@author: fcosta
"""
# -*- coding: utf-8 -*-

""".

Created on Sat May 16 14:58:24 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c)

Here we calculate the uncertainties related with components in the `optics`
module. 

    
   - noise definitions (reference in literature)
   
 
"""

from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
from Utils import Scanning_patterns as SP
from Utils import Qlunc_Plotting as QPlot

#%% SCANNER:
def UQ_Scanner(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    """.
    
    Scanner uncertainty estimation. Location: ./UQ_Functions/UQ_Optics_Classes.py   
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
    Coord=[]
    StdvMean_DISTANCE=[]  
    SimMean_DISTANCE=[]
    X,Y,Z,X0,Y0,Z0,R=[],[],[],[],[],[],[]
    Noisy_Coord=[]
    NoisyX=[]
    NoisyY=[]
    NoisyZ=[]
    coorFinal_noisy=[]
    rho_noisy0,theta_noisy0,psi_noisy0,rho_noisy,theta_noisy,psi_noisy,rho_noisy1, theta_noisy,psi_noisy1,wind_direction_noisy ,wind_pitch_noisy   = [],[],[],[],[],[],[],[],[],[],[]
    Coordfinal_noisy,Coordfinal=[],[]
    coun=0
    sample_rate_count=0
    Href= Qlunc_yaml_inputs['Components']['Scanner']['Href'],
    Vref= Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']
    alpha =  Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp']
    
    # #Call probe volume uncertainty function. 
    # Probe_param = Lidar.probe_volume.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)


    # R: Implement error in deployment of the tripod as a rotation over yaw, pitch and roll
    stdv_yaw    = np.array(np.radians(Lidar.lidar_inputs.yaw_error_dep))
    stdv_pitch  = np.array(np.radians(Lidar.lidar_inputs.pitch_error_dep))
    stdv_roll   = np.array(np.radians(Lidar.lidar_inputs.roll_error_dep))
    
    # Lidar position:
    x_Lidar = Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0]
    y_Lidar = Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1]
    z_Lidar = Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2]
    
    # Rho, theta and phi values
    rho = Lidar.optics.scanner.focus_dist    
    theta = Lidar.optics.scanner.cone_angle
    psi = Lidar.optics.scanner.azimuth
    wind_direction = np.array([Atmospheric_Scenario.wind_direction]*len(psi))
    wind_pitch = np.array([Atmospheric_Scenario.wind_pitch]*len(psi))
    
    # In cartesian coordinates
    x,y,z=SA.sph2cart(rho,np.radians(psi),np.radians(np.array(90)-theta))
    
    # Add lidar position:
    x=x+x_Lidar
    y=y+y_Lidar
    z=z+z_Lidar
    
    # stdv focus distance, cone angle and azimuth:
    # stdv_param1 = Probe_param['Focus Distance uncertainty']
    stdv_rho   = Lidar.optics.scanner.stdv_focus_dist    
    stdv_theta = Lidar.optics.scanner.stdv_cone_angle
    stdv_psi   = Lidar.optics.scanner.stdv_azimuth
    stdv_wind_direction =0
    stdv_wind_pitch = 0
    # Differentiate between 'VAD' or 'Scanning' lidar depending on user's choice:
    # if Qlunc_yaml_inputs['Components']['Scanner']['Type']=='VAD':
    #     param1=Lidar.optics.scanner.focus_dist
    #     # param1 = [np.array(Probe_param['Focus Distance'])]
    #     param2 = np.deg2rad(Lidar.optics.scanner.cone_angle)
    #     param3 = np.deg2rad(Lidar.optics.scanner.azimuth)
        
    # elif Qlunc_yaml_inputs['Components']['Scanner']['Type']=='SCAN':
        
    #     # 'Transform coordinates from cartesians to spherical'
    #     param1=[]
    #     param2=[]
    #     param3=[]
        
    #     # When SCAN is selected user can choose specific patterns already implemented (./Qlunc/Utils/Scanning_patterns.py)
    #     if Qlunc_yaml_inputs['Components']['Scanner']['Pattern']=='lissajous':
    #         # x_init,y_init,z_init = SP.lissajous_pattern(Lidar.optics.scanner.lissajous_param[0],Lidar.optics.scanner.lissajous_param[1],Lidar.optics.scanner.lissajous_param[2],Lidar.optics.scanner.lissajous_param[3],Lidar.optics.scanner.lissajous_param[4])
            
    #         # x_init =np.array( [Probe_param['Focus Distance']])
    #         x_init,y_init,z_init = SP.lissajous_pattern(Lidar,Lidar.optics.scanner.lissajous_param[0],Lidar.optics.scanner.lissajous_param[1],Lidar.optics.scanner.lissajous_param[2],Lidar.optics.scanner.lissajous_param[3],Lidar.optics.scanner.lissajous_param[4])
        
    #     elif Qlunc_yaml_inputs['Components']['Scanner']['Pattern']=='None':
    #         x_init = Lidar.optics.scanner.x
    #         # x_init = np.array([Probe_param['Focus Distance']]) # This needs to be changed
    #         y_init = Lidar.optics.scanner.y
    #         z_init = Lidar.optics.scanner.z
            
    #     # Calculating parameter1, parameter2 and parameter3 depending on the quadrant (https://es.wikipedia.org/wiki/Coordenadas_esf%C3%A9ricas):           
    #     param1,param3,param2=SA.cart2sph(x_init,y_init,z_init)
    #     # xc,yc,zc=SA.sph2cart(param1,param3,param2)
        
    
    # Find the noisy points for the initial coordinates:
    for ind_noise in range(Lidar.optics.scanner.N_Points):
        rho_noisy.append(np.random.normal(rho[ind_noise],stdv_rho,Lidar.optics.scanner.N_MC))
        theta_noisy.append(np.random.normal(theta[ind_noise],stdv_theta,Lidar.optics.scanner.N_MC))
        psi_noisy.append(np.random.normal(psi[ind_noise],stdv_psi,Lidar.optics.scanner.N_MC))
        wind_direction_noisy.append(np.random.normal(wind_direction[ind_noise],stdv_wind_direction,Lidar.optics.scanner.N_MC))
        wind_pitch_noisy.append(np.random.normal(wind_pitch[ind_noise],stdv_wind_pitch,Lidar.optics.scanner.N_MC))
        
        
        # # This part commented for now id to calculate the error due to the inclinomenters:
            
        # rho_noisy0.append(np.random.normal(rho[ind_noise],stdv_rho,Lidar.optics.scanner.N_MC))
        # theta_noisy0.append(np.random.normal(theta[ind_noise],stdv_theta,Lidar.optics.scanner.N_MC))
        # psi_noisy0.append(np.random.normal(psi[ind_noise],stdv_psi,Lidar.optics.scanner.N_MC))
        
        # # Apply error in inclinometers   
        # # Rotation, due to inclinometers
        # noisy_yaw     = np.random.normal(0,stdv_yaw,Lidar.optics.scanner.N_MC)
        # noisy_pitch   = np.random.normal(0,stdv_pitch,Lidar.optics.scanner.N_MC)
        # noisy_roll    = np.random.normal(0,stdv_roll,Lidar.optics.scanner.N_MC)
        
        
        # # Convert noisy coordinates into cartesians to apply the rotation matrices
        # x_noisy, y_noisy,z_noisy = SA.sph2cart(rho_noisy0[ind_noise],np.radians(psi_noisy0[ind_noise]),np.radians(np.array(90)-theta_noisy0[ind_noise]))
        
        # # Create the rotate matrix to apply the error in inclinometers
        # for ind_inclinometer in range(len(noisy_yaw)):
        #     R=(np.array([[np.cos(noisy_yaw[ind_inclinometer])*np.cos(noisy_pitch[ind_inclinometer])  ,  np.cos(noisy_yaw[ind_inclinometer])*np.sin(noisy_pitch[ind_inclinometer])*np.sin(noisy_roll[ind_inclinometer])-np.sin(noisy_yaw[ind_inclinometer])*np.cos(noisy_roll[ind_inclinometer])  ,  np.cos(noisy_yaw[ind_inclinometer])*np.sin(noisy_pitch[ind_inclinometer])*np.cos(noisy_roll[ind_inclinometer])+np.sin(noisy_yaw[ind_inclinometer])*np.sin(noisy_roll[ind_inclinometer])],
        #               [np.sin(noisy_yaw[ind_inclinometer])*np.cos(noisy_pitch[ind_inclinometer])  ,  np.sin(noisy_yaw[ind_inclinometer])*np.sin(noisy_pitch[ind_inclinometer])*np.sin(noisy_roll[ind_inclinometer])+np.cos(noisy_yaw[ind_inclinometer])*np.cos(noisy_roll[ind_inclinometer])  ,  np.sin(noisy_yaw[ind_inclinometer])*np.sin(noisy_pitch[ind_inclinometer])*np.cos(noisy_roll[ind_inclinometer])-np.cos(noisy_yaw[ind_inclinometer])*np.sin(noisy_roll[ind_inclinometer])],
        #               [       -np.sin(noisy_pitch[ind_inclinometer])               ,  np.cos(noisy_pitch[ind_inclinometer])*np.sin(noisy_roll[ind_inclinometer])                                                                  ,  np.cos(noisy_pitch[ind_inclinometer])*np.cos(noisy_roll[ind_inclinometer])]]))
            
        #     # Rotation                    
        #     Coordfinal_noisy.append(np.matmul(R, np.array([x_noisy[ind_inclinometer],y_noisy[ind_inclinometer],z_noisy[ind_inclinometer]])))
            
        # xx_noisy,yy_noisy,zz_noisy=[],[],[]
        # # pdb.set_trace()
        # # Apply the rotation to the original points
        # # Coordfinal.append(np.matmul(R, np.array([x[ind_noise],y[ind_noise],z[ind_noise]])))
        # coorFinal_noisy.append(Coordfinal_noisy)   
        # for ix in range(len(Coordfinal_noisy)):
        #     xx_noisy.append(Coordfinal_noisy[ix][0])
        #     yy_noisy.append(Coordfinal_noisy[ix][1])
        #     zz_noisy.append(Coordfinal_noisy[ix][2])
        
        # rho_noisy1, theta_noisy1,psi_noisy1 = SA.cart2sph(xx_noisy,yy_noisy,zz_noisy)
        
        # # Store the noisy spherical coordinates including the error in inclinometers
        # rho_noisy.append(rho_noisy1)
        # theta_noisy.append(np.array(90)-theta_noisy1)
        # psi_noisy.append(psi_noisy1)
        # Coordfinal_noisy=[]      
    

       

    #%% MC Method 
    
    
    # 1. Calculate radial speed uncertainty for an homogeneous flow
    U_Vrad_homo_MC = []
    
    # 1.1 Relative uncertainty:
    # Vrad_homo = ([100*np.cos(np.radians(theta_noisy[ind_theta]))*np.cos(np.radians(psi_noisy[ind_theta]))/(np.cos(np.radians(theta[ind_theta]))*np.cos(np.radians(psi[ind_theta]))) for ind_theta in range (len(theta_noisy))])    
    
    
    # 1.2 Absolute uncertainty:
    # Vrad_homo = ([Vref*np.cos(np.radians(theta_noisy[ind_theta]))*np.cos(np.radians(psi_noisy[ind_theta])) for ind_theta in range (len(theta_noisy))])
    
    
    # 1.3 New approach(Absolute uncertainty):
    Vrad_homo = ([Vref*(-np.cos(np.radians(theta_noisy[ind_theta]))*np.cos(np.radians(psi_noisy[ind_theta]))*np.cos(np.radians(wind_direction_noisy[ind_theta]))-np.cos(np.radians(theta_noisy[ind_theta]))*np.sin(np.radians(psi_noisy[ind_theta]))*np.sin(np.radians(wind_direction_noisy[ind_theta]))+np.sin(np.radians(theta_noisy[ind_theta]))*np.tan(np.radians(wind_pitch_noisy[ind_theta]))) for ind_theta in range (len(theta_noisy))])
            
    
    # 1.4 Uncertainty (stdv):
    U_Vrad_homo_MC.append([np.std(Vrad_homo[ind_stdv])  for ind_stdv in range(len(Vrad_homo))])
        

   # 2 Uncertainty (power law)
    U_Vh_PL,U_Vrad_S_MC,Vrad_PL,Vrad_PL0=[],[],[],[]
    for in_alpha in range(len(alpha)):   
        for ind_npoints in range(len(rho)): # Calculate the radial speed uncertainty for the noisy points 
        
            # 2.1 Relative uncertainty:        
            # Vrad_PL.append (100*(np.cos(np.radians(psi_noisy[ind_npoints]))*np.cos(np.radians(theta_noisy[ind_npoints])))*(((z_Lidar+np.sin(np.radians(theta_noisy[ind_npoints]))*rho_noisy[ind_npoints])/Href)**alpha)\
            #                     /((np.cos(np.radians(psi[ind_npoints]))*np.cos(np.radians(theta[ind_npoints])))*(((z_Lidar+np.sin(np.radians(theta[ind_npoints]))*rho[ind_npoints])/Href)**alpha)))
            
            # 2.2 Absolute uncertainty:
            # Vrad_PL.append (Vref*(np.cos(np.radians(psi_noisy[ind_npoints]))*np.cos(np.radians(theta_noisy[ind_npoints])))*(((Href+np.sin(np.radians(theta_noisy[ind_npoints]))*rho_noisy[ind_npoints])/Href)**alpha[0]))
            
            # 2.3 New approach (Absolute uncertainty):
            Vrad_PL.append(Vref*(((Href+(np.sin(np.radians(theta_noisy[ind_npoints]))*rho_noisy[ind_npoints]))/Href)**alpha[in_alpha])*(-np.cos(np.radians(theta_noisy[ind_npoints]))*np.cos(np.radians(psi_noisy[ind_npoints]))*np.cos(np.radians(wind_direction_noisy[ind_npoints]))-np.cos(np.radians(theta_noisy[ind_npoints]))*np.sin(np.radians(psi_noisy[ind_npoints]))*np.sin(np.radians(wind_direction_noisy[ind_npoints]))+np.sin(np.radians(theta_noisy[ind_npoints]))*np.tan(np.radians(wind_pitch_noisy[ind_npoints]))))
        
        
        # 2.4 Uncertainty (stdv): For this to be compared with Vrad_weighted[1] I need to weight Vrad_PL 
        U_Vrad_S_MC.append([np.nanstd(Vrad_PL[ind_stdv]) for ind_stdv in range(len(Vrad_PL))])
        Vrad_PL=[]
    
    
    # Scatter plot: calculating all values among the range of theta, psi and rho
    Vrad0_PL,U_Vrad0_S_MC=[],[]   
    Vrad0_PL_T= ([Vref*(((Href+(np.sin(np.radians(theta_noisy0))*rho_noisy0))/Href)**alpha[0])*(-np.cos(np.radians(theta_noisy0))*np.cos(np.radians(psi_noisy0))*np.cos(np.radians(0))-np.cos(np.radians(theta_noisy0))*np.sin(np.radians(psi_noisy0))*np.sin(np.radians(0))+np.sin(np.radians(theta_noisy0))*np.tan(np.radians(0)))  for theta_noisy0 in theta_noisy  for rho_noisy0 in rho_noisy for psi_noisy0 in psi_noisy])   
    # rfr= ([(theta_noisy0,psi_noisy0,rho_noisy0) for theta_noisy0 in theta_noisy for psi_noisy0 in psi_noisy for rho_noisy0 in rho_noisy])   

    U_Vrad0_S_MC.append([np.nanstd(Vrad0_PL_T[ind_T]) for ind_T in range(len(Vrad0_PL_T))])
    # pdb.set_trace()
    #%% GUM method
    
    # 1. Calculate radial speed uncertainty for an homogeneous flow
    U_Vrad_homo_GUM,U_Vrad_theta,U_Vrad_psi,U_Vh,U_Vrad_range=[],[],[],[],[]
    
    # 1.1 Relative Unceratinty (%)
    # U_Vrad_theta.append([100*np.tan(np.radians(theta[ind_u]))*np.radians(stdv_theta) for ind_u in range(len(theta))])    
    # U_Vrad_psi.append([100*np.tan(np.radians(psi[ind_u]))*np.radians(stdv_psi) for ind_u in range(len(theta))]) 
    
    
    # 1.2 Absolute uncertainty
    # U_Vrad_theta.append([Vref*np.cos(np.radians(psi[ind_u]))*np.sin(np.radians(theta[ind_u]))*np.radians(stdv_theta) for ind_u in range(len(theta))])
    # U_Vrad_psi.append([Vref*np.cos(np.radians(theta[ind_u]))*np.sin(np.radians(psi[ind_u]))*np.radians(stdv_psi) for ind_u in range(len(theta))])
    
    
    # 1.3 New approach (Absolute uncertainty):   
    U_Vrad_theta.append([Vref*(np.cos(np.radians(psi[ind_u]))*np.sin(np.radians(theta[ind_u]))*np.cos(np.radians(wind_direction[ind_u]))+np.sin(np.radians(psi[ind_u]))*np.sin(np.radians(theta[ind_u]))*np.sin(np.radians(wind_direction[ind_u]))+np.cos(np.radians(theta[ind_u]))*np.tan(np.radians(wind_pitch[ind_u])))*np.radians(stdv_theta) for ind_u in range(len(theta))])
    U_Vrad_psi.append([Vref*(np.sin(np.radians(psi[ind_u]))*np.cos(np.radians(theta[ind_u]))*np.cos(np.radians(wind_direction[ind_u]))-np.cos(np.radians(psi[ind_u]))*np.cos(np.radians(theta[ind_u]))*np.sin(np.radians(wind_direction[ind_u])))*np.radians(stdv_psi) for ind_u in range(len(theta))])        
        
             
    # 1.4 Expanded uncertainty
    U_Vrad_homo_GUM.append([np.sqrt((U_Vrad_theta[0][ind_u])**2+(U_Vrad_psi[0][ind_u])**2) for ind_u in range(len(theta))])
    
    
    # 2. Calculate radial speed uncertainty for an heterogeneous flow:
    U_Vrad_sh_theta,U_Vrad_sh_psi,U_Vh_sh,U_Vrad_S_GUM,U_Vrad_sh_range= [],[],[],[],[]       
    
    # U_Vrad_sh_theta2,U_Vrad_S_GUM2=[],[]
    for ind_alpha in range(len(alpha)):
        
        # 2.1 Relative Uncertainty in %:
        # U_Vrad_sh_theta.append([np.sqrt((100*np.radians(stdv_theta)*((alpha[ind_alpha]*(rho[ind_u]*np.cos(np.radians(theta[ind_u]))/(z_Lidar+rho[ind_u]*np.sin(np.radians(theta[ind_u])))))-np.tan(np.radians(theta[ind_u])) ))**2) for ind_u in range(len(theta))])
        # U_Vrad_sh_psi.append([np.sqrt((100*np.tan(np.radians(psi[ind_u]))*np.radians(stdv_psi))**2) for ind_u in range(len(psi))])            
        # U_Vrad_sh_range.append([np.sqrt((100*np.sin(np.radians(theta[ind_u]))*alpha[ind_alpha]/(rho[ind_u]*np.sin(np.radians(theta[ind_u]))+z_Lidar)*stdv_rho)**2) for ind_u in range(len(rho))])
                 
        
       # 2.2 Absolute uncertainty
       #U_Vrad_sh_theta.append([Vref*(((np.sin(np.radians(theta_noisy[ind_u]))*rho_noisy[ind_u])/(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))**alpha[ind_alpha])*np.cos(np.radians(psi[ind_u]))*np.cos(np.radians(theta[ind_u]))*np.radians(stdv_theta*theta[ind_u])*abs((alpha[ind_alpha]/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
        # U_Vrad_sh_theta.append([Vref*(((Href+(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))/Href)**alpha[ind_alpha])*np.cos(np.radians(psi[ind_u]))*np.cos(np.radians(theta[ind_u]))*np.radians(stdv_theta)*((alpha[ind_alpha]*(rho[ind_u]*np.cos(np.radians(theta[ind_u]))/(Href+rho[ind_u]*np.sin(np.radians(theta[ind_u])))))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
        # U_Vrad_sh_psi.append([Vref*(((Href+np.sin(np.radians(theta[ind_u]))*rho[ind_u])/(Href))**alpha[ind_alpha])*np.cos(np.radians(theta[ind_u]))*np.sin(np.radians(psi[ind_u]))*np.radians(stdv_psi) for ind_u in range(len(psi))])            
        # U_Vrad_sh_range.append([Vref*(((Href+np.sin(np.radians(theta[ind_u]))*rho[ind_u])/(Href))**alpha[ind_alpha])*alpha[ind_alpha]*np.sin(np.radians(theta[ind_u]))/(Href+(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))*np.cos(np.radians(theta[ind_u]))*np.cos(np.radians(psi[ind_u]))*(stdv_rho) for ind_u in range(len(rho))])

        
        # 2.3 New approach (Absolute uncertainty):
        U_Vrad_sh_theta.append([Vref*np.cos(np.radians(theta[ind_u]))*(((Href+(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))/Href)**alpha[ind_alpha])*((alpha[ind_alpha]*(np.tan(np.radians(theta[ind_u]))*np.tan(np.radians(wind_pitch[ind_u]))-np.cos(np.radians(psi[ind_u]-wind_direction[ind_u])))*(rho[ind_u]*((np.cos(np.radians(theta[ind_u]))))/(Href+(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))))+((np.cos(np.radians(psi[ind_u]-wind_direction[ind_u]))*np.tan(np.radians(theta[ind_u])))+np.tan(np.radians(wind_pitch[ind_u]))))*np.radians(stdv_theta)  for ind_u in range(len(theta))])
        U_Vrad_sh_psi.append([  Vref*np.cos(np.radians(theta[ind_u]))*(((Href+(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))/Href)**alpha[ind_alpha])*np.sin(np.radians(psi[ind_u]-wind_direction[ind_u]))*np.radians(stdv_psi)  for ind_u in range(len(theta))])
        U_Vrad_sh_range.append([Vref*np.cos(np.radians(theta[ind_u]))*(((Href+(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))/Href)**alpha[ind_alpha])*alpha[ind_alpha]*(np.sin(np.radians(theta[ind_u]))/(Href+(np.sin(np.radians(theta[ind_u]))*rho[ind_u])))*(-np.cos(np.radians(psi[ind_u]-wind_direction[ind_u]))+np.tan(np.radians(theta[ind_u]))*np.tan(np.radians(wind_pitch[ind_u])))*(stdv_rho)      for ind_u in range(len(theta))])
    
    
        # 2.4 Expanded uncertainty with contributions of theta, psi and rho
        U_Vrad_S_GUM.append([np.sqrt(((U_Vrad_sh_theta[ind_alpha][ind_u]))**2+((U_Vrad_sh_psi[ind_alpha][ind_u]))**2+((U_Vrad_sh_range[ind_alpha][ind_u]))**2) for ind_u in range(len(rho)) ])
        # U_Vrad_S_GUM.append([np.sqrt((np.mean(U_Vrad_sh_theta[ind_alpha][ind_u]))**2+(np.mean(U_Vrad_sh_psi[ind_alpha][ind_u]))**2+(np.mean(U_Vrad_sh_range[ind_alpha][ind_u]))**2) for ind_u in range(len(rho)) ])    


    
    # for param1_or,param2_or,param3_or in zip(param1,param2,param3):# Take coordinates from inputs
    #     Mean_DISTANCE=[]
    #     DISTANCE=[]        
    #     stdv_DISTANCE=[]  
        
    #     # Coordinates of the original points:
    #     # Calculating the theoretical point coordinate transformation + lidar origin + sample rate  (conversion from spherical to cartesians):
    #     x0 = (param1_or)*np.cos(param3_or)*np.sin(param2_or) + Lidar.optics.scanner.origin[0]
    #     y0 = (param1_or)*np.sin(param3_or)*np.sin(param2_or) + Lidar.optics.scanner.origin[1]
    #     z0 = (param1_or)*np.cos(param2_or) + Lidar.optics.scanner.origin[2] + sample_rate_count
    #     # Storing coordinates
    #     X0.append(x0)
    #     Y0.append(y0)
    #     Z0.append(z0)
    #     # pdb.set_trace()
    #     for trial in range(0,10):
            
    #         # Create white noise with stdv selected by user:
    #         n=10000 # Number of cases to combine           
    #         # Position, due to pointing accuracy
    #         del_param1 = np.array(np.random.normal(0,stdv_param1,n)) # why a normal distribution??Does it have sense, can be completely random? --> Because of the central limit theorem!
    #         del_param2 = np.array(np.random.normal(0,stdv_param2,n))
    #         del_param3 = np.array(np.random.normal(0,stdv_param3,n))                        
            
    #         # Adding noise to the theoretical position:
    #         noisy_param1 = param1_or + del_param1
    #         noisy_param2 = param2_or + del_param2 
    #         noisy_param3 = param3_or + del_param3 
    #         # Coordinates of the noisy points:            
    #         x = noisy_param1*np.cos(noisy_param3)*np.sin(noisy_param2)
    #         y = noisy_param1*np.sin(noisy_param3)*np.sin(noisy_param2) 
    #         z = noisy_param1*np.cos(noisy_param2) + sample_rate_count
            
    #         # Apply error in inclinometers   
    #         # Rotation, due to inclinometers
    #         del_yaw     = np.random.normal(0,stdv_yaw,n)
    #         del_pitch   = np.random.normal(0,stdv_pitch,n)
    #         del_roll    = np.random.normal(0,stdv_roll,n)
            
    #         # Adding noise to the inclinometer stdv
    #         noisy_yaw   = stdv_yaw + del_yaw
    #         noisy_pitch = stdv_pitch + del_pitch
    #         noisy_roll  = stdv_roll + del_roll
            
    #         R = SA.sum_mat(noisy_yaw,noisy_pitch,noisy_roll)
                           
    #         xfinal_noisy = np.matmul(R,[x,y,z])[0] + Lidar.optics.scanner.origin[0] # Rotation
    #         yfinal = np.matmul(R,[x,y,z])[1] + Lidar.optics.scanner.origin[1]
    #         xfinal_noisy = np.matmul(R,[x,y,z])[2] + Lidar.optics.scanner.origin[2]

    #         # Distance between theoretical measured points and noisy points:
    #         DISTANCE.append(np.sqrt((xfinal_noisy-x0)**2+(yfinal-y0)**2+(xfinal_noisy-z0)**2))
    #         Mean_DISTANCE.append(np.mean(DISTANCE[trial]))    
    #         stdv_DISTANCE.append(np.std(DISTANCE[trial]))

    #     sample_rate_count+=Lidar.optics.scanner.sample_rate    
    #     SimMean_DISTANCE.append(np.mean(Mean_DISTANCE))        # Mean error distance of each point in the pattern  
    #     StdvMean_DISTANCE.append(np.mean(stdv_DISTANCE)) # Mean error distance stdv for each point in the pattern
        
    #     # Storing coordinates:
    #     X.append(xfinal_noisy)
    #     Y.append(yfinal)
    #     Z.append(xfinal_noisy)
    #     NoisyX.append(X[coun][0])
    #     NoisyY.append(Y[coun][0])
    #     NoisyZ.append(Z[coun][0])
    #     coun+=1
    # Noisy_Coord=[NoisyX,NoisyY,NoisyZ]
    # Coord=[X0,Y0,Z0]
    # # pdb.set_trace()
    # SA.cart2sph(Coord[0],Coord[1],Coord[2])
    
    #Call probe volume uncertainty function
    # Probe_param = Lidar.probe_volume.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,param1)

    # Saving coordenates to a file in desktop
    # file=open('C:/Users/fcosta/Desktop/data_'+Qlunc_yaml_inputs['Components']['Scanner']['Type']+'.txt','w')
    # XX=repr(param1)
    # YY=repr(np.degrees(param2))
    # ZZ=repr(np.degrees(param3))
    # XX_noisy=repr(NoisyX)
    # Y_noisy=repr(NoisyY)
    # ZZ_noisy=repr(NoisyZ)    

    # file.write('\n'+Qlunc_yaml_inputs['Components']['Scanner']['Type'] +'\nParam1:'+XX+"\n"+'\nParam2:'+YY+"\n"+'\nParam3:'+ZZ+"\n")
    # file.close()   
    
    # Final_Output_UQ_Scanner                 = {'Simu_Mean_Distance_Error':SimMean_DISTANCE,'STDV_Distance':StdvMean_DISTANCE,'MeasPoint_Coordinates':Coord,'NoisyMeasPoint_Coordinates':Noisy_Coord,'Rayleigh length':Probe_param['Rayleigh Length'],'Rayleigh length uncertainty':Probe_param['Rayleigh Length uncertainty']}
    # Lidar.lidar_inputs.dataframe['Scanner'] = ([np.mean(Final_Output_UQ_Scanner['Simu_Mean_Distance_Error'])])*len(Atmospheric_Scenario.temperature)  
    
    
    # Storing data
    Final_Output_UQ_Scanner                 = {'Vr Uncertainty homo MC [m/s]':U_Vrad_homo_MC,'Vr Uncertainty homo GUM [m/s]':U_Vrad_homo_GUM,'Vr Uncertainty MC [m/s]':U_Vrad_S_MC,'Vr Uncertainty GUM [m/s]':U_Vrad_S_GUM,'x':x,'y':y,'z':z,'rho':rho,'theta':theta,'psi':psi} #, 'Rayleigh length':Probe_param['Rayleigh Length'],'Rayleigh length uncertainty':Probe_param['Rayleigh Length uncertainty']}
    Lidar.lidar_inputs.dataframe['Scanner'] = (Final_Output_UQ_Scanner['Vr Uncertainty MC [m/s]'])*len(Atmospheric_Scenario.temperature)  
    
    # Plotting
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_Scanner,Qlunc_yaml_inputs['Flags']['Scanning Pattern'],False,False,False)
    
    
    # Scatter plot
    # pdb.set_trace()
    rho_scat,theta_scat,psi_scat,box=SA.mesh(rho,theta,psi)
    QPlot.scatter3d(theta_scat,psi_scat,rho_scat,U_Vrad0_S_MC[0])
    
    pdb.set_trace()
    return Final_Output_UQ_Scanner,Lidar.lidar_inputs.dataframe

#%% Optical circulator:

def UQ_OpticalCirculator(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs):    
    """.
    
    Optical circulator uncertainty estimation. Location: ./UQ_Functions/UQ_Optics_Classes.py
    Parameters.  
  
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
    list
    
    """   
    # To take into account insertion losses (with correlated uncertainties)
    #Optical_Circulator_losses = [np.array(Lidar.optics.optical_circulator.insertion_loss)]
    #Pratio=10**(-Lidar.optics.optical_circulator.insertion_loss/10)# P_in/P_out
    
    #  if the insertion loss is expressed in % (X% losses):    
    #Optical_Circulator_losses = 10*np.log10(1-(X/100)) # output in dB
    
    # If we assume an SNR:

    # Optical_Circulator_Uncertainty = [np.array(Lidar.optics.optical_circulator.insertion_loss)]
    Optical_Circulator_Uncertainty_w = [Qlunc_yaml_inputs['Components']['Laser']['Output power']/(10**(Lidar.optics.optical_circulator.SNR/10))]
    Optical_Circulator_Uncertainty_dB = 10*np.log10(Optical_Circulator_Uncertainty_w)
    
    # Add to the dictionary
    Final_Output_UQ_Optical_Circulator={'Optical_Circulator_Uncertainty':Optical_Circulator_Uncertainty_dB}
    Lidar.lidar_inputs.dataframe['Optical circulator']=Final_Output_UQ_Optical_Circulator['Optical_Circulator_Uncertainty']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))  # linspace to create the appropiate length for the xarray. 
    
    return Final_Output_UQ_Optical_Circulator,Lidar.lidar_inputs.dataframe

#%% TELESCOPE NOT IMPLEMENTED
def UQ_Telescope(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
     # UQ_telescope=[(temp*0.5+hum*0.1+curvature_lens*0.1+aberration+o_c_tele) \
     #               for temp           in inputs.atm_inp.Atmospheric_inputs['temperature']\
     #               for hum            in inputs.atm_inp.Atmospheric_inputs['humidity']\
     #               for curvature_lens in inputs.optics_inp.Telescope_uncertainty_inputs['curvature_lens'] \
     #               for aberration     in inputs.optics_inp.Telescope_uncertainty_inputs['aberration'] \
     #               for o_c_tele       in inputs.optics_inp.Telescope_uncertainty_inputs['OtherChanges_tele']]
     # Telescope_Losses =Lidar.optics.telescope.Mirror_losses
     # pdb.set_trace()
     UQ_telescope=[-100]
     Final_Output_UQ_Telescope={'Telescope_Uncertainty':UQ_telescope}
     Lidar.lidar_inputs.dataframe['Telescope']=Final_Output_UQ_Telescope['Telescope_Uncertainty']*np.linspace(1,1,len(Atmospheric_Scenario.temperature)) # linspace to create the appropriate length for the xarray. 
     return Final_Output_UQ_Telescope,Lidar.lidar_inputs.dataframe

#%% Sum of uncertainties in `optics` module: 
def sum_unc_optics(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    List_Unc_optics = []
    
    # Scanner
    if Lidar.optics.scanner != None:
        try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations            
            if Lidar.wfr_model.reconstruction_model != 'None':
                # pdb.set_trace()    
                Scanner_Uncertainty,DataFrame=Lidar.optics.scanner.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
                WFR_Uncertainty=None#Lidar.wfr_model.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,Scanner_Uncertainty)            
            else:            
                Scanner_Uncertainty,DataFrame=Lidar.optics.scanner.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
                WFR_Uncertainty = None
        except:
            Scanner_Uncertainty=None
            print(colored('Error in scanner uncertainty calculations!','cyan', attrs=['bold']))
    else:
        print (colored('You didn´t include a head scanner in the lidar.','cyan', attrs=['bold']))       
    
    # Telescope
    if Lidar.optics.telescope != 'None':
        try:
            Telescope_Uncertainty,DataFrame=Lidar.optics.telescope.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
            List_Unc_optics.append(Telescope_Uncertainty['Telescope_Uncertainty'])                   
        except:
            Telescope_Uncertainty=None
            print(colored('Error in telescope uncertainty calculations!','cyan', attrs=['bold']))
    else:
        print (colored('You didn´t include a telescope in the lidar,so that telescope uncertainty contribution is not in lidar uncertainty estimations.','cyan', attrs=['bold']))
    
    # Optical Circulator
    if Lidar.optics.optical_circulator != 'None': 
        try:
            Optical_circulator_Uncertainty,DataFrame = Lidar.optics.optical_circulator.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
            List_Unc_optics.append(Optical_circulator_Uncertainty['Optical_Circulator_Uncertainty'])       
        except:
            Optical_circulator_Uncertainty = None
            print(colored('Error in optical circulator uncertainty calculations!','cyan', attrs=['bold']))    
    else:
        print(colored('You didn´t include an optical circulator in the lidar,so that optical circulator uncertainty contribution is not in lidar uncertainty estimations.','cyan', attrs=['bold']))
    Uncertainty_Optics_Module=SA.unc_comb(List_Unc_optics)
    Final_Output_UQ_Optics = {'Uncertainty_Optics':Uncertainty_Optics_Module, 'Uncertainty_WFR':WFR_Uncertainty['WFR_Uncertainty'],'Mean_error_PointingAccuracy':Scanner_Uncertainty['Simu_Mean_Distance_Error'],'Stdv_PointingAccuracy':Scanner_Uncertainty['STDV_Distance'], 'Rayleigh length':Scanner_Uncertainty['Rayleigh length'],'Rayleigh length uncertainty':Scanner_Uncertainty['Rayleigh length uncertainty']}
    pdb.set_trace()
    Lidar.lidar_inputs.dataframe['Optics Module']=Final_Output_UQ_Optics['Uncertainty_Optics']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))  # linspace to create the appropiate length for the xarray. 
    return Final_Output_UQ_Optics,Lidar.lidar_inputs.dataframe

