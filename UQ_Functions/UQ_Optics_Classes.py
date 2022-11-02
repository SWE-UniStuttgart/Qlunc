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
    rho_noisy0,theta_noisy0,psi_noisy0,rho_noisy,theta_noisy1,theta_noisy2,psi_noisy,rho_noisy1,rho_noisy2, theta_noisy,psi_noisy1,psi_noisy2,wind_direction_noisy ,wind_tilt_noisy   = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
    Coordfinal_noisy,Coordfinal=[],[]
    coun=0
    sample_rate_count=0
    Href  = Qlunc_yaml_inputs['Components']['Scanner']['Href'],
    Vref  = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']
    alpha = Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp']    
    Hg    = Qlunc_yaml_inputs['Atmospheric_inputs']['Height_ground'] 
    Hl    = Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0][2]
    # #Call probe volume uncertainty function. 
    # Probe_param = Lidar.probe_volume.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)


    # R: Implement error in deployment of the tripod as a rotation over yaw, pitch and roll
    stdv_yaw    = np.array(np.radians(Lidar.lidar_inputs.yaw_error_dep))
    stdv_pitch  = np.array(np.radians(Lidar.lidar_inputs.pitch_error_dep))
    stdv_roll   = np.array(np.radians(Lidar.lidar_inputs.roll_error_dep))
    
   
    # Rho, theta and psi values of the measuring point
    
    rho0           = Lidar.optics.scanner.focus_dist  
    theta0         = np.radians(Lidar.optics.scanner.cone_angle)
    psi0           = np.radians(Lidar.optics.scanner.azimuth)
    # wind_direction = np.radians(np.array([Atmospheric_Scenario.wind_direction]*len(psi0)))
    wind_direction = np.radians(np.linspace(0,359,360))
    wind_tilt      = np.radians( np.array([Atmospheric_Scenario.wind_tilt]*len(psi0)))

    # pdb.set_trace()
   
    # MEasurement point in cartesian coordinates before applying lidar position
    x,y,z=SA.sph2cart(rho0,theta0,psi0)
    
    
     # Lidars' position:
    class lidar:
        def __init__(self, x,y,z,x_Lidar,y_Lidar,z_Lidar):
            self.x_Lidar=x_Lidar
            self.y_Lidar=y_Lidar
            self.z_Lidar=z_Lidar
            self.x=x
            self.y=y
            self.z=z
        @classmethod
        def vector_pos(cls,x,y,z,x_Lidar,y_Lidar,z_Lidar):
            fx=(x-x_Lidar)
            fy=(y-y_Lidar)
            fz=(z-z_Lidar)
            return(cls,fx,fy,fz)
        @classmethod
        def Cart2Sph (cls, x_vector_pos,y_vector_pos,z_vector_pos):
            rho1,theta1,psi1 =SA.cart2sph(x_vector_pos,y_vector_pos,z_vector_pos)
            return (cls,rho1,theta1,psi1)
        
    
    x_Lidar1,y_Lidar1,z_Lidar1,lidars=[],[],[],[]
    lidars={}
    for ind_origin in range(np.shape(Qlunc_yaml_inputs['Components']['Scanner']['Origin'])[0]):
        
        lidars['Lidar'+str(ind_origin+1)+'_Rectangular']={'x':(lidar.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][2])[1]),
                                                        'y':(lidar.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][2])[2]),
                                                        'z':(lidar.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][2])[3])}
        lidars['Lidar'+str(ind_origin+1)+'_Spherical']={'rho':np.round((lidar.Cart2Sph(lidars['Lidar'+str(ind_origin+1)+'_Rectangular']['x'],lidars['Lidar'+str(ind_origin+1)+'_Rectangular']['y'],lidars['Lidar'+str(ind_origin+1)+'_Rectangular']['z']))[1],4),
                                                      'theta':np.round((lidar.Cart2Sph(lidars['Lidar'+str(ind_origin+1)+'_Rectangular']['x'],lidars['Lidar'+str(ind_origin+1)+'_Rectangular']['y'],lidars['Lidar'+str(ind_origin+1)+'_Rectangular']['z']))[2],4),
                                                      'psi':np.round((lidar.Cart2Sph(lidars['Lidar'+str(ind_origin+1)+'_Rectangular']['x'],lidars['Lidar'+str(ind_origin+1)+'_Rectangular']['y'],lidars['Lidar'+str(ind_origin+1)+'_Rectangular']['z']))[3],4)}
        
        
        x_Lidar1.append(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][0])
        y_Lidar1.append(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][1])
        z_Lidar1.append(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_origin][2])
    
    # pdb.set_trace()
        # stdv focus distance, cone angle and azimuth:
    # stdv_param1 = Probe_param['Focus Distance uncertainty']
    # stdv_rho   = Lidar.optics.scanner.stdv_focus_dist    
    # stdv_theta = np.radians(Lidar.optics.scanner.stdv_cone_angle)
    # stdv_psi   = np.radians(Lidar.optics.scanner.stdv_azimuth)
    # stdv_wind_direction =np.radians(0)
    # stdv_wind_tilt = np.radians(0)
    ##################################################################################################################################################
    ##################################################################################################################################################
    theta1,U_theta1 = lidars['Lidar1_Spherical']['theta'],np.radians(Lidar.optics.scanner.stdv_cone_angle[0])
    theta2,U_theta2 = lidars['Lidar2_Spherical']['theta'],np.radians(Lidar.optics.scanner.stdv_cone_angle[1])
    psi1  ,U_psi1   = lidars['Lidar1_Spherical']['psi'],np.radians(Lidar.optics.scanner.stdv_azimuth[0])
    psi2  ,U_psi2   = lidars['Lidar2_Spherical']['psi'],np.radians(Lidar.optics.scanner.stdv_azimuth[1])
    rho1  ,U_rho1   = lidars['Lidar1_Spherical']['rho'],Lidar.optics.scanner.stdv_focus_dist [0]
    rho2  ,U_rho2   = lidars['Lidar2_Spherical']['rho'],Lidar.optics.scanner.stdv_focus_dist[1] 
    # wind_direction = np.radians(np.array([180]))*len(psi1)
    

    #%% 2) State the correlations to calculate the correlation in the different VLOS:
    # VLOS1_VLOS2_corr_n = 1 # correlation betweem Vlos1 and Vlos2
    # pdb.set_trace()
    psi1_psi2_corr_n     = Lidar.optics.scanner.correlations[0]  # correlation between psi1 and psi2
    theta1_theta2_corr_n = Lidar.optics.scanner.correlations[1] # correlation Theta1 and Theta2
    rho1_rho2_corr_n     = Lidar.optics.scanner.correlations[2]  # correlation Rho1 and Rho2
    
    # Cross correlations:
    psi1_theta1_corr_n  = Lidar.optics.scanner.correlations[3]
    psi1_theta2_corr_n  = Lidar.optics.scanner.correlations[4]
    psi2_theta1_corr_n  = Lidar.optics.scanner.correlations[5]
    psi2_theta2_corr_n  = Lidar.optics.scanner.correlations[6]
        
    # There are no correlation between range and angles since the range is determined by the AOM (at least in pulsed lidars) and the angles accuracy is related to the alignment of the telescope mirrors, to the position of the lense and also to the servos orienting the scanner
    psi1_rho1_corr_n    = 0
    psi1_rho2_corr_n    = 0
    psi2_rho1_corr_n    = 0
    psi2_rho2_corr_n    = 0
    
    theta1_rho1_corr_n  = 0 
    theta1_rho2_corr_n  = 0
    theta2_rho1_corr_n  = 0
    theta2_rho2_corr_n  = 0
    
    # VLOS1_psi1_corr_n   = 0
    # VLOS1_psi2_corr_n   = 0 
    # VLOS2_psi1_corr_n   = 0
    # VLOS2_psi2_corr_n   = 0 
    
    # VLOS1_theta1_corr_n = 0  
    # VLOS1_theta2_corr_n = 0 
    # VLOS2_theta1_corr_n = 0   
    # VLOS2_theta2_corr_n = 0 
    
    # VLOS1_rho1_corr_n   = 0  
    # VLOS1_rho2_corr_n   = 0   
    # VLOS2_rho1_corr_n   = 0  
    # VLOS2_rho2_corr_n   = 0 
    
    Corr_vec =[psi1_psi2_corr_n,
               theta1_theta2_corr_n ,
               rho1_rho2_corr_n,
               psi1_theta1_corr_n,
               psi1_theta2_corr_n,
               psi2_theta1_corr_n,  
               psi2_theta2_corr_n,
               psi1_rho1_corr_n,
               psi1_rho2_corr_n,
               psi2_rho1_corr_n,
               psi2_rho2_corr_n,
               theta1_rho1_corr_n, 
               theta1_rho2_corr_n,
               theta2_rho1_corr_n,
               theta2_rho2_corr_n]
    
    # Uncertainty of VLOS with wind direction
    
    # U_VLOS1_W,U_VLOS2_W,CORRCOEF_W = dVLOS_dw(theta1,theta2,psi1,psi2,rho1,rho2,wind_direction,Href,Vref,alpha)
    # pdb.set_trace()


    #%% 3) Create the noisy distributions (assume normal distributions):
    Uncertainty_V,Uncertainty_U,Uncertainty_Vh_MC,Uncertainty_Vh_GUM=[],[],[],[]
    u_wind_GUM, v_wind_GUM=[],[]
    Vwind_MC,Uwind_MC,u_wind_G2,v_wind_G2=[],[],[],[]
    CORR_COEF_uv,CORR_COEF_VLOS=[],[]
    U_VLOS1_GUM,U_VLOS2_GUM,U_VLOS1_MC,U_VLOS2_MC=[],[],[],[]
    VLOS1_list,VLOS2_list=[],[]
    
    for ind_wind_dir in range(len(wind_direction)):  
        # Vlos1_noisy  = np.random.normal(Vlos1[0],u_Vlos1,N_MC)
        # Vlos2_noisy  = np.random.normal(Vlos2[0],u_Vlos2,N_MC) 
        # pdb.set_trace()
        theta1_noisy = np.random.normal(theta1[0],U_theta1,Lidar.optics.scanner.N_MC)
        theta2_noisy = np.random.normal(theta2[0],U_theta2,Lidar.optics.scanner.N_MC)
        psi1_noisy   = np.random.normal(psi1[0],U_psi1,Lidar.optics.scanner.N_MC)
        psi2_noisy   = np.random.normal(psi2[0],U_psi2,Lidar.optics.scanner.N_MC)
        rho1_noisy   = np.random.normal(rho1[0],U_rho1,Lidar.optics.scanner.N_MC)
        rho2_noisy   = np.random.normal(rho2[0],U_rho2,Lidar.optics.scanner.N_MC)
        
        
        CORRCOEF_T=np.corrcoef(theta1_noisy,theta2_noisy)
        CORRCOEF_P=np.corrcoef(psi1_noisy,psi2_noisy)
        CORRCOEF_R=np.corrcoef(rho1_noisy,rho2_noisy)
        
        
        
        #%% 4) Obtain the Correlated distributions:
        
        # VLOS_means = [Vlos1_noisy.mean(), Vlos2_noisy.mean()]  
        # VLOS_stds  = [Vlos1_noisy.std(), Vlos2_noisy.std()]
        
        theta_means = [theta1_noisy.mean(), theta2_noisy.mean()]  
        theta_stds  = [theta1_noisy.std(), theta2_noisy.std()]
        
        psi_means = [psi1_noisy.mean(), psi2_noisy.mean()]  
        psi_stds  = [psi1_noisy.std(), psi2_noisy.std()]
        
        rho_means = [rho1_noisy.mean(), rho2_noisy.mean()]  
        rho_stds  = [rho1_noisy.std(), rho2_noisy.std()]
        
        # Covariance Matrix:
        cov_MAT=[[              theta_stds[0]**2,                     theta_stds[1]*theta_stds[0]*theta1_theta2_corr_n,   psi_stds[0]*theta_stds[0]*psi1_theta1_corr_n,   psi_stds[1]*theta_stds[0]*psi2_theta1_corr_n,   rho_stds[0]*theta_stds[0]*theta1_rho1_corr_n,  rho_stds[1]*theta_stds[0]*theta1_rho2_corr_n],
                  [theta_stds[0]*theta_stds[1]*theta1_theta2_corr_n,                 theta_stds[1]**2,                     psi_stds[0]*theta_stds[1]*psi1_theta2_corr_n,   psi_stds[1]*theta_stds[1]*psi2_theta2_corr_n,   rho_stds[0]*theta_stds[1]*theta2_rho1_corr_n,  rho_stds[1]*theta_stds[1]*theta2_rho2_corr_n],
                  [theta_stds[0]*psi_stds[0]*psi1_theta1_corr_n ,      theta_stds[1]*psi_stds[0]*psi1_theta2_corr_n,                   psi_stds[0]**2,                     psi_stds[1]*psi_stds[0]*psi1_psi2_corr_n,       rho_stds[0]*psi_stds[0]*psi1_rho1_corr_n,      rho_stds[1]*psi_stds[0]*psi1_rho2_corr_n],
                  [theta_stds[0]*psi_stds[1]*psi2_theta1_corr_n,       theta_stds[1]*psi_stds[1]*psi2_theta2_corr_n,       psi_stds[0]*psi_stds[1]*psi1_psi2_corr_n,                   psi_stds[1]**2,                     rho_stds[0]*psi_stds[1]*psi2_rho1_corr_n,      rho_stds[1]*psi_stds[1]*psi2_rho2_corr_n],
                  [theta_stds[0]*rho_stds[0]*theta1_rho1_corr_n,       theta_stds[1]*rho_stds[0]*theta2_rho1_corr_n,       psi_stds[0]*rho_stds[0]*psi1_rho1_corr_n,       psi_stds[1]*rho_stds[0]*psi2_rho1_corr_n,                   rho_stds[0]**2,                    rho_stds[1]*rho_stds[0]*rho1_rho2_corr_n],
                  [theta_stds[0]*rho_stds[1]*theta1_rho2_corr_n,       theta_stds[1]*rho_stds[1]*theta2_rho2_corr_n,       psi_stds[0]*rho_stds[1]*psi1_rho2_corr_n,       psi_stds[1]*rho_stds[1]*psi2_rho2_corr_n,       rho_stds[0]*rho_stds[1]*rho1_rho2_corr_n,                  rho_stds[1]**2]]
        # cov_MAT=[[              theta_stds[0]**2,                     theta_stds[1]*theta_stds[0]*theta1_theta2_corr_n,   psi_stds[0]*theta_stds[0]*psi1_theta1_corr_n,   psi_stds[1]*theta_stds[0]*psi2_theta1_corr_n  ],
        #           [theta_stds[0]*theta_stds[1]*theta1_theta2_corr_n,                 theta_stds[1]**2,                     psi_stds[0]*theta_stds[1]*psi1_theta2_corr_n,   psi_stds[1]*theta_stds[1]*psi2_theta2_corr_n],
        #           [theta_stds[0]*psi_stds[0]*psi1_theta1_corr_n ,      theta_stds[1]*psi_stds[0]*psi1_theta2_corr_n,                   psi_stds[0]**2,                     psi_stds[1]*psi_stds[0]*psi1_psi2_corr_n],
        #           [theta_stds[0]*psi_stds[1]*psi2_theta1_corr_n,       theta_stds[1]*psi_stds[1]*psi2_theta2_corr_n,       psi_stds[0]*psi_stds[1]*psi1_psi2_corr_n,                   psi_stds[1]**2]]
          
        # cov_MAT = np.cov([theta1_noisy,theta2_noisy,psi1_noisy,psi2_noisy,rho1_noisy,rho2_noisy])
        
        # # Multivariate distributions:
        Theta1_cr,Theta2_cr,Psi1_cr,Psi2_cr,Rho1_cr,Rho2_cr = multivariate_normal.rvs([theta_means[0],theta_means[1],psi_means[0],psi_means[1],rho_means[0],rho_means[1]], cov_MAT,Lidar.optics.scanner.N_MC).T
        
       
        # Theta
        #Covariance (theta1, theta2) as defined in GUM
        theta_covariance = 1/(Lidar.optics.scanner.N_MC-1)*sum((Theta1_cr-theta_means[0])*(Theta2_cr-theta_means[1]))
        # Correlation coefficients Theta 
        C_theta = theta_covariance/(theta_stds[0]*theta_stds[1])
        Corr_coef_theta=np.corrcoef(Theta1_cr,Theta2_cr)
        
        
        # Psi
        #Covariance(psi1, psi2) as defined in GUM
        psi_covariance = 1/(Lidar.optics.scanner.N_MC-1)*sum((Psi1_cr-psi_means[0])*(Psi2_cr-psi_means[1]))
        # Correlation coefficients PSi
        C_psi=psi_covariance/(psi_stds[0]*psi_stds[1])
        Corr_coef_psi=np.corrcoef(Psi1_cr,Psi2_cr)
        
        
        # Rho
        #Covariance(psi1, psi2) as defined in GUM
        rho_covariance = 1/(Lidar.optics.scanner.N_MC-1)*sum((Rho1_cr-rho_means[0])*(Rho2_cr-rho_means[1]))
        # Correlation coefficients PSi
        C_rho=rho_covariance/(rho_stds[0]*rho_stds[1])
        Corr_coef_rho=np.corrcoef(Rho1_cr, Rho2_cr)
        
        
        # Cross correlations
        Corr_coef_theta1_psi1 = np.corrcoef(Theta1_cr,Psi1_cr)
        Corr_coef_theta1_rho1 = np.corrcoef(Theta1_cr,Rho1_cr)
        Corr_coef_rho1_psi1   = np.corrcoef(Rho1_cr,Psi1_cr)
        
        Corr_coef_theta2_psi2 = np.corrcoef(Theta2_cr,Psi2_cr)
        Corr_coef_theta2_rho2 = np.corrcoef(Theta2_cr,Rho2_cr)
        Corr_coef_rho2_psi2   = np.corrcoef(Rho2_cr,Psi2_cr)
        
        Corr_coef_theta1_psi2 = np.corrcoef(Theta1_cr,Psi2_cr)
        Corr_coef_theta1_rho2 = np.corrcoef(Theta1_cr,Rho2_cr)
        Corr_coef_rho1_psi2   = np.corrcoef(Rho1_cr,Psi2_cr)
        
        Corr_coef_theta2_psi1 = np.corrcoef(Theta2_cr,Psi1_cr)
        Corr_coef_theta2_rho1 = np.corrcoef(Theta2_cr,Rho1_cr)
        Corr_coef_rho2_psi1   = np.corrcoef(Rho2_cr,Psi1_cr)
        
        # Cross correlations
        # CROS_CORR = [Corr_coef_theta1_psi1[0][1],Corr_coef_theta1_rho1[0][1],Corr_coef_rho1_psi1[0][1],Corr_coef_theta2_psi2[0][1],
        #               Corr_coef_theta2_rho2[0][1],Corr_coef_rho2_psi2[0][1],Corr_coef_psi[0][1],Corr_coef_theta[0][1],Corr_coef_rho[0][1]]
        CROS_CORR = [psi1_theta1_corr_n,theta1_rho1_corr_n,psi1_rho1_corr_n,psi2_theta2_corr_n,theta2_rho2_corr_n,
                      psi2_rho2_corr_n,  psi1_psi2_corr_n,theta1_theta2_corr_n,rho1_rho2_corr_n]
        
        
        #%% 5) VLOS uncertainty
        VLOS01,VLOS02,U_VLOS01,U_VLOS02,CORR_COEF,VLOS1_list,VLOS2_list = SA.U_VLOS_MC([Theta1_cr,Theta2_cr],[Psi1_cr,Psi2_cr],[Rho1_cr,Rho2_cr],theta1_noisy,Hl,Href,alpha,wind_direction,Vref,ind_wind_dir,VLOS1_list,VLOS2_list)
        CORR_COEF_VLOS.append(CORR_COEF)
        U_VLOS1_MC.append(U_VLOS01)
        U_VLOS2_MC.append(U_VLOS02)
        
        # Function calculating the uncertainties in VLOS:
        U_VLOS = SA.U_VLOS_GUM (theta1,theta2,psi1,psi2,rho1,rho2,U_theta1,U_theta2,U_psi1,U_psi2,U_rho1,U_rho2,U_VLOS01,U_VLOS02,Hl,Vref,Href,alpha,wind_direction,ind_wind_dir,CROS_CORR,CORR_COEF)
        U_VLOS1_GUM.append(U_VLOS[0][0])
        U_VLOS2_GUM.append(U_VLOS[1][0])
        # pdb.set_trace()
        
        
        
        
        
        
    
        #%% 6) VH Uncertainty
        # Calculate the u and v wind components
        
        # Break down large equations
        A0,B0,C0,D0,E0,F0,G,H0 = SA.U_Vh_MC([Theta1_cr,Theta2_cr],[Psi1_cr,Psi2_cr],[Rho1_cr,Rho2_cr],wind_direction,ind_wind_dir,Href,Vref,alpha,Hl)   
        u_wind = (H0[0]/G)
        Uwind_MC.append(np.mean(u_wind))
        # U_u_wind=(np.std(u_wind))
        # Uncertainty as standard deviation (k=1) in the u wind velocity component estimation
        Uncertainty_U.append(np.std(u_wind))
        
        # V component estimation
        v_wind = (H0[1]/G)
        Vwind_MC.append(np.mean(v_wind))
        
        
        # Uncertainty as standard deviation (k=1) in the v wind velocity component estimation
        Uncertainty_V.append(np.std(v_wind))
        # pdb.set_trace()
    #%% VH Montecarlo uncertainy
        # Horizontal velocity estimation
        Vh_MC=np.sqrt((u_wind**2)+(v_wind**2))
        # Vh_MC=numerator
        # Uncertainty as standard deviation (k=1) in the horizontal velocity estimation
        Uncertainty_Vh_MC.append(np.std(Vh_MC))
    
    #%% VH GUM uncertainty
        # pdb.set_trace()
        
        # Calculate coefficients for the GUM approach
        F,dert11,dert12,dert13,dert21, dert22, dert23,derp11,derp12,derp21,derp22,derr11,derr12,derr21,derr22, numerator,dernumerator = SA.U_Vh_GUM([theta1,theta2],[psi1[0],psi2[0]],[rho1,rho2],wind_direction,ind_wind_dir,Href,Vref,alpha,Hl)   
        
        # With the coefficients we calculate the partial derivatives: 
        dVh_dTheta1 = (dernumerator*(dert11+dert12+dert13)*F +numerator*np.cos(theta2)*np.sin(theta1)*(np.sin(psi1[0]-psi2[0])))/F**2
        dVh_dTheta2 = (dernumerator*(dert21+dert22+dert23)*F +numerator*np.cos(theta1)*np.sin(theta2)*(np.sin(psi1[0]-psi2[0])))/F**2
        
        dVh_dPsi1   = (dernumerator*(derp11+derp12)*F-numerator*np.cos(theta2)*np.cos(theta1)*np.cos(psi1[0]-psi2[0]))/F**2
        dVh_dPsi2   = (dernumerator*(derp21+derp22)*F+numerator*np.cos(theta2)*np.cos(theta1)*np.cos(psi1[0]-psi2[0]))/F**2
        
        dVh_dRho1   = (dernumerator*(derr11+derr12))/F
        dVh_dRho2   = (dernumerator*(derr21+derr22))/F
        
        # Correlation terms:
            
        R = (dVh_dTheta1*dVh_dTheta2*U_theta1*U_theta2*theta1_theta2_corr_n+
             dVh_dPsi1*dVh_dPsi2*U_psi1*U_psi2*psi1_psi2_corr_n+
             dVh_dRho1*dVh_dRho2*U_rho1*U_rho2*rho1_rho2_corr_n+
             
                # dVh_dTheta1*dVh_dPsi1*U_theta1*U_psi1*psi1_theta1_corr_n+
                # dVh_dTheta2*dVh_dPsi1*U_theta2*U_psi1*psi1_theta2_corr_n+
                # dVh_dTheta1*dVh_dPsi2*U_theta1*U_psi2*psi2_theta1_corr_n+
                # dVh_dTheta2*dVh_dPsi2*U_theta2*U_psi2*psi2_theta2_corr_n+
             
                # dVh_dTheta1*dVh_dRho1*U_theta1*U_rho1*theta1_rho1_corr_n+
                # dVh_dTheta2*dVh_dRho1*U_theta2*U_rho1*theta2_rho1_corr_n+
                # dVh_dTheta1*dVh_dRho2*U_theta1*U_rho2*theta1_rho2_corr_n+
                # dVh_dTheta2*dVh_dRho2*U_theta2*U_rho2*theta2_rho2_corr_n+
             
                # dVh_dPsi1*dVh_dRho1*U_psi1*U_rho1*psi1_rho1_corr_n+
                # dVh_dPsi2*dVh_dRho1*U_psi2*U_rho1*psi2_rho1_corr_n+
                # dVh_dPsi1*dVh_dRho2*U_psi1*U_rho2*psi1_rho2_corr_n+
                # dVh_dPsi2*dVh_dRho2*U_psi2*U_rho2*psi2_rho2_corr_n
                # U_theta1*U_theta2*CORRCOEF_T[0][1]+U_psi1*U_psi2*CORRCOEF_P[0][1]+U_rho1*U_rho2*CORRCOEF_R[0][1]
              
                dVh_dTheta1*dVh_dPsi1*U_theta1*U_psi1*Corr_coef_theta1_psi1[0][1]+
                dVh_dTheta2*dVh_dPsi1*U_theta2*U_psi1*Corr_coef_theta2_psi1[0][1]+
                dVh_dTheta1*dVh_dPsi2*U_theta1*U_psi2*Corr_coef_theta1_psi2[0][1]+
                dVh_dTheta2*dVh_dPsi2*U_theta2*U_psi2*Corr_coef_theta2_psi2[0][1]+
             
                dVh_dTheta1*dVh_dRho1*U_theta1*U_rho1*Corr_coef_theta1_rho1[0][1]+
                dVh_dTheta2*dVh_dRho1*U_theta2*U_psi1*Corr_coef_theta2_rho1[0][1]+
                dVh_dTheta1*dVh_dRho2*U_theta1*U_psi2*Corr_coef_theta1_rho2[0][1]+
                dVh_dTheta2*dVh_dRho2*U_theta2*U_psi2*Corr_coef_theta2_rho2[0][1]+
             
                dVh_dPsi1*dVh_dRho1*U_psi1*U_rho1*Corr_coef_rho1_psi1[0][1]+
                dVh_dPsi2*dVh_dRho1*U_psi2*U_rho1*Corr_coef_rho1_psi2[0][1]+
                dVh_dPsi1*dVh_dRho2*U_psi1*U_rho2*Corr_coef_rho2_psi1[0][1]+
                dVh_dPsi2*dVh_dRho2*U_psi2*U_rho2*Corr_coef_rho2_psi2[0][1]         
             )
        
        Uncertainty_Vh_GUM.append(np.sqrt((dVh_dTheta1*U_theta1)**2+(dVh_dTheta2*U_theta2)**2+(dVh_dPsi1*U_psi1)**2+(dVh_dPsi2*U_psi2)**2+(dVh_dRho1*U_rho1)**2+(dVh_dRho2*U_rho2)**2+2*R)[0])
    

    ##################################################################################################################################################
    ##################################################################################################################################################
    
#%% Print

    plt.figure()
    
    # plt.plot(np.degrees(wind_direction),u_wind_GUM,'-g', label='Uncertainty u component - GUM')
    # plt.plot(np.degrees(wind_direction),v_wind_GUM,'-b', label='Uncertainty v component - GUM')
    
    # plt.annotate('psi1_psi2_corr_n='+str(psi1_psi2_corr_n)+'\n'+'theta1_theta2_corr_n='+str(theta1_theta2_corr_n)+'\nrho1_rho2_corr_n='+str(rho1_rho2_corr_n)+'\nU_psi='+str(rho1_rho2_corr_n), xy=(0.05, 0.9), xycoords='axes fraction')
    # pdb.set_trace()
    plt.plot(np.degrees(wind_direction),Uncertainty_Vh_GUM,'-r', label='Uncertgainty Vh - GUM')
    plt.annotate('psi1_psi2_corr_n='+str(psi1_psi2_corr_n)+'\n'+'theta1_theta2_corr_n='+str(theta1_theta2_corr_n)+'\nrho1_rho2_corr_n='+str(rho1_rho2_corr_n), xy=(0.05, 0.9), xycoords='axes fraction')
    
    
    plt.plot(np.degrees(wind_direction),Uncertainty_U,'og',alpha=0.3,label=' Uncertainty u component - MC')
    # plt.annotate('Uncertainty U', xy=(0.05, 0.9), xycoords='axes fraction')
    
    # plt.figure()
    plt.plot(np.degrees(wind_direction),Uncertainty_V,'ob',alpha=0.3,label='Uncertainty v component - MC')
    # plt.annotate('Uncertainty V', xy=(0.05, 0.9), xycoords='axes fraction')
    plt.plot(np.degrees(wind_direction),Uncertainty_Vh_MC,'or',alpha=0.3, label='Uncertainty Vh - MC')
    plt.xlabel('Wind Direction (°)')
    plt.ylabel('Vh uncertainty [m/s]')
    
    plt.legend()
    
    
    plt.figure()
    plt.plot(np.degrees(wind_direction),U_VLOS1_MC,'.k',np.degrees(wind_direction),U_VLOS1_GUM,'-k',label='VLOS1')
    plt.plot(np.degrees(wind_direction),U_VLOS2_MC,'.b',np.degrees(wind_direction),U_VLOS2_GUM,'-b',label='VLOS2')
    plt.title(['Correlations = ['+str(psi1_theta1_corr_n)+', '+str(psi1_rho1_corr_n)+', '+str(theta1_rho1_corr_n)+']'])
    plt.xlabel('Wind Direction (°)')
    plt.ylabel('Vlos uncertainty [m/s]')
    
    
    plt.legend()

    
    
    
    pdb.set_trace()
    # Add lidar position:
    # vector_pos = x-x_Lidar,y-y_Lidar,z-z_Lidar
    
    vector_pos1=[]
    for ind_or in range(np.shape(Qlunc_yaml_inputs['Components']['Scanner']['Origin'])[0]):
        vector_pos1.append((x-x_Lidar1[ind_or],y-y_Lidar1[ind_or],z-z_Lidar1[ind_or]) ) 
    
    rho2,theta2,psi2=[],[],[]
    for indf in range(np.shape(vector_pos1)[0]):
        rho1,theta1,psi1 =SA.cart2sph(vector_pos1[indf][0],vector_pos1[indf][1],vector_pos1[indf][2])
        rho2.append(rho1)
        theta2.append(theta1)
        psi2.append(psi1)
    rho,theta,psi=np.round([rho2,theta2,psi2],4)
    
    # # MEasurement point in cartesian coordinates before applying lidar position
    # x1,y1,z1=SA.sph2cart(rho,theta,psi)
    
    # stdv focus distance, cone angle and azimuth:
    # stdv_param1 = Probe_param['Focus Distance uncertainty']
    # stdv_rho   = Lidar.optics.scanner.stdv_focus_dist    
    # stdv_theta = np.radians(Lidar.optics.scanner.stdv_cone_angle)
    # stdv_psi   = np.radians(Lidar.optics.scanner.stdv_azimuth)
    # stdv_wind_direction =np.radians(0)
    # stdv_wind_tilt = np.radians(0)

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
        
    pdb.set_trace()
    # Find the noisy points for the initial coordinates:
    
    # for ind_lid in range(np.shape(rho)[0]):
    #     for ind_noise in range(Lidar.optics.scanner.N_Points):
    #         rho_noisy.append(np.random.normal(rho[ind_lid][ind_noise],stdv_rho,Lidar.optics.scanner.N_MC))
    #         theta_noisy.append(np.random.normal(theta[ind_lid][ind_noise],stdv_theta,Lidar.optics.scanner.N_MC))
    #         psi_noisy.append(np.random.normal(psi[ind_lid][ind_noise],stdv_psi,Lidar.optics.scanner.N_MC))
    #         wind_direction_noisy.append(np.random.normal(wind_direction[ind_noise],stdv_wind_direction,Lidar.optics.scanner.N_MC))
    #         wind_tilt_noisy.append(np.random.normal(wind_tilt[ind_noise],stdv_wind_tilt,Lidar.optics.scanner.N_MC))
    rho_noise,theta_noise,psi_noise=[],[],[]
    for ind_lid in range(np.shape(rho)[0]):
        for ind_noise in range(Lidar.optics.scanner.N_Points):
            rho_noise.append(np.random.normal(rho[ind_lid][ind_noise],stdv_rho,Lidar.optics.scanner.N_MC))
            # rho_noisy2.append(np.random.normal(rho[ind_lid][ind_noise],stdv_rho,Lidar.optics.scanner.N_MC))
            theta_noise.append(np.random.normal(theta[ind_lid][ind_noise],stdv_theta,Lidar.optics.scanner.N_MC))
            # theta_noisy2.append(np.random.normal(theta[ind_lid][ind_noise],stdv_theta,Lidar.optics.scanner.N_MC))
            psi_noise.append(np.random.normal(psi[ind_lid][ind_noise],stdv_psi,Lidar.optics.scanner.N_MC))
            # psi_noisy2.append(np.random.normal(psi[ind_lid][ind_noise],stdv_psi,Lidar.optics.scanner.N_MC))
            wind_direction_noisy.append(np.random.normal(wind_direction[ind_noise],stdv_wind_direction,Lidar.optics.scanner.N_MC))
            wind_tilt_noisy.append(np.random.normal(wind_tilt[ind_noise],stdv_wind_tilt,Lidar.optics.scanner.N_MC))    
    rho_noisy1 = rho_noise[:len(x)]
    rho_noisy2 = rho_noise[len(x):]
    theta_noisy1 = theta_noise[:len(x)]
    theta_noisy2 = theta_noise[len(x):]    
    psi_noisy1 = psi_noise[:len(x)]
    psi_noisy2 = psi_noise[len(x):]
    # Create teh correlated distributions:
    # Create the correlated data for Vlos1 Vlos2
    # VLOS_means = [Vlos1_noisy.mean(), Vlos2_noisy.mean()]  
    # VLOS_stds = [Vlos1_noisy.std(), Vlos2_noisy.std()]
    
    # VLOS_covs = [[VLOS_stds[0]**2          , VLOS_stds[0]*VLOS_stds[1]*VLOS_corr_n], 
    #              [VLOS_stds[0]*VLOS_stds[1]*VLOS_corr_n,           VLOS_stds[1]**2]] 
    
    # VLOS_corr = np.random.multivariate_normal(VLOS_means, VLOS_covs, N_MC).T
    
    # #Covariance (Vlos1, Vlos2)
    # VLOS_covariance = 1/(N_MC-1)*sum((VLOS_corr[0]-VLOS_means[0])*(VLOS_corr[1]-VLOS_means[1]))
    # # Correlation coefficients Vlos
    # C_VLos = VLOS_covariance/(VLOS_stds[0]*VLOS_stds[1])
    
    # Create the correlated data for Theta1 Theta2
    theta_means = [theta_noisy1[0].mean(), theta_noisy2[0].mean()]  
    theta_stds  = [theta_noisy1[0].std(), theta_noisy2[0].std()]
    theta_corr_n=0.9
    theta_covs  = [[theta_stds[0]**2          , theta_stds[0]*theta_stds[1]*theta_corr_n], 
                   [theta_stds[0]*theta_stds[1]*theta_corr_n,           theta_stds[1]**2]] 
    
    theta_corr = np.random.multivariate_normal(theta_means, theta_covs, Lidar.optics.scanner.N_MC).T
    
    #Covariance (theta1, theta2)
    theta_covariance = 1/(Lidar.optics.scanner.N_MC-1)*sum((theta_corr[0]-theta_means[0])*(theta_corr[1]-theta_means[1]))
    # Correlation coefficients Theta
    C_theta = theta_covariance/(theta_stds[0]*theta_stds[1])
    
    
    # Create the correlated data for Psi1 Psi2
    
    psi_means = [psi1_noisy.mean(), psi2_noisy.mean()]  
    psi_stds  = [psi1_noisy.std(), psi2_noisy.std()]

    psi_covs = [[psi_stds[0]**2          , psi_stds[0]*psi_stds[1]*psi_corr_n], 
                [psi_stds[0]*psi_stds[1]*psi_corr_n,           psi_stds[1]**2]] 
    
    psi_corr = np.random.multivariate_normal(psi_means, psi_covs, Lidar.optics.scanner.N_MC).T
    
    #Covariance(psi1, psi2)
    psi_covariance = 1/(N_MC-1)*sum((psi_corr[0]-psi_means[0])*(psi_corr[1]-psi_means[1]))
    # Correlation coefficients Psi
    C_psi=psi_covariance/(psi_stds[0]*psi_stds[1])
    # Corr_coef_psi=np.corrcoef(psi_corr[0], psi_corr[1])

    # Convert angles to radians (just for calculations)
    
    theta_deg     = np.degrees(theta)
    psi_deg        = np.degrees(psi)  
    wind_direction_deg  = np.degrees(wind_direction)
    wind_tilt_deg  = np.degrees(wind_tilt)
    theta_noisy_deg = np.degrees(theta_noisy)
    psi_noisy_deg = np.degrees(psi_noisy)
    wind_direction_noisy_deg = np.degrees(wind_direction_noisy)
    wind_tilt_noisy_deg = np.radians(wind_tilt_noisy)
    stdv_theta_deg  = np.degrees(stdv_theta)
    stdv_psi_deg    = np.degrees(stdv_psi)

        # # This part commented out for now aim to calculate the error due to the inclinomenters:
            
        # rho_noisy0.append(np.random.normal(rho[ind_noise],stdv_rho,Lidar.optics.scanner.N_MC))
        # theta_noisy0.append(np.random.normal(theta[ind_noise],stdv_theta,Lidar.optics.scanner.N_MC))
        # psi_noisy0.append(np.random.normal(psi[ind_noise],stdv_psi,Lidar.optics.scanner.N_MC))
        
        # # Apply error in inclinometers   
        # # Rotation, due to inclinometers
        # noisy_yaw     = np.random.normal(0,stdv_yaw,Lidar.optics.scanner.N_MC)
        # noisy_pitch   = np.random.normal(0,stdv_pitch,Lidar.optics.scanner.N_MC)
        # noisy_roll    = np.random.normal(0,stdv_roll,Lidar.optics.scanner.N_MC)
        
        
        # # Convert noisy coordinates into cartesians to apply the rotation matrices
        # x_noisy, y_noisy,z_noisy = SA.sph2cart(rho_noisy0[ind_noise],(psi_noisy0[ind_noise]),(np.array(90)-theta_noisy0[ind_noise]))
        
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
    
    # pdb.set_trace()
    
    #%% MC Method 
    
    
    # 1. Calculate radial speed uncertainty for an homogeneous flow
    U_Vrad_homo_MC,U_Vrad_homo_MC_LOS1,U_Vrad_homo_MC_LOS2 = [],[],[]
    
    # 1.1 Relative uncertainty:
    # Vrad_homo = ([100*np.cos((theta_noisy[ind_theta]))*np.cos((psi_noisy[ind_theta]))/(np.cos((theta[ind_theta]))*np.cos((psi[ind_theta]))) for ind_theta in range (len(theta_noisy))])    
    
    # pdb.set_trace()
    # 1.2 Absolute uncertainty:
    # Vrad_homo2 = ([Vref*np.cos((theta_noisy[ind_theta]))*np.cos((psi_noisy[ind_theta])) for ind_theta in range (len(theta_noisy))])
    
    # 1.3 New approach(Absolute uncertainty):
    Vrad_homo1,Vrad_homo2=[],[]
    pdb.set_trace()
    Vrad_homo = ([Vref*(-np.cos(theta_noisy[ind_theta])*np.cos(psi_noisy[ind_theta])*np.cos(wind_direction_noisy[ind_theta])-np.cos(theta_noisy[ind_theta])*np.sin(psi_noisy[ind_theta])*np.sin(wind_direction_noisy[ind_theta])-np.sin(theta_noisy[ind_theta])*np.tan(wind_tilt_noisy[ind_theta])) for ind_theta in range (len(theta_noisy))])
    for i in range(0,100):
        Vrad_homo1.append(Vrad_homo[i])
        Vrad_homo2.append(Vrad_homo[i+100])
    # 1.4 Uncertainty (stdv):
    U_Vrad_homo_MC_LOS1.append([np.std(Vrad_homo1[ind_stdv])  for ind_stdv in range(len(Vrad_homo1))])
    U_Vrad_homo_MC_LOS2.append([np.std(Vrad_homo2[ind_stdv])  for ind_stdv in range(len(Vrad_homo2))])
    # U_Vrad_homo_MC2.append([np.std(Vrad_homo2[ind_stdv2])  for ind_stdv2 in range(len(Vrad_homo2))])
    pdb.set_trace()

   # 2 Uncertainty (power law)
    U_Vh_PL,U_Vrad_S_MC_REL,Vrad_PL_REL,Vrad_PL_REL0,U_Vrad_S_MC_ABS,Vrad_PL_ABS,Vrad_PL_REL1,U_Vrad_S_MC_REL1,Vrad_PL_ABS1,U_Vrad_S_MC_ABS1=[],[],[],[],[],[],[],[],[],[]
    for in_alpha in range(len(alpha)):   
        for ind_npoints in range(len(rho)): # Calculate the radial speed uncertainty for the noisy points 
            A=(((z_Lidar-Hg)+(np.sin((theta_noisy[ind_npoints]))*rho_noisy[ind_npoints]))/Href)
            B=(((z_Lidar-Hg)+(np.sin((theta[ind_npoints]))*rho[ind_npoints]))/Href)        
            # 2.1 Relative uncertainty:        
            # Vrad_PL_REL1.append (100*(np.cos((psi_noisy[ind_npoints]))*np.cos((theta_noisy[ind_npoints])))*((((z_Lidar-Hg)+np.sin((theta_noisy[ind_npoints]))*rho_noisy[ind_npoints])/Href)**alpha[0])\
            #                     /((np.cos((psi[ind_npoints]))*np.cos((theta[ind_npoints])))*((((z_Lidar-Hg)+np.sin((theta[ind_npoints]))*rho[ind_npoints])/Href)**alpha[0])))
            
            # 2.2 Absolute uncertainty:
            # Vrad_PL_ABS1.append (Vref*(np.cos((psi_noisy[ind_npoints]))*np.cos((theta_noisy[ind_npoints])))*(((Href+np.sin((theta_noisy[ind_npoints]))*rho_noisy[ind_npoints])/Href)**alpha[0]))
            
            # 2.3 New approach (Absolute uncertainty):
            Vrad_PL_ABS.append((Vref*np.sign(A)*(np.abs(A)**alpha[in_alpha])*(-np.cos(theta_noisy[ind_npoints])*np.cos(psi_noisy[ind_npoints])*np.cos(wind_direction_noisy[ind_npoints])-np.cos(theta_noisy[ind_npoints])*np.sin(psi_noisy[ind_npoints])*np.sin(wind_direction_noisy[ind_npoints])-np.sin(theta_noisy[ind_npoints])*np.tan(wind_tilt_noisy[ind_npoints]))))

            # 2.4 New approach (Relative uncertainty)
            Vrad_PL_REL.append(100*(np.sign(A)*(np.abs(A)**alpha[in_alpha])*(-np.cos((theta_noisy[ind_npoints]))*np.cos((psi_noisy[ind_npoints]))*np.cos((wind_direction_noisy[ind_npoints]))-np.cos((theta_noisy[ind_npoints]))*np.sin((psi_noisy[ind_npoints]))*np.sin((wind_direction_noisy[ind_npoints]))+np.sin((theta_noisy[ind_npoints]))*np.tan((wind_tilt_noisy[ind_npoints]))))\
                                /(np.sign(B)*(np.abs(B)**alpha[in_alpha])*(-np.cos((theta[ind_npoints]))*np.cos((psi[ind_npoints]))*np.cos((wind_direction[ind_npoints]))-np.cos((theta[ind_npoints]))*np.sin((psi[ind_npoints]))*np.sin((wind_direction[ind_npoints]))+np.sin((theta[ind_npoints]))*np.tan((wind_tilt[ind_npoints])))))

            
        
        # 2.4 Uncertainty (stdv): For this to be compared with Vrad_weighted[1] I need to weight Vrad_PL_REL
        
        # U_Vrad_S_MC_REL1.append([np.nanstd(Vrad_PL_REL1[ind_stdv]) for ind_stdv in range(len(Vrad_PL_REL1))])
        # U_Vrad_S_MC_ABS1.append([np.nanstd(Vrad_PL_ABS1[ind_stdv]) for ind_stdv in range(len(Vrad_PL_ABS1))])
        
        #New approach
        U_Vrad_S_MC_REL.append([np.nanstd(Vrad_PL_REL[ind_stdv]) for ind_stdv in range(len(Vrad_PL_REL))])
        U_Vrad_S_MC_ABS.append([np.nanstd(Vrad_PL_ABS[ind_stdv]) for ind_stdv in range(len(Vrad_PL_ABS))])
        

        
        
        # Vrad_PL_REL,Vrad_PL_ABS=[],[]
    
    # pdb.set_trace()
    # Scatter plot: calculating all values among the range of theta, psi and rho
    Vrad0_PL,U_Vrad_PL_REL_MC_Total=[],[]   
    Vrad_PL_REL_Total= ([Vref*((((z_Lidar-Hg)+(np.sin((theta_noisy0))*rho_noisy0))/Href)**alpha[0])*(-np.cos((theta_noisy0))*np.cos((psi_noisy0))*np.cos((wind_direction_noisy[0][0]))-np.cos((theta_noisy0))*np.sin((psi_noisy0))*np.sin((wind_direction_noisy[0][0]))+np.sin((theta_noisy0))*np.tan((wind_tilt_noisy[0][0])))  for theta_noisy0 in theta_noisy  for rho_noisy0 in rho_noisy for psi_noisy0 in psi_noisy])   

    # rfr= ([(theta_noisy0,psi_noisy0,rho_noisy0) for theta_noisy0 in theta_noisy for psi_noisy0 in psi_noisy for rho_noisy0 in rho_noisy])   
    
    U_Vrad_PL_REL_MC_Total.append([np.nanstd(Vrad_PL_REL_Total[ind_T]) for ind_T in range(len(Vrad_PL_REL_Total))])
    # U_Vrad_S_MC_REL=np.reshape(U_Vrad_PL_REL_MC_Total[0],(11,11,11))
    # pdb.set_trace()


    #%% GUM method
    
    # 1. Calculate radial speed uncertainty for an homogeneous flow
    U_Vrad_homo_GUM,U_Vrad_theta,U_Vrad_psi,U_Vh,U_Vrad_range=[],[],[],[],[]
    
    # 1.1 Relative Unceratinty (%)
    # U_Vrad_theta.append([100*np.tan((theta[ind_u]))*(stdv_theta) for ind_u in range(len(theta))])    
    # U_Vrad_psi.append([100*np.tan((psi[ind_u]))*(stdv_psi) for ind_u in range(len(theta))]) 
    
    
    # 1.2 Absolute uncertainty
    # U_Vrad_theta.append([Vref*np.cos((psi[ind_u]))*np.sin((theta[ind_u]))*(stdv_theta) for ind_u in range(len(theta))])
    # U_Vrad_psi.append([Vref*np.cos((theta[ind_u]))*np.sin((psi[ind_u]))*(stdv_psi) for ind_u in range(len(theta))])
    
    
    # 1.3 New approach (Absolute uncertainty):   
    U_Vrad_theta.append([Vref*(np.cos(theta[ind_u])*(np.tan(wind_tilt[ind_u])-np.tan(theta[ind_u])*np.cos(psi[ind_u]-wind_direction[ind_u])))*stdv_theta for ind_u in range(len(theta))])
    U_Vrad_psi.append([Vref*(np.cos(theta[ind_u]))*np.sin(psi[ind_u]-wind_direction[ind_u])*stdv_psi for ind_u in range(len(theta))])        

             
    # 1.4 Expanded uncertainty
    U_Vrad_homo_GUM.append([np.sqrt((U_Vrad_theta[0][ind_u])**2+(U_Vrad_psi[0][ind_u])**2) for ind_u in range(len(theta))])
    
    
    # 2. Calculate radial speed uncertainty for an heterogeneous flow:
    U_Vrad_sh_theta,U_Vrad_sh_psi,U_Vh_sh,U_Vrad_S_GUM,U_Vrad_sh_range= [],[],[],[],[]       
    
    # U_Vrad_sh_theta2,U_Vrad_S_GUM2=[],[]
    for ind_alpha in range(len(alpha)):
        
        # 2.1 Relative Uncertainty in %:
        # U_Vrad_sh_theta.append([np.sqrt((100*(stdv_theta)*((alpha[ind_alpha]*(rho[ind_u]*np.cos((theta[ind_u]))/(z_Lidar+rho[ind_u]*np.sin((theta[ind_u])))))-np.tan((theta[ind_u])) ))**2) for ind_u in range(len(theta))])
        # U_Vrad_sh_psi.append([np.sqrt((100*np.tan((psi[ind_u]))*(stdv_psi))**2) for ind_u in range(len(psi))])            
        # U_Vrad_sh_range.append([np.sqrt((100*np.sin((theta[ind_u]))*alpha[ind_alpha]/(rho[ind_u]*np.sin((theta[ind_u]))+z_Lidar)*stdv_rho)**2) for ind_u in range(len(rho))])
                 
        
       # 2.2 Absolute uncertainty
       #U_Vrad_sh_theta.append([Vref*(((np.sin((theta_noisy[ind_u]))*rho_noisy[ind_u])/(np.sin((theta[ind_u]))*rho[ind_u]))**alpha[ind_alpha])*np.cos((psi[ind_u]))*np.cos((theta[ind_u]))*(stdv_theta*theta[ind_u])*abs((alpha[ind_alpha]/math.tan((theta[ind_u])))-np.tan((theta[ind_u])) ) for ind_u in range(len(theta))])
        # U_Vrad_sh_theta.append([Vref*((((z_Lidar-Hg)+(np.sin((theta[ind_u]))*rho[ind_u]))/Href)**alpha[ind_alpha])*np.cos((psi[ind_u]))*np.cos((theta[ind_u]))*(stdv_theta)*((alpha[ind_alpha]*(rho[ind_u]*np.cos((theta[ind_u]))/(Href+rho[ind_u]*np.sin((theta[ind_u])))))-np.tan((theta[ind_u])) ) for ind_u in range(len(theta))])
        # U_Vrad_sh_psi.append([Vref*(((Href+np.sin((theta[ind_u]))*rho[ind_u])/(Href))**alpha[ind_alpha])*np.cos((theta[ind_u]))*np.sin((psi[ind_u]))*(stdv_psi) for ind_u in range(len(psi))])            
        # U_Vrad_sh_range.append([Vref*(((Href+np.sin((theta[ind_u]))*rho[ind_u])/(Href))**alpha[ind_alpha])*alpha[ind_alpha]*np.sin((theta[ind_u]))/(Href+(np.sin((theta[ind_u]))*rho[ind_u]))*np.cos((theta[ind_u]))*np.cos((psi[ind_u]))*(stdv_rho) for ind_u in range(len(rho))])

        
        # 2.3 New approach (Absolute uncertainty):
        # pdb.set_trace()
        # This is another approach for theta uncertainty: U_Vrad_sh_theta.append([Vref*np.cos((theta[ind_u]))*(((Href+(np.sin((theta[ind_u]))*rho[ind_u]))/Href)**alpha[ind_alpha])*((alpha[ind_alpha]*(np.tan((theta[ind_u]))*np.tan((wind_tilt[ind_u]))-np.cos((psi[ind_u]-wind_direction[ind_u])))*(rho[ind_u]*((np.cos((theta[ind_u]))))/(Href+(np.sin((theta[ind_u]))*rho[ind_u]))))+((np.cos((psi[ind_u]-wind_direction[ind_u]))*np.tan((theta[ind_u])))+np.tan((wind_tilt[ind_u]))))*(stdv_theta)  for ind_u in range(len(theta))])

        U_Vrad_sh_theta.append([(Vref*((np.sign(((z_Lidar-Hg)+(np.sin(theta[ind_u])*rho[ind_u]))/Href)*(np.abs(((z_Lidar-Hg)+(np.sin(theta[ind_u])*rho[ind_u]))/Href)**alpha[in_alpha])))*np.cos((theta[ind_u]))*(-np.tan((wind_tilt[ind_u]))*(1+(np.tan((theta[ind_u]))*alpha[ind_alpha]*rho[ind_u]*np.cos((theta[ind_u]))/((z_Lidar-Hg)+(np.sin((theta[ind_u]))*rho[ind_u]))))+(np.cos((psi[ind_u]-wind_direction[ind_u]))*(np.tan((theta[ind_u]))-((alpha[ind_alpha]*rho[ind_u]*np.cos((theta[ind_u]))/((z_Lidar-Hg)+(np.sin((theta[ind_u]))*rho[ind_u])))))))*(stdv_theta)) for ind_u in range(len(theta))])        
        U_Vrad_sh_psi.append([  Vref*np.cos((theta[ind_u]))*((np.sign(((z_Lidar-Hg)+(np.sin(theta[ind_u])*rho[ind_u]))/Href)*(np.abs(((z_Lidar-Hg)+(np.sin(theta[ind_u])*rho[ind_u]))/Href)**alpha[in_alpha])))*np.sin((psi[ind_u]-wind_direction[ind_u]))*(stdv_psi)  for ind_u in range(len(theta))])
        U_Vrad_sh_range.append([Vref*np.cos((theta[ind_u]))*((np.sign(((z_Lidar-Hg)+(np.sin(theta[ind_u])*rho[ind_u]))/Href)*(np.abs(((z_Lidar-Hg)+(np.sin(theta[ind_u])*rho[ind_u]))/Href)**alpha[in_alpha])))*alpha[ind_alpha]*(np.sin((theta[ind_u]))/((z_Lidar-Hg)+(np.sin((theta[ind_u]))*rho[ind_u])))*(-np.cos((psi[ind_u]-wind_direction[ind_u]))+(np.tan((theta[ind_u]))*np.tan((wind_tilt[ind_u]))))*(stdv_rho) for ind_u in range(len(theta))])

    
        # 2.4 Expanded uncertainty with contributions of theta, psi and rho
        U_Vrad_S_GUM.append([np.sqrt(((U_Vrad_sh_theta[ind_alpha][ind_u]))**2+((U_Vrad_sh_psi[ind_alpha][ind_u]))**2+((U_Vrad_sh_range[ind_alpha][ind_u]))**2) for ind_u in range(len(rho)) ])
        # U_Vrad_S_GUM.append([np.sqrt((np.mean(U_Vrad_sh_theta[ind_alpha][ind_u]))**2+(np.mean(U_Vrad_sh_psi[ind_alpha][ind_u]))**2+(np.mean(U_Vrad_sh_range[ind_alpha][ind_u]))**2) for ind_u in range(len(rho)) ])    

            # C=((z_Lidar-Hg)+(np.sin(theta[ind_u])*rho[ind_u]))/Href
            # (np.sign(((z_Lidar-Hg)+(np.sin(theta[ind_u])*rho[ind_u]))/Href)*(np.abs(((z_Lidar-Hg)+(np.sin(theta[ind_u])*rho[ind_u]))/Href)**alpha[in_alpha]))


#%% PArt of the former code    
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
    
    
    #%% Storing data
    Final_Output_UQ_Scanner                 = {'Vr Uncertainty homo MC [m/s]':U_Vrad_homo_MC,'Vr Uncertainty homo GUM [m/s]':U_Vrad_homo_GUM,'Vr Uncertainty MC [m/s]':U_Vrad_S_MC_ABS,'Vr Uncertainty GUM [m/s]':U_Vrad_S_GUM,'x':x,'y':y,'z':z,'rho':rho,'theta':theta,'psi':psi,'beam vector':vector_pos} #, 'Rayleigh length':Probe_param['Rayleigh Length'],'Rayleigh length uncertainty':Probe_param['Rayleigh Length uncertainty']}
    Lidar.lidar_inputs.dataframe['Scanner'] = (Final_Output_UQ_Scanner['Vr Uncertainty MC [m/s]'])*len(Atmospheric_Scenario.temperature)  
    
    # Plotting
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_Scanner,Qlunc_yaml_inputs['Flags']['Scanning Pattern'],False,False,False)
    
    
    # Scatter plot
    # pdb.set_trace()
    rho_scat,theta_scat,psi_scat,box=SA.mesh(rho,theta,psi)
    # QPlot.scatter3d(theta_scat,psi_scat,rho_scat,U_Vrad_PL_REL_MC_Total[0])
    
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
                   
                Scanner_Uncertainty,DataFrame=Lidar.optics.scanner.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
                # pdb.set_trace() 
                WFR_Uncertainty=None#Lidar.wfr_model.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,Scanner_Uncertainty)            
            
            else:  
                
                Scanner_Uncertainty,DataFrame=Lidar.optics.scanner.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
                # pdb.set_trace()
                WFR_Uncertainty = None
                # pdb.set_trace()
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
    
    # Final_Output_UQ_Optics = {'Uncertainty_Optics':Uncertainty_Optics_Module, 'Uncertainty_WFR':WFR_Uncertainty['WFR_Uncertainty'],'Mean_error_PointingAccuracy':Scanner_Uncertainty['Simu_Mean_Distance_Error'],'Stdv_PointingAccuracy':Scanner_Uncertainty['STDV_Distance'], 'Rayleigh length':Scanner_Uncertainty['Rayleigh length'],'Rayleigh length uncertainty':Scanner_Uncertainty['Rayleigh length uncertainty']}
    Final_Output_UQ_Optics = {'Uncertainty_Optics':Uncertainty_Optics_Module,'Uncertainty ': Scanner_Uncertainty}
    
    Lidar.lidar_inputs.dataframe['Optics Module']=Final_Output_UQ_Optics['Uncertainty_Optics']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))  # linspace to create the appropiate length for the xarray. 
    # pdb.set_trace()
    return Final_Output_UQ_Optics,Lidar.lidar_inputs.dataframe

