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
    rho_noisy0,theta_noisy0,psi_noisy0,rho_noisy,theta_noisy1,theta_noisy2,psi_noisy,rho_noisy1,rho_noisy2, theta_noisy,psi_noisy1,psi_noisy2,wind_direction_TEST ,wind_tilt_TEST   = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
    Coordfinal_noisy,Coordfinal=[],[]
    coun=0
    sample_rate_count=0
    Href  = Qlunc_yaml_inputs['Components']['Scanner']['Href'],
    Vref  = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']
    alpha = Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']    
    Hg    = Qlunc_yaml_inputs['Atmospheric_inputs']['Height ground'] 
    Hl    = Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2]
    # #Call probe volume uncertainty function. 
    


    # R: Implement error in deployment of the tripod as a rotation over yaw, pitch and roll
    stdv_yaw    = np.array(np.radians(Lidar.lidar_inputs.yaw_error_dep))
    stdv_pitch  = np.array(np.radians(Lidar.lidar_inputs.pitch_error_dep))
    stdv_roll   = np.array(np.radians(Lidar.lidar_inputs.roll_error_dep))
    
   
    # Rho, theta and psi values of the measuring point
    
    rho0           = [Lidar.optics.scanner.focus_dist]  
    theta0         = [np.radians(Lidar.optics.scanner.cone_angle)]
    psi0           = [np.radians(Lidar.optics.scanner.azimuth)]
    # wind_direction = np.radians(np.array([Atmospheric_Scenario.wind_direction]*len(psi0)))
    wind_direction = np.radians(np.linspace(0,359,360))
    # wind_tilt      = np.radians( np.array([Atmospheric_Scenario.wind_tilt]*len(psi0)))

   
    # MEasurement point in cartesian coordinates before applying lidar position
    x,y,z=SA.sph2cart(rho0,theta0,psi0)
    
    
     # Lidars' position:
    class lidar_coor:
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
    
       
    lidars['Lidar_Rectangular']={'x':(lidar_coor.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2])[1]),
                                                    'y':(lidar_coor.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2])[2]),
                                                    'z':(lidar_coor.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2])[3])}
    lidars['Lidar_Spherical']={'rho':np.round((lidar_coor.Cart2Sph(lidars['Lidar_Rectangular']['x'],lidars['Lidar_Rectangular']['y'],lidars['Lidar_Rectangular']['z']))[1],4),
                                                  'theta':np.round((lidar_coor.Cart2Sph(lidars['Lidar_Rectangular']['x'],lidars['Lidar_Rectangular']['y'],lidars['Lidar_Rectangular']['z']))[2],4),
                                                   'psi':np.round((lidar_coor.Cart2Sph(lidars['Lidar_Rectangular']['x'],lidars['Lidar_Rectangular']['y'],lidars['Lidar_Rectangular']['z']))[3],4)}
    # if lidars['Lidar'+str(ind_origin+1)+'_Spherical']['psi']<0:
    #     lidars['Lidar'+str(ind_origin+1)+'_Spherical']['psi']=np.pi+lidars['Lidar'+str(ind_origin+1)+'_Spherical']['psi']
    
    # x_Lidar1.append(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0])
    # y_Lidar1.append(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1])
    # z_Lidar1.append(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2])
    # print(['Lidar_Spherical_psi'], '=' ,np.degrees(lidars['Lidar_Spherical']['psi']))
    # print(['Lidar_Spherical_theta'], '=' ,np.degrees(lidars['Lidar_Spherical']['theta']))
    pdb.set_trace()

    # Rho, theta and psi inputs and their uncertainties
    theta1,U_theta1 = lidars['Lidar_Spherical']['theta'],np.radians(Lidar.optics.scanner.stdv_cone_angle[0])
    # theta2,U_theta2 = np.array([0.0873]),0.026179938779914945      #   lidars['Lidar2_Spherical']['theta'],np.radians(Lidar.optics.scanner.stdv_cone_angle[1])
    psi1  ,U_psi1   = lidars['Lidar_Spherical']['psi'],np.radians(Lidar.optics.scanner.stdv_azimuth[0])
    # psi2  ,U_psi2   = np.array([0.0873]),0.026179938779914945  #lidars['Lidar2_Spherical']['psi'],np.radians(Lidar.optics.scanner.stdv_azimuth[1])
    rho1  ,U_rho1   = lidars['Lidar_Spherical']['rho'],Lidar.optics.scanner.stdv_focus_dist [0]
    # rho2  ,U_rho2   = np.array([1000.]),1   #lidars['Lidar2_Spherical']['rho'],Lidar.optics.scanner.stdv_focus_dist[1] 
    #Uncertainty in the probe volume    
    Probe_param = Lidar.probe_volume.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,lidars)
    
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
        
    # There is NO correlation between range and angles since the range is determined by the AOM (at least in pulsed lidars) and the angles accuracy is related to the alignment of the telescope mirrors,
    # to the position of the lense and also to the servos orienting the scanner
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
    
    # Corr_vec =[psi1_psi2_corr_n,
    #            theta1_theta2_corr_n ,
    #            rho1_rho2_corr_n,
    #            psi1_theta1_corr_n,
    #            psi1_theta2_corr_n,
    #            psi2_theta1_corr_n,  
    #            psi2_theta2_corr_n,
    #            psi1_rho1_corr_n,
    #            psi1_rho2_corr_n,
    #            psi2_rho1_corr_n,
    #            psi2_rho2_corr_n,
    #            theta1_rho1_corr_n, 
    #            theta1_rho2_corr_n,
    #            theta2_rho1_corr_n,
    #            theta2_rho2_corr_n]
    
    # Uncertainty of VLOS with wind direction
    
    # U_VLOS1_W,U_VLOS2_W,CORRCOEF_W = dVLOS_dw(theta1,theta2,psi1,psi2,rho1,rho2,wind_direction,Href,Vref,alpha)
    # pdb.set_trace()


    #%% 3) Create the noisy distributions (assume normal distributions):
    Uncertainty_V,Uncertainty_U,Uncertainty_Vh_MC,Uncertainty_Vh_GUM=[],[],[],[]
    u_wind_GUM, v_wind_GUM=[],[]
    Vwind_MC,Uwind_MC=[],[]
    CORR_COEF_uv,CORR_COEF_VLOS=[],[]
    U_VLOS1_GUM,U_VLOS1_MC,VLOS1_list=[],[],[]
 
    
    for ind_wind_dir in range(len(wind_direction)):  
        # Vlos1_noisy  = np.random.normal(Vlos1[0],u_Vlos1,N_MC)
        # Vlos2_noisy  = np.random.normal(Vlos2[0],u_Vlos2,N_MC) 
        # pdb.set_trace()
        theta1_noisy = np.random.normal(theta1[0],U_theta1,Lidar.optics.scanner.N_MC)
        psi1_noisy   = np.random.normal(psi1[0],U_psi1,Lidar.optics.scanner.N_MC)
        rho1_noisy   = np.random.normal(rho1[0],U_rho1,Lidar.optics.scanner.N_MC)
        # theta2_noisy = np.random.normal(theta2[0],U_theta2,Lidar.optics.scanner.N_MC)
        # psi2_noisy   = np.random.normal(psi2[0],U_psi2,Lidar.optics.scanner.N_MC)
        # rho2_noisy   = np.random.normal(rho2[0],U_rho2,Lidar.optics.scanner.N_MC)
        # # pdb.set_trace()
        
        # CORRCOEF_T=np.corrcoef(theta1_noisy,theta2_noisy)
        # CORRCOEF_P=np.corrcoef(psi1_noisy,psi2_noisy)
        # CORRCOEF_R=np.corrcoef(rho1_noisy,rho2_noisy)
        
        
        
        #%% 4) Obtain the Correlated distributions:
        
        # VLOS_means = [Vlos1_noisy.mean(), Vlos2_noisy.mean()]  
        # VLOS_stds  = [Vlos1_noisy.std(), Vlos2_noisy.std()]
        
        theta_means = [theta1_noisy.mean()]#,theta2_noisy.mean()]  
        theta_stds  = [theta1_noisy.std()]#,theta2_noisy.std()]
        
        psi_means = [psi1_noisy.mean()]#,psi2_noisy.mean()]  
        psi_stds  = [psi1_noisy.std()]#,psi2_noisy.std()]
        
        rho_means = [rho1_noisy.mean()]#,rho2_noisy.mean()]  
        rho_stds  = [rho1_noisy.std()]#,rho2_noisy.std()]
        
        # Covariance Matrix:
        cov_MAT=[[              theta_stds[0]**2,                        psi_stds[0]*theta_stds[0]*psi1_theta1_corr_n,     rho_stds[0]*theta_stds[0]*theta1_rho1_corr_n  ],
                  
                  [theta_stds[0]*psi_stds[0]*psi1_theta1_corr_n ,                        psi_stds[0]**2,                   rho_stds[0]*psi_stds[0]*psi1_rho1_corr_n],
                  
                  [theta_stds[0]*rho_stds[0]*theta1_rho1_corr_n,         psi_stds[0]*rho_stds[0]*psi1_rho1_corr_n,                     rho_stds[0]**2]]
        # cov_MAT=[[              theta_stds[0]**2,                     theta_stds[1]*theta_stds[0]*theta1_theta2_corr_n,   psi_stds[0]*theta_stds[0]*psi1_theta1_corr_n,   psi_stds[1]*theta_stds[0]*psi2_theta1_corr_n,   rho_stds[0]*theta_stds[0]*theta1_rho1_corr_n,  rho_stds[1]*theta_stds[0]*theta1_rho2_corr_n],
        #               [theta_stds[0]*theta_stds[1]*theta1_theta2_corr_n,                 theta_stds[1]**2,                     psi_stds[0]*theta_stds[1]*psi1_theta2_corr_n,   psi_stds[1]*theta_stds[1]*psi2_theta2_corr_n,   rho_stds[0]*theta_stds[1]*theta2_rho1_corr_n,  rho_stds[1]*theta_stds[1]*theta2_rho2_corr_n],
        #               [theta_stds[0]*psi_stds[0]*psi1_theta1_corr_n ,      theta_stds[1]*psi_stds[0]*psi1_theta2_corr_n,                   psi_stds[0]**2,                     psi_stds[1]*psi_stds[0]*psi1_psi2_corr_n,       rho_stds[0]*psi_stds[0]*psi1_rho1_corr_n,      rho_stds[1]*psi_stds[0]*psi1_rho2_corr_n],
        #               [theta_stds[0]*psi_stds[1]*psi2_theta1_corr_n,       theta_stds[1]*psi_stds[1]*psi2_theta2_corr_n,       psi_stds[0]*psi_stds[1]*psi1_psi2_corr_n,                   psi_stds[1]**2,                     rho_stds[0]*psi_stds[1]*psi2_rho1_corr_n,      rho_stds[1]*psi_stds[1]*psi2_rho2_corr_n],
        #               [theta_stds[0]*rho_stds[0]*theta1_rho1_corr_n,       theta_stds[1]*rho_stds[0]*theta2_rho1_corr_n,       psi_stds[0]*rho_stds[0]*psi1_rho1_corr_n,       psi_stds[1]*rho_stds[0]*psi2_rho1_corr_n,                   rho_stds[0]**2,                    rho_stds[1]*rho_stds[0]*rho1_rho2_corr_n],
        #               [theta_stds[0]*rho_stds[1]*theta1_rho2_corr_n,       theta_stds[1]*rho_stds[1]*theta2_rho2_corr_n,       psi_stds[0]*rho_stds[1]*psi1_rho2_corr_n,       psi_stds[1]*rho_stds[1]*psi2_rho2_corr_n,       rho_stds[0]*rho_stds[1]*rho1_rho2_corr_n,                  rho_stds[1]**2]]
                    
        # pdb.set_trace()          
     
        # # Multivariate distributions:
        # Theta1_cr,Theta2_cr,Psi1_cr,Psi2_cr,Rho1_cr,Rho2_cr = multivariate_normal.rvs([theta_means[0],theta_means[1],psi_means[0],psi_means[1],rho_means[0],rho_means[1]], cov_MAT,Lidar.optics.scanner.N_MC).T
        
        Theta1_cr,Psi1_cr,Rho1_cr = multivariate_normal.rvs([theta_means[0],psi_means[0],rho_means[0]], cov_MAT,Lidar.optics.scanner.N_MC).T

          # Theta - Psi
        #Covariance (theta1, psi1) as defined in GUM
        theta_psi_covariance = 1/(Lidar.optics.scanner.N_MC-1)*sum((Theta1_cr-theta_means[0])*(Psi1_cr-psi_means[0]))
        # Correlation coefficients Theta - Psi
        C_theta_psi = theta_psi_covariance/(theta_stds[0]*psi_stds[0])
        Corr_coef_theta_psi=np.corrcoef(Theta1_cr,Psi1_cr)
        
          # Theta - Rho
        #Covariance (theta1, rho1) as defined in GUM
        theta_rho_covariance = 1/(Lidar.optics.scanner.N_MC-1)*sum((Theta1_cr-theta_means[0])*(Rho1_cr-rho_means[0]))
        # Correlation coefficients Theta - Rho
        C_theta_rho = theta_rho_covariance/(theta_stds[0]*rho_stds[0])
        Corr_coef_theta_rho=np.corrcoef(Theta1_cr,Psi1_cr)        
        
       
           # Psi - Rho
        #Covariance (psi1, rho1) as defined in GUM
        psi_rho_covariance = 1/(Lidar.optics.scanner.N_MC-1)*sum((Psi1_cr-psi_means[0])*(Rho1_cr-rho_means[0]))
        # Correlation coefficients Theta - Rho
        C_psi_rho = psi_rho_covariance/(psi_stds[0]*rho_stds[0])
        Corr_coef_theta_rho=np.corrcoef(Psi1_cr,Rho1_cr)        
       
        #############
       
        # # Theta
        # #Covariance (theta1, theta2) as defined in GUM
        # theta_covariance = 1/(Lidar.optics.scanner.N_MC-1)*sum((Theta1_cr-theta_means[0])*(Theta2_cr-theta_means[1]))
        # # Correlation coefficients Theta 
        # C_theta = theta_covariance/(theta_stds[0]*theta_stds[1])
        # Corr_coef_theta=np.corrcoef(Theta1_cr,Theta2_cr)
        
        
        # # Psi
        # #Covariance(psi1, psi2) as defined in GUM
        # psi_covariance = 1/(Lidar.optics.scanner.N_MC-1)*sum((Psi1_cr-psi_means[0])*(Psi2_cr-psi_means[1]))
        # # Correlation coefficients PSi
        # C_psi=psi_covariance/(psi_stds[0]*psi_stds[1])
        # Corr_coef_psi=np.corrcoef(Psi1_cr,Psi2_cr)
        
        
        # # Rho
        # #Covariance(psi1, psi2) as defined in GUM
        # rho_covariance = 1/(Lidar.optics.scanner.N_MC-1)*sum((Rho1_cr-rho_means[0])*(Rho2_cr-rho_means[1]))
        # # Correlation coefficients PSi
        # C_rho=rho_covariance/(rho_stds[0]*rho_stds[1])
        # Corr_coef_rho=np.corrcoef(Rho1_cr, Rho2_cr)
        
        ################################
        
        
        
        
        
        
        # Cross correlations
        Corr_coef_theta1_psi1 = np.corrcoef(Theta1_cr,Psi1_cr)
        Corr_coef_theta1_rho1 = np.corrcoef(Theta1_cr,Rho1_cr)
        Corr_coef_rho1_psi1   = np.corrcoef(Rho1_cr,Psi1_cr)
        
        # Corr_coef_theta2_psi2 = np.corrcoef(Theta2_cr,Psi2_cr)
        # Corr_coef_theta2_rho2 = np.corrcoef(Theta2_cr,Rho2_cr)
        # Corr_coef_rho2_psi2   = np.corrcoef(Rho2_cr,Psi2_cr)
        
        # Corr_coef_theta1_psi2 = np.corrcoef(Theta1_cr,Psi2_cr)
        # Corr_coef_theta1_rho2 = np.corrcoef(Theta1_cr,Rho2_cr)
        # Corr_coef_rho1_psi2   = np.corrcoef(Rho1_cr,Psi2_cr)
        
        # Corr_coef_theta2_psi1 = np.corrcoef(Theta2_cr,Psi1_cr)
        # Corr_coef_theta2_rho1 = np.corrcoef(Theta2_cr,Rho1_cr)
        # Corr_coef_rho2_psi1   = np.corrcoef(Rho2_cr,Psi1_cr)
        
        # Cross correlations
                # CROS_CORR = [Corr_coef_theta1_psi1[0][1],Corr_coef_theta1_rho1[0][1],Corr_coef_rho1_psi1[0][1],Corr_coef_theta2_psi2[0][1],
                #               Corr_coef_theta2_rho2[0][1],Corr_coef_rho2_psi2[0][1],Corr_coef_psi[0][1],Corr_coef_theta[0][1],Corr_coef_rho[0][1]]
        
        
        
        
        # CROS_CORR = [psi1_theta1_corr_n,theta1_rho1_corr_n,psi1_rho1_corr_n,psi2_theta2_corr_n,theta2_rho2_corr_n,
        #               psi2_rho2_corr_n,  psi1_psi2_corr_n,theta1_theta2_corr_n,rho1_rho2_corr_n]
        # CROS_CORR = [C_theta_psi,C_theta_rho,C_psi_rho]
        CROS_CORR = [psi1_theta1_corr_n,theta1_rho1_corr_n,psi1_rho1_corr_n]
        #%% 5) VLOS uncertainty
        # pdb.set_trace()
        VLOS01,U_VLOS01,VLOS1_list = SA.U_VLOS_MC(Theta1_cr,Psi1_cr,Rho1_cr,Hl,Href,alpha,wind_direction,Vref,ind_wind_dir,VLOS1_list)
        U_VLOS1_MC.append(U_VLOS01)
        
        # Function calculating the uncertainties in VLOS:
        U_VLOS1 = SA.U_VLOS_GUM (theta1,psi1,rho1,U_theta1,U_psi1,U_rho1,U_VLOS01,Hl,Vref,Href,alpha,wind_direction,ind_wind_dir,CROS_CORR)
        U_VLOS1_GUM.append(U_VLOS1[0])
        # pdb.set_trace()
    
        # #%% 6) VH Uncertainty
        # # Calculate the u and v wind components and their uncertainties
        
        # # Break down large equations
        # u_wind,v_wind = SA.U_Vh_MC([Theta1_cr,Theta2_cr],[Psi1_cr,Psi2_cr],[Rho1_cr,Rho2_cr],wind_direction,ind_wind_dir,Href,Vref,alpha,Hl)   
        
        # # ucomponent estimation        
        # Uwind_MC.append(np.mean(u_wind))
        # # Uncertainty as standard deviation (k=1) in the u wind velocity component estimation
        # Uncertainty_U.append(np.std(u_wind))
        
        # # v component estimation        
        # Vwind_MC.append(np.mean(v_wind))
        # # Uncertainty as standard deviation (k=1) in the v wind velocity component estimation
        # Uncertainty_V.append(np.std(v_wind))

        # # VH Montecarlo uncertainy ##############
        # # Horizontal velocity estimation
        # Vh_MC=np.sqrt((u_wind**2)+(v_wind**2))
        # # Uncertainty as standard deviation (k=1) in the horizontal velocity estimation
        # Uncertainty_Vh_MC.append(np.std(Vh_MC))
    
        # # VH GUM uncertainty#####################
        # U = [U_theta1,U_theta2,U_psi1,U_psi2,U_rho1,U_rho2]
        # Coef = [theta1_theta2_corr_n,psi1_psi2_corr_n,rho1_rho2_corr_n,
        #         Corr_coef_theta1_psi1[0][1],Corr_coef_theta2_psi1[0][1],Corr_coef_theta1_psi2[0][1],Corr_coef_theta2_psi2[0][1],
        #         Corr_coef_theta1_rho1[0][1],Corr_coef_theta2_rho1[0][1],Corr_coef_theta1_rho2[0][1],Corr_coef_theta2_rho2[0][1],
        #         Corr_coef_rho1_psi1[0][1],Corr_coef_rho1_psi2[0][1],Corr_coef_rho2_psi1[0][1],Corr_coef_rho2_psi2[0][1]]
        # # Calculate coefficients for the GUM approach
        # Uncertainty_Vh_GUM_F = SA.U_Vh_GUM([theta1,theta2],[psi1[0],psi2[0]],[rho1,rho2],wind_direction,ind_wind_dir,Href,Vref,alpha,Hl,U,Coef)   

        # Uncertainty_Vh_GUM.append(Uncertainty_Vh_GUM_F)

    
    #########################################################################################################################
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
     #########################################################################################################################
       
    #%% MC Method for uncertainty when varying theta, psi OR rho
    
    wind_direction_TEST = np.radians([180])
    wind_tilt_TEST      = np.radians([0])
    theta_TEST = np.radians(np.linspace(1,89,500))
    psi_TEST   = np.radians(np.linspace(45,45,500))
    rho_TEST   = np.linspace(1000,1000,500)
    
    
    # 1. Calculate radial speed uncertainty for an heterogeneous flow
    
    
    U_Vrad_homo_MC,U_Vrad_homo_MC_LOS1,U_Vrad_homo_MC_LOS2 = [],[],[]
    VLOS_list_T,U_VLOS_T_MC,U_VLOS_T_GUM,U_VLOS_THomo_MC=[],[],[],[]
    for ind_0 in range(len(theta_TEST)):
        # 1.1 MC method
        VLOS_T_MC1=[]
        theta1_T_noisy = np.random.normal(theta_TEST[ind_0],U_theta1,Lidar.optics.scanner.N_MC)
        # theta2_noisy = np.random.normal(theta2[ind_0],U_theta2,Lidar.optics.scanner.N_MC)
        psi1_T_noisy   = np.random.normal(psi_TEST[ind_0],U_psi1,Lidar.optics.scanner.N_MC)
        # psi2_noisy   = np.random.normal(psi2[ind_0],U_psi2,Lidar.optics.scanner.N_MC)
        rho1_T_noisy   = np.random.normal(rho_TEST[ind_0],U_rho1,Lidar.optics.scanner.N_MC)
        # rho2_noisy   = np.random.normal(rho2[ind_0],U_rho2,Lidar.optics.scanner.N_MC)

        VLOS_T_MC,U_VLOS_T,VLOS_LIST_T         = SA.U_VLOS_MC(theta1_T_noisy,psi1_T_noisy,rho1_T_noisy,Hl,Href,alpha,wind_direction_TEST,Vref,0,VLOS_list_T)
        VLOS_THomo_MC,U_VLOS_THomo,VLOS_LIST_T = SA.U_VLOS_MC(theta1_T_noisy,psi1_T_noisy,rho1_T_noisy,Hl,Href, [0], wind_direction_TEST,Vref,0,VLOS_list_T)
        
        U_VLOS_T_MC.append(U_VLOS_T)
        U_VLOS_THomo_MC.append(U_VLOS_THomo)
    
        # 1.2 GUM method
    # pdb.set_trace()    
    U_VLOS_T_GUM     = SA.U_VLOS_GUM (theta_TEST,psi_TEST,rho_TEST,U_theta1,U_psi1,U_rho1,U_VLOS1,Hl,Vref,Href,alpha,wind_direction_TEST,0,[0,0,0]) 
    U_VLOS_THomo_GUM = SA.U_VLOS_GUM (theta_TEST,psi_TEST,rho_TEST,U_theta1,U_psi1,U_rho1,U_VLOS1,Hl,Vref,Href,[0],wind_direction_TEST,0,[0,0,0])        
            
 
####################################################################################    
    
####################################################################################    
    # pdb.set_trace()
   #  wind_direction_TEST = 0
   #  wind_tilt_TEST      = 0
   #  theta_TEST = np.linspace(1,89,100)
   #  psi_TEST   = np.linspace(43,43,100)
   #  rho_TEST   = np.linspace(1000,1000,100)
   #  # 1. Calculate radial speed uncertainty for an homogeneous flow
   #  U_Vrad_homo_MC,U_Vrad_homo_MC_LOS1,U_Vrad_homo_MC_LOS2 = [],[],[]
    
   #  # 1.1 Relative uncertainty:
   #  # Vrad_homo = ([100*np.cos((theta_noisy[ind_theta]))*np.cos((psi_noisy[ind_theta]))/(np.cos((theta[ind_theta]))*np.cos((psi_TEST[ind_theta]))) for ind_theta in range (len(theta_noisy))])    
    
    
   #  # 1.2 Absolute uncertainty:
   #  # Vrad_homo2 = ([Vref*np.cos((theta_noisy[ind_theta]))*np.cos((psi_noisy[ind_theta])) for ind_theta in range (len(theta_noisy))])
    
   #  # 1.3 New approach(Absolute uncertainty):
   #  Vrad_homo1,Vrad_homo2=[],[]
   
   #  Vrad_H1 = ([Vref*(-np.cos(theta1_noisy[ind_theta])*np.cos(psi1_noisy[ind_theta])*np.cos(wind_direction_TEST)-np.cos(theta1_noisy[ind_theta])*np.sin(psi1_noisy[ind_theta])*np.sin(wind_direction_TEST)-np.sin(theta1_noisy[ind_theta])*np.tan(wind_tilt_TEST)) for ind_theta in range (len(theta1_noisy))])
   #  Vrad_H2 = ([Vref*(-np.cos(theta2_noisy[ind_theta])*np.cos(psi2_noisy[ind_theta])*np.cos(wind_direction_TEST)-np.cos(theta2_noisy[ind_theta])*np.sin(psi2_noisy[ind_theta])*np.sin(wind_direction_TEST)-np.sin(theta2_noisy[ind_theta])*np.tan(wind_tilt_TEST)) for ind_theta in range (len(theta2_noisy))])
   
   #  for i in range(len(Vrad_H1)):
   #      Vrad_homo1.append(Vrad_H1)
   #      Vrad_homo2.append(Vrad_H2)
   #  # 1.4 Uncertainty (stdv):
   #  U_Vrad_homo_MC_LOS1.append([np.std(Vrad_homo1[ind_stdv])  for ind_stdv in range(len(Vrad_homo1))])
   #  U_Vrad_homo_MC_LOS2.append([np.std(Vrad_homo2[ind_stdv])  for ind_stdv in range(len(Vrad_homo2))])
   #  # U_Vrad_homo_MC2.append([np.std(Vrad_homo2[ind_stdv2])  for ind_stdv2 in range(len(Vrad_homo2))])
    

   # # 2 Uncertainty (power law)
   #  U_Vh_PL,U_Vrad_S_MC_REL,Vrad_PL_REL,Vrad_PL_REL0,U_Vrad_S_MC_ABS,Vrad_PL_ABS,Vrad_PL_REL1,U_Vrad_S_MC_REL1,Vrad_PL_ABS1,U_Vrad_S_MC_ABS1=[],[],[],[],[],[],[],[],[],[]
   #  for in_alpha in range(len(alpha)):   
   #      for ind_npoints in range(len(rho_TEST)): # Calculate the radial speed uncertainty for the noisy points 
   #          A=(((Hl)+(np.sin((theta1_noisy[ind_npoints]))*rho1_noisy[ind_npoints]))/Href)
   #          B=(((Hl)+(np.sin((theta_TEST[ind_npoints]))*rho_TEST[ind_npoints]))/Href)        
   #          # 2.1 Relative uncertainty:        
   #          # Vrad_PL_REL1.append (100*(np.cos((psi_noisy[ind_npoints]))*np.cos((theta_noisy[ind_npoints])))*((((z_Lidar-Hg)+np.sin((theta_noisy[ind_npoints]))*rho_noisy[ind_npoints])/Href)**alpha[0])\
   #          #                     /((np.cos((psi_TEST[ind_npoints]))*np.cos((theta[ind_npoints])))*((((z_Lidar-Hg)+np.sin((theta[ind_npoints]))*rho[ind_npoints])/Href)**alpha[0])))
            
   #          # 2.2 Absolute uncertainty:
   #          # Vrad_PL_ABS1.append (Vref*(np.cos((psi_noisy[ind_npoints]))*np.cos((theta_noisy[ind_npoints])))*(((Href+np.sin((theta_noisy[ind_npoints]))*rho_noisy[ind_npoints])/Href)**alpha[0]))
            
   #          # 2.3 New approach (Absolute uncertainty):
   #          Vrad_PL_ABS.append((Vref*np.sign(A)*(np.abs(A)**alpha[in_alpha])*(-np.cos(theta1_noisy[ind_npoints])*np.cos(psi_TEST_noisy[ind_npoints])*np.cos(wind_direction_TEST)-np.cos(theta1_noisy[ind_npoints])*np.sin(psi1_noisy[ind_npoints])*np.sin(wind_direction_TEST)-np.sin(theta1_noisy[ind_npoints])*np.tan(wind_tilt_TEST))))

   #          # 2.4 New approach (Relative uncertainty)
   #          Vrad_PL_REL.append(100*(np.sign(A)*(np.abs(A)**alpha[in_alpha])*(-np.cos((theta1_noisy[ind_npoints]))*np.cos((psi1_noisy[ind_npoints]))*np.cos((wind_direction_TEST))-np.cos((theta1_noisy[ind_npoints]))*np.sin((psi1_noisy[ind_npoints]))*np.sin((wind_direction_TEST))+np.sin((theta1_noisy[ind_npoints]))*np.tan((wind_tilt_TEST))))\
   #                              /(np.sign(B)*(np.abs(B)**alpha[in_alpha])*(-np.cos((theta_TEST[ind_npoints]))*np.cos((psi_TEST[ind_npoints]))*np.cos((wind_direction[ind_npoints]))-np.cos((theta_TEST[ind_npoints]))*np.sin((psi_TEST[ind_npoints]))*np.sin((wind_direction[ind_npoints]))+np.sin((theta_TEST[ind_npoints]))*np.tan((wind_tilt_TEST)))))

            
        
   #      # 2.4 Uncertainty (stdv): For this to be compared with Vrad_weighted[1] I need to weight Vrad_PL_REL
        
   #      # U_Vrad_S_MC_REL1.append([np.nanstd(Vrad_PL_REL1[ind_stdv]) for ind_stdv in range(len(Vrad_PL_REL1))])
   #      # U_Vrad_S_MC_ABS1.append([np.nanstd(Vrad_PL_ABS1[ind_stdv]) for ind_stdv in range(len(Vrad_PL_ABS1))])
        
   #      #New approach
   #      U_Vrad_S_MC_REL.append([np.nanstd(Vrad_PL_REL[ind_stdv]) for ind_stdv in range(len(Vrad_PL_REL))])
   #      U_Vrad_S_MC_ABS.append([np.nanstd(Vrad_PL_ABS[ind_stdv]) for ind_stdv in range(len(Vrad_PL_ABS))])
        

        
        
   #      # Vrad_PL_REL,Vrad_PL_ABS=[],[]
    
    
   #  # Scatter plot: calculating all values among the range of theta, psi and rho
   #  Vrad0_PL,U_Vrad_PL_REL_MC_Total=[],[]   
   #  Vrad_PL_REL_Total= ([Vref*((((z_Lidar-Hg)+(np.sin((theta_noisy0))*rho_noisy0))/Href)**alpha[0])*(-np.cos((theta_noisy0))*np.cos((psi_noisy0))*np.cos((wind_direction_TEST))-np.cos((theta_noisy0))*np.sin((psi_noisy0))*np.sin((wind_direction_TEST))+np.sin((theta_noisy0))*np.tan((wind_tilt_TEST)))  for theta_noisy0 in theta_noisy  for rho_noisy0 in rho_noisy for psi_noisy0 in psi_noisy])   

   #  # rfr= ([(theta_noisy0,psi_noisy0,rho_noisy0) for theta_noisy0 in theta_noisy for psi_noisy0 in psi_noisy for rho_noisy0 in rho_noisy])   
    
   #  U_Vrad_PL_REL_MC_Total.append([np.nanstd(Vrad_PL_REL_Total[ind_T]) for ind_T in range(len(Vrad_PL_REL_Total))])
   #  # U_Vrad_S_MC_REL=np.reshape(U_Vrad_PL_REL_MC_Total[0],(11,11,11))
    


   #  #%% GUM method
    
   #  # 1. Calculate radial speed uncertainty for an homogeneous flow
   #  U_Vrad_homo_GUM,U_Vrad_theta1,U_Vrad_psi1,U_Vh,U_Vrad_range=[],[],[],[],[]
    
   #  # 1.1 Relative Unceratinty (%)
   #  # U_Vrad_theta.append([100*np.tan((theta[ind_u]))*(stdv_theta) for ind_u in range(len(theta))])    
   #  # U_Vrad_psi.append([100*np.tan((psi[ind_u]))*(stdv_psi) for ind_u in range(len(theta))]) 
    
    
   #  # 1.2 Absolute uncertainty
   #  # U_Vrad_theta.append([Vref*np.cos((psi[ind_u]))*np.sin((theta[ind_u]))*(stdv_theta) for ind_u in range(len(theta))])
   #  # U_Vrad_psi.append([Vref*np.cos((theta[ind_u]))*np.sin((psi[ind_u]))*(stdv_psi) for ind_u in range(len(theta))])
    
    
   #  # 1.3 New approach (Absolute uncertainty):   
   #  U_Vrad_theta1.append([Vref*(np.cos(theta_TEST[ind_u])*(np.tan(wind_tilt_TEST)-np.tan(theta_TEST[ind_u])*np.cos(psi_TEST[ind_u]-wind_direction_TEST)))*U_theta1 for ind_u in range(len(theta_TEST))])
   #  U_Vrad_psi1.append([Vref*(np.cos(theta_TEST[ind_u]))*np.sin(psi_TEST[ind_u]-wind_direction_TEST)*U_psi1 for ind_u in range(len(theta_TEST))])        

             
   #  # 1.4 Expanded uncertainty
   #  U_Vrad_homo_GUM.append([np.sqrt((U_Vrad_theta1[0][ind_u])**2+(U_Vrad_psi1[0][ind_u])**2) for ind_u in range(len(theta_TEST))])
    
    
   #  # 2. Calculate radial speed uncertainty for an heterogeneous flow:
   #  U_Vrad_sh_theta1,U_Vrad_sh_psi1,U_Vh_sh,U_Vrad_S_GUM,U_Vrad_sh_range1= [],[],[],[],[]       
    
   #  # U_Vrad_sh_theta2,U_Vrad_S_GUM2=[],[]
   #  for ind_alpha in range(len(alpha)):
        
   #      # 2.1 Relative Uncertainty in %:
   #      # U_Vrad_sh_theta.append([np.sqrt((100*(stdv_theta)*((alpha[ind_alpha]*(rho[ind_u]*np.cos((theta[ind_u]))/(z_Lidar+rho[ind_u]*np.sin((theta[ind_u])))))-np.tan((theta[ind_u])) ))**2) for ind_u in range(len(theta))])
   #      # U_Vrad_sh_psi1.append([np.sqrt((100*np.tan((psi_TEST[ind_u]))*(U_psi1))**2) for ind_u in range(len(psi_TEST))])            
   #      # U_Vrad_sh_range.append([np.sqrt((100*np.sin((theta[ind_u]))*alpha[ind_alpha]/(rho[ind_u]*np.sin((theta[ind_u]))+Hl)*stdv_rho)**2) for ind_u in range(len(rho))])
                 
        
   #     # 2.2 Absolute uncertainty
   #     #U_Vrad_sh_theta.append([Vref*(((np.sin((theta_noisy[ind_u]))*rho_noisy[ind_u])/(np.sin((theta[ind_u]))*rho[ind_u]))**alpha[ind_alpha])*np.cos((psi_TEST[ind_u]))*np.cos((theta[ind_u]))*(stdv_theta*theta[ind_u])*abs((alpha[ind_alpha]/math.tan((theta[ind_u])))-np.tan((theta[ind_u])) ) for ind_u in range(len(theta))])
   #      # U_Vrad_sh_theta.append([Vref*((((z_Lidar-Hg)+(np.sin((theta[ind_u]))*rho[ind_u]))/Href)**alpha[ind_alpha])*np.cos((psi_TEST[ind_u]))*np.cos((theta[ind_u]))*(stdv_theta)*((alpha[ind_alpha]*(rho[ind_u]*np.cos((theta[ind_u]))/(Href+rho[ind_u]*np.sin((theta[ind_u])))))-np.tan((theta[ind_u])) ) for ind_u in range(len(theta))])
   #      # U_Vrad_sh_psi1.append([Vref*(((Href+np.sin((theta[ind_u]))*rho[ind_u])/(Href))**alpha[ind_alpha])*np.cos((theta[ind_u]))*np.sin((psi_TEST[ind_u]))*(U_psi1) for ind_u in range(len(psi_TEST))])            
   #      # U_Vrad_sh_range.append([Vref*(((Href+np.sin((theta[ind_u]))*rho[ind_u])/(Href))**alpha[ind_alpha])*alpha[ind_alpha]*np.sin((theta[ind_u]))/(Href+(np.sin((theta[ind_u]))*rho[ind_u]))*np.cos((theta[ind_u]))*np.cos((psi_TEST[ind_u]))*(stdv_rho) for ind_u in range(len(rho))])

        
   #      # 2.3 New approach (Absolute uncertainty):
   #      # pdb.set_trace()
   #      # This is another approach for theta uncertainty: U_Vrad_sh_theta.append([Vref*np.cos((theta[ind_u]))*(((Href+(np.sin((theta[ind_u]))*rho[ind_u]))/Href)**alpha[ind_alpha])*((alpha[ind_alpha]*(np.tan((theta[ind_u]))*np.tan((wind_direction_TEST))-np.cos((psi_TEST[ind_u]-wind_direction[ind_u])))*(rho[ind_u]*((np.cos((theta[ind_u]))))/(Href+(np.sin((theta[ind_u]))*rho[ind_u]))))+((np.cos((psi_TEST[ind_u]-wind_direction[ind_u]))*np.tan((theta[ind_u])))+np.tan((wind_direction_TEST))))*(stdv_theta)  for ind_u in range(len(theta))])

   #      U_Vrad_sh_theta1.append([(Vref*((np.sign(((Hl-Hg)+(np.sin(theta_TEST[ind_u])*rho_TEST[ind_u]))/Href)*(np.abs(((Hl-Hg)+(np.sin(theta_TEST[ind_u])*rho_TEST[ind_u]))/Href)**alpha[in_alpha])))*np.cos((theta_TEST[ind_u]))*(-np.tan((wind_direction_TEST))*(1+(np.tan((theta_TEST[ind_u]))*alpha[ind_alpha]*rho_TEST[ind_u]*np.cos((theta_TEST[ind_u]))/((Hl-Hg)+(np.sin((theta_TEST[ind_u]))*rho_TEST[ind_u]))))+(np.cos((psi_TEST[ind_u]-wind_direction_TEST))*(np.tan((theta_TEST[ind_u]))-((alpha[ind_alpha]*rho_TEST[ind_u]*np.cos((theta_TEST[ind_u]))/((Hl-Hg)+(np.sin((theta_TEST[ind_u]))*rho_TEST[ind_u])))))))*U_theta1) for ind_u in range(len(theta_TEST))])        
   #      U_Vrad_sh_psi1.append([  Vref*np.cos((theta_TEST[ind_u]))*((np.sign(((Hl-Hg)+(np.sin(theta_TEST[ind_u])*rho_TEST[ind_u]))/Href)*(np.abs(((Hl-Hg)+(np.sin(theta_TEST[ind_u])*rho_TEST[ind_u]))/Href)**alpha[in_alpha])))*np.sin((psi_TEST[ind_u]-wind_direction_TEST))*U_psi1  for ind_u in range(len(theta_TEST))])
   #      U_Vrad_sh_range1.append([Vref*np.cos((theta_TEST[ind_u]))*((np.sign(((Hl-Hg)+(np.sin(theta_TEST[ind_u])*rho_TEST[ind_u]))/Href)*(np.abs(((Hl-Hg)+(np.sin(theta_TEST[ind_u])*rho_TEST[ind_u]))/Href)**alpha[in_alpha])))*alpha[ind_alpha]*(np.sin((theta_TEST[ind_u]))/((Hl-Hg)+(np.sin((theta_TEST[ind_u]))*rho_TEST[ind_u])))*(-np.cos((psi_TEST[ind_u]-wind_direction_TEST))+(np.tan((theta_TEST[ind_u]))*np.tan((wind_direction_TEST))))*U_rho1 for ind_u in range(len(theta_TEST))])

    
   #      # 2.4 Expanded uncertainty with contributions of theta, psi_TEST and rho_TEST
   #      U_Vrad_S_GUM.append([np.sqrt(((U_Vrad_sh_theta1[ind_alpha][ind_u]))**2+((U_Vrad_sh_psi1[ind_alpha][ind_u]))**2+((U_Vrad_sh_range1[ind_alpha][ind_u]))**2) for ind_u in range(len(rho_TEST)) ])
   #      # U_Vrad_S_GUM.append([np.sqrt((np.mean(U_Vrad_sh_theta[ind_alpha][ind_u]))**2+(np.mean(U_Vrad_sh_psi1[ind_alpha][ind_u]))**2+(np.mean(U_Vrad_sh_range[ind_alpha][ind_u]))**2) for ind_u in range(len(rho_TEST)) ])    

   #          # C=((Hl-Hg)+(np.sin(theta[ind_u])*rho_TEST[ind_u]))/Href
   #          # (np.sign(((Hl-Hg)+(np.sin(theta[ind_u])*rho_TEST[ind_u]))/Href)*(np.abs(((Hl-Hg)+(np.sin(theta[ind_u])*rho_TEST[ind_u]))/Href)**alpha[in_alpha]))

####################################################################################    

####################################################################################    
    
    #%% Storing data
    
    Final_Output_UQ_Scanner                 = {'VLOS1 Uncertainty MC [m/s]':U_VLOS1_MC,'VLOS1 Uncertainty GUM [m/s]':U_VLOS1_GUM,
                                               'Vr Uncertainty homo MC [m/s]':U_VLOS_THomo_MC,'Vr Uncertainty homo GUM [m/s]':U_VLOS_THomo_GUM,'Vr Uncertainty MC [m/s]':U_VLOS_T_MC,'Vr Uncertainty GUM [m/s]':U_VLOS_T_GUM,
                                               'x':x,'y':y,'z':z,'rho':rho_TEST,'theta':theta_TEST,'psi':psi_TEST,'wind direction':wind_direction,'rho_Vh':rho1,'theta_Vh':theta1,'psi_Vh':psi1,'STDVs':[U_theta1,U_psi1,U_rho1]} #, 'Rayleigh length':Probe_param['Rayleigh Length'],'Rayleigh length uncertainty':Probe_param['Rayleigh Length uncertainty']}
    
    Lidar.lidar_inputs.dataframe['Scanner'] = (Final_Output_UQ_Scanner['Vr Uncertainty MC [m/s]'])*len(Atmospheric_Scenario.temperature)
    Lidar.lidar_inputs.dataframe['Probe Volume'] = Probe_param
    
    # Plotting
    # pdb.set_trace()
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_Scanner,Qlunc_yaml_inputs['Flags']['Line of sight Velocity Uncertainty'],False,False,False,False)  #Qlunc_yaml_inputs['Flags']['Scanning Pattern']
    
    
    # Scatter plot
    # rho_scat,theta_scat,psi_scat,box=SA.mesh(rho_TEST,theta_TEST,psi_TEST)
    # QPlot.scatter3d(theta_scat,psi_scat,rho_scat,U_Vrad_PL_REL_MC_Total[0])
    
    # pdb.set_trace()
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
    # pdb.set_trace()
    # Optical_Circulator_Uncertainty = [np.array(Lidar.optics.optical_circulator.insertion_loss)]
    Optical_Circulator_Uncertainty_w = [Qlunc_yaml_inputs['Components']['Laser']['Output power']/(10**(Lidar.optics.optical_circulator.SNR/10))]
    Optical_Circulator_Uncertainty_dB = 10*np.log10(Optical_Circulator_Uncertainty_w)
    
    # Add to the dictionary
    Final_Output_UQ_Optical_Circulator={'Optical_Circulator_Uncertainty':Optical_Circulator_Uncertainty_dB}
    Lidar.lidar_inputs.dataframe['Optical circulator']=Final_Output_UQ_Optical_Circulator['Optical_Circulator_Uncertainty']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))  # linspace to create the appropiate length for the xarray. 
    
    return Final_Output_UQ_Optical_Circulator,Lidar.lidar_inputs.dataframe

#%% TELESCOPE NOT IMPLEMENTED
def UQ_Telescope(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
     # pdb.set_trace()
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
    # pdb.set_trace()
    # Scanner
    if Lidar.optics.scanner != None:
        try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations            
            # pdb.set_trace()     
        
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
        # pdb.set_trace()
    # Telescope
    if Lidar.optics.telescope != 'None':
        try:
            Telescope_Uncertainty,DataFrame=Lidar.optics.telescope.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
            List_Unc_optics.append(Telescope_Uncertainty['Telescope_Uncertainty'])            
            # pdb.set_trace()
        except:
            Telescope_Uncertainty=None
            print(colored('Error in telescope uncertainty calculations!','cyan', attrs=['bold']))
            # pdb.set_trace()
    else:
        print (colored('You didn´t include a telescope in the lidar,so that telescope uncertainty contribution is not in lidar uncertainty estimations.','cyan', attrs=['bold']))
        # pdb.set_trace()
    # Optical Circulator
    if Lidar.optics.optical_circulator != 'None': 
        try:
            Optical_circulator_Uncertainty,DataFrame = Lidar.optics.optical_circulator.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
            List_Unc_optics.append(Optical_circulator_Uncertainty['Optical_Circulator_Uncertainty'])       
        except:
            Optical_circulator_Uncertainty = None
            print(colored('Error in optical circulator uncertainty calculations!','cyan', attrs=['bold']))    
            # pdb.set_trace()
    else:
        print(colored('You didn´t include an optical circulator in the lidar,so that optical circulator uncertainty contribution is not in lidar uncertainty estimations.','cyan', attrs=['bold']))
        # pdb.set_trace()
    Uncertainty_Optics_Module=SA.unc_comb(List_Unc_optics)
    
    # Final_Output_UQ_Optics = {'Uncertainty_Optics':Uncertainty_Optics_Module, 'Uncertainty_WFR':WFR_Uncertainty['WFR_Uncertainty'],'Mean_error_PointingAccuracy':Scanner_Uncertainty['Simu_Mean_Distance_Error'],'Stdv_PointingAccuracy':Scanner_Uncertainty['STDV_Distance'], 'Rayleigh length':Scanner_Uncertainty['Rayleigh length'],'Rayleigh length uncertainty':Scanner_Uncertainty['Rayleigh length uncertainty']}
    Final_Output_UQ_Optics = {'Uncertainty_Optics':Uncertainty_Optics_Module,'Uncertainty': Scanner_Uncertainty}
    
    Lidar.lidar_inputs.dataframe['Optics Module']=Final_Output_UQ_Optics['Uncertainty_Optics']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))  # linspace to create the appropiate length for the xarray. 
    # pdb.set_trace()
    return Final_Output_UQ_Optics,Lidar.lidar_inputs.dataframe

