# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 20:51:10 2022

@author: fcosta



Calculate the wind direction uncertainty. The function reads the lidars in "Projects" and calculates the uncertainty in wind direction.
"""
from Utils.Qlunc_ImportModules import *
import Utils.Qlunc_Help_standAlone as SA
from Utils import Qlunc_Plotting as QPlot
def UQ_WinDir(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,Lidars):
    """.
    
    Estimates the uncertainty in the horizontal wind speed $V_{h}$ by using the Guide to the expression of Uncertainty in Measurements (GUM). Location: Qlunc_Help_standAlone.py
    
    Parameters
    ----------    
    * Lidar: Lidar object
    
    * Atomspheric_Scenario
    
    * cts: constants for calculations (don't need inputs from user)
    
    * Qlunc_yaml_inputs: inputs from yaml file. Lidar characteristics and measuring setup
    
    * Lidars: ID of the lidars meant to be compared
  
    Returns
    -------    
    Estimated uncertainty in horizontal wind speed
    
    """
    Href  = Qlunc_yaml_inputs['Components']['Scanner']['Href'],
    Vref  = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']
    alpha = Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']    
    Hg    = Qlunc_yaml_inputs['Atmospheric_inputs']['Height ground'] 
    Hl    = Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2]
    
    Lids = Lidars
        # Read the saved dictionary
    loaded_dict=[]
    for ind in Lids:
        with open('./Lidar_Projects/'+ind, 'rb') as f:
            loaded_dict.append( pickle.load(f))
    wind_direction =   loaded_dict[0]['Uncertainty']['wind direction']  
    
    if len (Lids)==2:
              
        Uncertainty_WindDir_MC,Uncertainty_WindDir_GUM=[],[]
        u_wind_GUM, v_wind_GUM=[],[]
        Vwind_MC,Uwind_MC,=[],[]

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
        for ind_wind_dir in range(len(wind_direction)):  
            # Vlos1_noisy  = np.random.normal(Vlos1[0],u_Vlos1,N_MC)
            # Vlos2_noisy  = np.random.normal(Vlos2[0],u_Vlos2,N_MC) 
            theta1_noisy = np.random.normal(loaded_dict[0]['Uncertainty']['Elevation angle'],loaded_dict[0]['Uncertainty']['STDVs'][0],Lidar.optics.scanner.N_MC)
            theta2_noisy = np.random.normal(loaded_dict[1]['Uncertainty']['Elevation angle'],loaded_dict[1]['Uncertainty']['STDVs'][0],Lidar.optics.scanner.N_MC)
            psi1_noisy   = np.random.normal(loaded_dict[0]['Uncertainty']['Azimuth'],loaded_dict[0]['Uncertainty']['STDVs'][1],Lidar.optics.scanner.N_MC)
            psi2_noisy   = np.random.normal(loaded_dict[1]['Uncertainty']['Azimuth'],loaded_dict[1]['Uncertainty']['STDVs'][1],Lidar.optics.scanner.N_MC)
            rho1_noisy   = np.random.normal(loaded_dict[0]['Uncertainty']['Focus distance'][0],loaded_dict[0]['Uncertainty']['STDVs'][2],Lidar.optics.scanner.N_MC)
            rho2_noisy   = np.random.normal(loaded_dict[1]['Uncertainty']['Focus distance'][0],loaded_dict[1]['Uncertainty']['STDVs'][2],Lidar.optics.scanner.N_MC)
            
            
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
            
            # Calculate the u and v wind components and their uncertainties
            
            # Break down large equations
            WindDir_MC = SA.U_WindDir_MC([Theta1_cr,Theta2_cr],[Psi1_cr,Psi2_cr],[Rho1_cr,Rho2_cr],wind_direction,ind_wind_dir,Href,Vref,alpha,Hl)   

            # # ucomponent estimation        
            # Uwind_MC.append(np.mean(u_wind))
            # # Uncertainty as standard deviation (k=1) in the u wind velocity component estimation
            # Uncertainty_U.append(np.std(u_wind))
            
            # # v component estimation        
            # Vwind_MC.append(np.mean(v_wind))
            # # Uncertainty as standard deviation (k=1) in the v wind velocity component estimation
            # Uncertainty_V.append(np.std(v_wind))
            # pdb.set_trace()
    
            #%% VH Montecarlo uncertainy ##############
            # Horizontal velocity estimation
            # Vh_MC=np.sqrt((u_wind**2)+(v_wind**2))
            # Uncertainty as standard deviation (k=1) in the horizontal velocity estimation
            Uncertainty_WindDir_MC.append(np.std(WindDir_MC))
            # pdb.set_trace()
            #%% VH GUM uncertainty#####################
            U = [loaded_dict[0]['Uncertainty']['STDVs'][0],loaded_dict[1]['Uncertainty']['STDVs'][0],loaded_dict[0]['Uncertainty']['STDVs'][1],loaded_dict[1]['Uncertainty']['STDVs'][1],loaded_dict[0]['Uncertainty']['STDVs'][2],loaded_dict[1]['Uncertainty']['STDVs'][2]]
            # pdb.set_trace()
            Coef = [theta1_theta2_corr_n,psi1_psi2_corr_n,rho1_rho2_corr_n,
                    Corr_coef_theta1_psi1[0][1],Corr_coef_theta2_psi1[0][1],Corr_coef_theta1_psi2[0][1],Corr_coef_theta2_psi2[0][1],
                    Corr_coef_theta1_rho1[0][1],Corr_coef_theta2_rho1[0][1],Corr_coef_theta1_rho2[0][1],Corr_coef_theta2_rho2[0][1],
                    Corr_coef_rho1_psi1[0][1],Corr_coef_rho1_psi2[0][1],Corr_coef_rho2_psi1[0][1],Corr_coef_rho2_psi2[0][1]]
            # Calculate coefficients for the GUM approach
            Uncertainty_WinDir_GUM_F = SA.U_WindDir_GUM([loaded_dict[0]['Uncertainty']['Elevation angle'],loaded_dict[1]['Uncertainty']['Elevation angle']],[loaded_dict[0]['Uncertainty']['Azimuth'][0],loaded_dict[1]['Uncertainty']['Azimuth'][0]],[loaded_dict[0]['Uncertainty']['Focus distance'],loaded_dict[1]['Uncertainty']['Focus distance']],wind_direction,ind_wind_dir,Href,Vref,alpha,Hl,U,Coef)   
    
            Uncertainty_WindDir_GUM.append(Uncertainty_WinDir_GUM_F)
        
        Final_Output_WindDir_Uncertainty = {'Wind direction Uncertainty GUM [m/s]':Uncertainty_WindDir_GUM,'Wind direction Uncertainty MC [m/s]':Uncertainty_WindDir_MC,'wind direction':wind_direction}
        
        
    #%% Plotting
    pdb.set_trace()
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_WindDir_Uncertainty,False,False,False,False,Qlunc_yaml_inputs['Flags']['Horizontal Velocity Uncertainty'])  #Qlunc_yaml_inputs['Flags']['Scanning Pattern']