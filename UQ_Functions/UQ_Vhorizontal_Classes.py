# -*- coding: utf-8 -*-
""".

Created on Sat Nov  5 19:04:36 2022

@author: fcosta

Calculate the Vh uncertainty. The function reads the lidars in "Projects" and calculates the uncertainty in Vh if 2 lidars or 3D vector wind
velocity uncertainty if there are three lidars. 
"""
from Utils.Qlunc_ImportModules import *
import Utils.Qlunc_Help_standAlone as SA
from Utils import Qlunc_Plotting as QPlot
def Call_a_Project(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,Lidars):
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
    pdb.set_trace()
    # Href  = Qlunc_yaml_inputs['Components']['Scanner']['Href'],
    # Vref  = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']
    # alpha = Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']    
    # Hg    = Qlunc_yaml_inputs['Atmospheric_inputs']['Height ground'] 
    # Hl    = Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2]
    
    Lids = Lidars
        # Read the saved dictionary
    loaded_dict=[]
    for ind in Lids:
        print(ind)
        with open('./Lidar_Projects/'+ind, 'rb') as f:
            loaded_dict.append( pickle.load(f))
    # pdb.set_trace()
    wind_direction =   loaded_dict[0]['Uncertainty'][0]['wind direction']  
    
    # if len (Lids)==2:
    #     Vh_U,Vlos1,Vlos2=[],[],[]
    #     for n_pp in range(len(loaded_dict[0]['Uncertainty'])):
    #         Uncertainty_V,Uncertainty_U,Uncertainty_Vh_MC,Uncertainty_Vh_GUM,Uncertainty_u_GUM=[],[],[],[],[]
    #         u_wind_GUM, v_wind_GUM=[],[]
    #         Vwind_MC,Uwind_MC,=[],[]
    
    #         psi1_psi2_corr_n     = Lidar.optics.scanner.correlations[0]  # correlation between psi1 and psi2
    #         theta1_theta2_corr_n = Lidar.optics.scanner.correlations[1] # correlation Theta1 and Theta2
    #         rho1_rho2_corr_n     = Lidar.optics.scanner.correlations[2]  # correlation Rho1 and Rho2
            
    #         # Cross correlations:
    #         psi1_theta1_corr_n  = Lidar.optics.scanner.correlations[3]
    #         psi1_theta2_corr_n  = Lidar.optics.scanner.correlations[4]
    #         psi2_theta1_corr_n  = Lidar.optics.scanner.correlations[5]
    #         psi2_theta2_corr_n  = Lidar.optics.scanner.correlations[6]
                
    #         # There is NO correlation between range and angles since the range is determined by the AOM (at least in pulsed lidars) and the angles accuracy is related to the alignment of the telescope mirrors,
    #         # to the position of the lense and also to the servos orienting the scanner
    #         psi1_rho1_corr_n    = 0
    #         psi1_rho2_corr_n    = 0
    #         psi2_rho1_corr_n    = 0
    #         psi2_rho2_corr_n    = 0
            
    #         theta1_rho1_corr_n  = 0 
    #         theta1_rho2_corr_n  = 0
    #         theta2_rho1_corr_n  = 0
    #         theta2_rho2_corr_n  = 0
            
    #         # VLOS1_psi1_corr_n   = 0
    #         # VLOS1_psi2_corr_n   = 0 
    #         # VLOS2_psi1_corr_n   = 0
    #         # VLOS2_psi2_corr_n   = 0 
            
    #         # VLOS1_theta1_corr_n = 0  
    #         # VLOS1_theta2_corr_n = 0 
    #         # VLOS2_theta1_corr_n = 0   
    #         # VLOS2_theta2_corr_n = 0 
            
    #         # VLOS1_rho1_corr_n   = 0  
    #         # VLOS1_rho2_corr_n   = 0   
    #         # VLOS2_rho1_corr_n   = 0  
    #         # VLOS2_rho2_corr_n   = 0  

            
    #         H_t1 = ((rho1*np.sin(theta1)+Hl)/Href)
    #         H_t2 = ((rho2*np.sin(theta2)+Hl)/Href)                                
    #         for ind_wind_dir in range(len(wind_direction)):  
    #             # Vlos1_noisy  = np.random.normal(Vlos1[0],u_Vlos1,N_MC)
    #             # Vlos2_noisy  = np.random.normal(Vlos2[0],u_Vlos2,N_MC) 
    #             theta1_noisy = np.random.normal((loaded_dict[0]['Uncertainty'][n_pp]['Elevation angle'])%np.radians(360),loaded_dict[0]['Uncertainty'][n_pp]['STDVs'][0],Lidar.optics.scanner.N_MC)
    #             theta2_noisy = np.random.normal((loaded_dict[1]['Uncertainty'][n_pp]['Elevation angle'])%np.radians(360),loaded_dict[1]['Uncertainty'][n_pp]['STDVs'][0],Lidar.optics.scanner.N_MC)
    #             psi1_noisy   = np.random.normal((loaded_dict[0]['Uncertainty'][n_pp]['Azimuth'])%np.radians(360),loaded_dict[0]['Uncertainty'][n_pp]['STDVs'][1],Lidar.optics.scanner.N_MC)
    #             psi2_noisy   = np.random.normal((loaded_dict[1]['Uncertainty'][n_pp]['Azimuth'])%np.radians(360),loaded_dict[1]['Uncertainty'][n_pp]['STDVs'][1],Lidar.optics.scanner.N_MC)
    #             rho1_noisy   = np.random.normal(loaded_dict[0]['Uncertainty'][n_pp]['Focus distance'][0],loaded_dict[0]['Uncertainty'][n_pp]['STDVs'][2],Lidar.optics.scanner.N_MC)
    #             rho2_noisy   = np.random.normal(loaded_dict[1]['Uncertainty'][n_pp]['Focus distance'][0],loaded_dict[1]['Uncertainty'][n_pp]['STDVs'][2],Lidar.optics.scanner.N_MC)
                
                              
    #             #%% 4) Obtain the Correlated distributions:
                
    #             # VLOS_means = [Vlos1_noisy.mean(), Vlos2_noisy.mean()]  
    #             # VLOS_stds  = [Vlos1_noisy.std(), Vlos2_noisy.std()]
                
    #             theta_means = [theta1_noisy.mean(), theta2_noisy.mean()]  
    #             theta_stds  = [theta1_noisy.std(), theta2_noisy.std()]
                
    #             psi_means = [psi1_noisy.mean(), psi2_noisy.mean()]  
    #             psi_stds  = [psi1_noisy.std(), psi2_noisy.std()]
                
    #             rho_means = [rho1_noisy.mean(), rho2_noisy.mean()]  
    #             rho_stds  = [rho1_noisy.std(), rho2_noisy.std()]
                
    #             # Covariance Matrix:
    #             cov_MAT_Vh=[[            0*theta_stds[0]**2,                     theta_stds[1]*theta_stds[0]*theta1_theta2_corr_n,   psi_stds[0]*theta_stds[0]*0                 ,   psi_stds[1]*theta_stds[0]*psi2_theta1_corr_n,   rho_stds[0]*theta_stds[0]*theta1_rho1_corr_n,                          rho_stds[1]*theta_stds[0]*theta1_rho2_corr_n               ,              0,                                                       0                              ],                          
    #                         [theta_stds[0]*theta_stds[1]*theta1_theta2_corr_n,                0*theta_stds[1]**2,                   psi_stds[0]*theta_stds[1]*psi1_theta2_corr_n,   psi_stds[1]*theta_stds[1]*0                 ,       rho_stds[0]*theta_stds[1]*theta2_rho1_corr_n,                          rho_stds[1]*theta_stds[1]*theta2_rho2_corr_n                  ,           0,                                                       0                              ],
    #                         [theta_stds[0]*psi_stds[0]*0                  ,      theta_stds[1]*psi_stds[0]*psi1_theta2_corr_n,                 0*psi_stds[0]**2,                     psi_stds[1]*psi_stds[0]*psi1_psi2_corr_n,          rho_stds[0]*psi_stds[0]*psi1_rho1_corr_n,                              rho_stds[1]*psi_stds[0]*psi1_rho2_corr_n                  ,               0,                                                       0                              ],
    #                         [theta_stds[0]*psi_stds[1]*psi2_theta1_corr_n,       theta_stds[1]*psi_stds[1]*0                 ,       psi_stds[0]*psi_stds[1]*psi1_psi2_corr_n,                 0*psi_stds[1]**2,                        rho_stds[0]*psi_stds[1]*psi2_rho1_corr_n,                              rho_stds[1]*psi_stds[1]*psi2_rho2_corr_n                      ,           0,                                                       0                              ],
    #                         [0,                                                                    0,                                                     0,                                          0,                                U_Vlos1_MC[ind_wind_dir]**2,                                           U_Vlos1_MC[ind_wind_dir]*U_Vlos2_MC[ind_wind_dir]*Vlos1_Vlos2_corr_n,     0,                                                       0                              ],
    #                         [0,                                                                    0,                                                     0,                                          0,                                U_Vlos1_MC[ind_wind_dir]*U_Vlos2_MC[ind_wind_dir]*Vlos1_Vlos2_corr_n,  U_Vlos2_MC[ind_wind_dir]**2                    ,                          0,                                                       0                              ],
    #                         [0,                                                                    0,                                                     0,                                          0,                                               0,                                                                               0                                ,              rho_stds[0]**2,                              rho_stds[0]*rho_stds[0]* rho1_rho2_corr_n   ],
    #                         [0,                                                                    0,                                                     0,                                          0,                                               0,                                                                               0                                 ,             rho_stds[0]*rho_stds[0]*rho1_rho2_corr_n    ,rho_stds[1]**2                              ]]
        
    #             # cov_MAT=[[              theta_stds[0]**2,                     theta_stds[1]*theta_stds[0]*theta1_theta2_corr_n,   psi_stds[0]*theta_stds[0]*psi1_theta1_corr_n,   psi_stds[1]*theta_stds[0]*psi2_theta1_corr_n  ],
    #             #           [theta_stds[0]*theta_stds[1]*theta1_theta2_corr_n,                 theta_stds[1]**2,                     psi_stds[0]*theta_stds[1]*psi1_theta2_corr_n,   psi_stds[1]*theta_stds[1]*psi2_theta2_corr_n],
    #             #           [theta_stds[0]*psi_stds[0]*psi1_theta1_corr_n ,      theta_stds[1]*psi_stds[0]*psi1_theta2_corr_n,                   psi_stds[0]**2,                     psi_stds[1]*psi_stds[0]*psi1_psi2_corr_n],
    #             #           [theta_stds[0]*psi_stds[1]*psi2_theta1_corr_n,       theta_stds[1]*psi_stds[1]*psi2_theta2_corr_n,       psi_stds[0]*psi_stds[1]*psi1_psi2_corr_n,                   psi_stds[1]**2]]
                  
                
    #             # Multivariate distributions:
    #             Theta1_cr2,Theta2_cr2,Psi1_cr2,Psi2_cr2,Vlos1_MC_cr2,Vlos2_MC_cr2,Rho1_cr2,Rho2_cr2 = multivariate_normal.rvs([theta_means[0],theta_means[1],psi_means[0],psi_means[1],np.mean(Vlos1[ind_wind_dir]),np.mean(Vlos2[ind_wind_dir]),rho_means[0],rho_means[1]], cov_MAT_Vh,N_MC).T
    #             pdb.set_trace()
              
    #             #%% VH Montecarlo uncertainty ##############                
    #             # Calculate the u and v wind components and their uncertainties
    #             # pdb.set_trace()
    #             # Break down large equations
    #             # u_wind,v_wind = SA.U_Vh_MC([Theta1_cr,Theta2_cr],[Psi1_cr,Psi2_cr],[Rho1_cr,Rho2_cr],wind_direction,ind_wind_dir,Href,Vref,alpha,Hl)   
    #             Vh,U_Vh,Uwind,U_u,Vwind,U_v = SA.U_Vh_MC([Theta1_cr2,Theta2_cr2],[Psi1_cr2,Psi2_cr2],[Rho1_cr2,Rho2_cr2],[Vlos1_MC_cr2,Vlos2_MC_cr2],loaded_dict,wind_direction,ind_wind_dir,Href,Vref,alpha,Hl)   
                
    #             # ucomponent estimation        
    #             Uwind_MC.append(Uwind)
    #             # Uncertainty as standard deviation (k=1) in the u wind velocity component estimation
    #             Uncertainty_U.append(U_u)
                
    #             # v component estimation        
    #             Vwind_MC.append(Vwind)
    #             # Uncertainty as standard deviation (k=1) in the v wind velocity component estimation
    #             Uncertainty_V.append(U_v)
        
    #             # Horizontal velocity estimation
    #             Vh_MC=Vh
    #             # Uncertainty as standard deviation (k=1) in the horizontal velocity estimation
    #             Uncertainty_Vh_MC.append(U_Vh)
    #             # pdb.set_trace()
               
            
    #             #%% VH GUM uncertainty#####################
    #             U = [loaded_dict[0]['Uncertainty'][n_pp]['STDVs'][0],loaded_dict[1]['Uncertainty'][n_pp]['STDVs'][0],loaded_dict[0]['Uncertainty'][n_pp]['STDVs'][1],loaded_dict[1]['Uncertainty'][n_pp]['STDVs'][1],loaded_dict[0]['Uncertainty'][n_pp]['STDVs'][2],loaded_dict[1]['Uncertainty'][n_pp]['STDVs'][2]]
                
               
    #             # Calculate uncertainty for the GUM approach # LoveU Pep!!
    #             # pdb.set_trace()
    #             Uncertainty_Vh_GUM_F= SA.U_Vh_GUM([loaded_dict[0]['Uncertainty'][n_pp]['Elevation angle'],loaded_dict[1]['Uncertainty'][n_pp]['Elevation angle']],[loaded_dict[0]['Uncertainty'][n_pp]['Azimuth'][0],loaded_dict[1]['Uncertainty'][n_pp]['Azimuth'][0]],[loaded_dict[0]['Uncertainty'][n_pp]['Focus distance'],loaded_dict[1]['Uncertainty'][n_pp]['Focus distance']],wind_direction,ind_wind_dir,Href,Vref,alpha,Hl,U,H_t1,H_t2)   
        
    #             Uncertainty_Vh_GUM.append(Uncertainty_Vh_GUM_F)
    #             # Uncertainty_u_GUM.append(U_comp)
            
           
    #         #%% Storing data
            # Final_Output_Vh_Uncertainty = {'Vh Uncertainty GUM [m/s]':loaded_dict[0]['Uncertainty'][0]['Uncertainty Vh GUM'],'Vh Uncertainty MC [m/s]':loaded_dict[0]['Uncertainty'][0]['Uncertainty Vh MCM'],'wind direction':loaded_dict[0]['Uncertainty'][0]['wind direction']}
    #         Vh_U.append(Final_Output_Vh_Uncertainty['Vh Uncertainty GUM [m/s]'][0])
        
    #         #%% Plotting
    #         # pdb.set_trace()
            # QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_Vh_Uncertainty,False,False,False,False,Qlunc_yaml_inputs['Flags']['Horizontal Velocity Uncertainty'],False)  #Qlunc_yaml_inputs['Flags']['Scanning Pattern']
        # return(Vh_U)