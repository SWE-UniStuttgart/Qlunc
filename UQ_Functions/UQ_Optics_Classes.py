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
from Main.Qlunc_Classes import lidar_coor
#%% SCANNER:
def UQ_Scanner(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs,DataFrame):
    """.
    
    Scanner uncertainty estimation. Location: ./UQ_Functions/UQ_Optics_Classes.py   
    Parameters
    ----------
    
    * Lidar
        Dictionary containing lidar info
        
    * Atmospheric_Scenario
        Atmospheric data. Integer or Time series

    * cts
        Physical constants

    * Qlunc_yaml_inputs
        Lidar parameters data        

    Returns
    -------
    
    Dictionary containing information about uncertainties in:
        1) Wind direction 
        2) LOS wind velocity
        3) Horizontal wind velocity
        4) Wind velocity 3D vector
    
    """
    
    U_Vh_GUM_T,U_Vh_MCM_mean,U_Wind_direction_MCM,U_Wind_direction_GUM      = [],[],[],[]
    U_VLOS_T_MC_rho_T,U_VLOS_T_GUM_rho_T                                    = [],[]
    U_VLOS_T_MC_theta_T,U_VLOS_T_GUM_theta_T                                = [],[]
    U_VLOS_T_MC_psi_T,U_VLOS_T_GUM_psi_T                                    = [],[]
    Scan_unc                                                                = []
    SensCoeffVh                                                             = []
    wind_direction_TEST                                                     = []
    M                                                                       = [] # Conditional number
    U_Vlos                                                                  = {'V1_MCM'   : [],'V2_MCM'   : [],'V3_MCM'   : [],'V1_GUM'  : [],'V2_GUM'  : [],'V3_GUM'  : []}
    Vh_                                                                     = {'V1_MCM'   : [],'V2_MCM'   : [],'V3_MCM'   : [],'V1_GUM'  : [],'V2_GUM'  : [],'V3_GUM'  : [],'V1_MCM_mean'  : [],'V2_MCM_mean'  : [],'V3_MCM_mean'  : []}
    Correlation_coeff                                                       = {'V12_MCM'  : [],'V13_MCM'  : [],'V23_MCM'  : [],'V12_GUM' : [],'V13_GUM' : [],'V23_GUM' : []}
    SensCoeff_Vlos                                                          = {'V1_theta' : [],'V2_theta' : [],'V3_theta' : [],'V1_psi'  : [],'V2_psi'  : [],'V3_psi'  : [],'V1_rho' : [],'V2_rho' : [],'V3_rho' : [],'W1' : [],'W2' : [],'W1W2' : [],'W4' : [],'W5' : [],'W6' : []}
    WindDirection_                                                           = {'V1_MCM'   : [],'V2_MCM'   : [],'V3_MCM'   : [],'V1_GUM'  : [],'V2_GUM'  : [],'V3_GUM':[],'V1_MCM_mean'  : [],'V2_MCM_mean'  : [],'V3_MCM_mean'  : []}
     
    Href  = Qlunc_yaml_inputs['Components']['Scanner']['Href'],
    V_ref  = Atmospheric_Scenario.Vref
    alpha = Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']    
    Hg    = Qlunc_yaml_inputs['Atmospheric_inputs']['Height ground']
    Hl    = [Lidar.optics.scanner.origin[0][2],Lidar.optics.scanner.origin[1][2]]    
    N_MC  = Lidar.optics.scanner.N_MC
    
    
    # # Implement error in deployment of the tripod as a rotation over yaw, pitch and roll
    # stdv_yaw    = np.array(np.radians(Lidar.lidar_inputs.yaw_error_dep))
    # stdv_pitch  = np.array(np.radians(Lidar.lidar_inputs.pitch_error_dep))
    # stdv_roll   = np.array(np.radians(Lidar.lidar_inputs.roll_error_dep))
    
    if Lidar.optics.scanner.pattern=='lissajous':
        x_out,y_out,z_out = SP.lissajous_pattern(Lidar,Lidar.optics.scanner.lissajous_param[0],Lidar.optics.scanner.lissajous_param[1],Lidar.optics.scanner.lissajous_param[2],Lidar.optics.scanner.lissajous_param[3],Lidar.optics.scanner.lissajous_param[4])
        L                 = len(x_out)
        wind_direction    = np.radians(np.linspace(Atmospheric_Scenario.wind_direction[0],Atmospheric_Scenario.wind_direction[1],1))
    
    elif Lidar.optics.scanner.pattern=='vertical plane':  
        x_out,y_out,z_out = SP.Verticalplane_pattern(Lidar)
        L                 = len(x_out)
        wind_direction = np.radians(np.linspace(Atmospheric_Scenario.wind_direction[0],Atmospheric_Scenario.wind_direction[1],1))        
    elif Lidar.optics.scanner.pattern=='horizontal plane':        
        x_out,y_out,z_out = SP.Horizontalplane_pattern(Lidar)
        L                 = len(z_out)
        wind_direction    = np.radians(np.linspace(Atmospheric_Scenario.wind_direction[0],Atmospheric_Scenario.wind_direction[1],1))        
    
    else: # One point for all wind directions stated in YAML file
        L=len(Lidar.optics.scanner.focus_dist)
        wind_direction = np.radians(np.linspace(Atmospheric_Scenario.wind_direction[0],Atmospheric_Scenario.wind_direction[1],Atmospheric_Scenario.wind_direction[2]))
        x_out,y_out,z_out = 0,0,0
    

    # Loop for the points in the pattern and the alpha exponents    
    for ind_alpha in range(len(alpha)):
        alpha = Atmospheric_Scenario.PL_exp[ind_alpha]    
        for meas_param in range(L):
            # LOVE U MAMA!!        
         
            #%% 2) Range and measuring angles are calculated based on the position of the lidars and the measuring points            
            
            # Measurement point(s) in cartesian coordinates
            if Lidar.optics.scanner.pattern=='lissajous' or Lidar.optics.scanner.pattern=='horizontal plane' or Lidar.optics.scanner.pattern=='vertical plane':
                x = np.array([x_out[meas_param]])
                y = np.array([y_out[meas_param]])
                z = np.array([z_out[meas_param]])               
            else:
                x,y,z = SA.sph2cart([Lidar.optics.scanner.focus_dist[meas_param]],[np.radians(Lidar.optics.scanner.cone_angle[meas_param])],[np.radians(Lidar.optics.scanner.azimuth[meas_param])])
    
            
            # Store lidar positionning   
            lidars={}       
            # pdb.set_trace()
            for n_lidars in range(len(Lidar.optics.scanner.origin)):
                lidars['Lidar{}_Rectangular'.format(n_lidars)] = {'LidarPosX': Lidar.optics.scanner.origin[n_lidars][0],
                                                                 'LidarPosY' : Lidar.optics.scanner.origin[n_lidars][1],
                                                                 'LidarPosZ' : Lidar.optics.scanner.origin[n_lidars][2],
                                                                 'x'         : (lidar_coor.vector_pos(x,y,z,x_Lidar=Lidar.optics.scanner.origin[n_lidars][0],y_Lidar=Lidar.optics.scanner.origin[n_lidars][1],z_Lidar=Lidar.optics.scanner.origin[n_lidars][2])[1]),
                                                                 'y'         : (lidar_coor.vector_pos(x,y,z,x_Lidar=Lidar.optics.scanner.origin[n_lidars][0],y_Lidar=Lidar.optics.scanner.origin[n_lidars][1],z_Lidar=Lidar.optics.scanner.origin[n_lidars][2])[2]),
                                                                 'z'         : (lidar_coor.vector_pos(x,y,z,x_Lidar=Lidar.optics.scanner.origin[n_lidars][0],y_Lidar=Lidar.optics.scanner.origin[n_lidars][1],z_Lidar=Lidar.optics.scanner.origin[n_lidars][2])[3])}
                
                lidars['Lidar{}_Spherical'.format(n_lidars)]   = {'rho'      : np.round((lidar_coor.Cart2Sph(lidars['Lidar{}_Rectangular'.format(n_lidars)]['x'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['y'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['z']))[1],10),
                                                                  'theta'    : np.round((lidar_coor.Cart2Sph(lidars['Lidar{}_Rectangular'.format(n_lidars)]['x'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['y'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['z']))[2]%np.radians(360),10),
                                                                  'psi'      : np.round((lidar_coor.Cart2Sph(lidars['Lidar{}_Rectangular'.format(n_lidars)]['x'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['y'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['z']))[3]%np.radians(360),10)}
            
            
            # Add coordinates to lidars dict to calculate uncertainties using a pattern instead of a single point
            lidars['Coord_Out'] = np.array([x_out,y_out,z_out])

            
            #%% 3) Wind velocity uncertainy estimation
 
            # 3.1) Vlos and Vh Uncertainties - MCM method
            Correlation_coeff_MCM, U_Vlos_MCM, Mult_param, U_Vh_MCM,Vh_MCM,Vh_MCM_mean = SA.MCM_Vh_lidar_uncertainty(Lidar,Atmospheric_Scenario,wind_direction,alpha,lidars,DataFrame)        
  
            # 3.2) Vlos and Vh Uncertainties - GUM method
            Correlation_coeff_GUM, U_Vlos_GUM, Vlos_GUM, SensitivityCoeff_VLOS_GUM = SA.GUM_Vlos_lidar_uncertainty(Lidar,Atmospheric_Scenario,wind_direction,alpha,lidars,DataFrame)
            U_Vh_GUM, Sensitivity_Coefficients_Vh,u,v,w,Vh_GUM                     = SA.GUM_Vh_lidar_uncertainty(Lidar,Atmospheric_Scenario,Correlation_coeff_GUM,wind_direction,lidars,Vlos_GUM,U_Vlos_GUM,DataFrame)
            
            #%% 4) Wind direction uncertainty estimation
            
            # 4.1) MCM
            U_WindDir_MCM,WindDir_MCM,WindDirect_mean = SA.U_WindDir_MC(Lidar,wind_direction,Mult_param,DataFrame)
            
            # 4.2) GUM
            U_Wind_direction_GUM0,dWinDir_Vlos1,dWinDir_Vlos2,dWinDir_Vlos12,dWinDir_Vlos4T,dWinDir_Vlos5T,dWinDir_Vlos6T, WindDir_GUM = SA.U_WindDir_GUM(Lidar,Atmospheric_Scenario,Correlation_coeff_GUM,wind_direction,lidars,Vlos_GUM,U_Vlos_GUM,u,v,w,DataFrame) 
                        
            #%% 5) Method for uncertainty when varying theta, psi 'OR' rho   
            U_VLOS_T_MC_rho,U_VLOS_T_GUM_rho,rho_TESTr,theta_TESTr,psi_TESTr      =  SA.VLOS_param(Lidar,np.linspace(10,5000,600),lidars['Lidar0_Spherical']['theta'],lidars['Lidar0_Spherical']['psi'],0,0,Lidar.optics.scanner.stdv_focus_dist[0][0],Lidar.optics.scanner.N_MC,Hl[0],V_ref,Href,alpha,wind_direction_TEST,0,DataFrame)
            U_VLOS_T_MC_theta,U_VLOS_T_GUM_theta,rho_TESTt,theta_TESTt,psi_TESTt  =  SA.VLOS_param(Lidar,lidars['Lidar0_Spherical']['rho'],np.radians(np.linspace(0,90,200)),lidars['Lidar0_Spherical']['psi'],np.radians(Lidar.optics.scanner.stdv_cone_angle[0][0]),0,0,Lidar.optics.scanner.N_MC,Hl[0],V_ref,Href,alpha,wind_direction_TEST,0,DataFrame)    
            U_VLOS_T_MC_psi,U_VLOS_T_GUM_psi,rho_TESTp,theta_TESTp,psi_TESTp      =  SA.VLOS_param(Lidar,lidars['Lidar0_Spherical']['rho'],lidars['Lidar0_Spherical']['theta'],np.radians(np.linspace(0,359,200)),0,np.radians(Lidar.optics.scanner.stdv_azimuth[0][0]),0,Lidar.optics.scanner.N_MC,Hl[0],V_ref,Href,alpha,wind_direction_TEST,0,DataFrame)
            # pdb.set_trace()
            
            #%% Conditional M number
            
            M.append(SA.condM(Lidar,lidars))
            
            
            # pdb.set_trace()
            #%% Store data 
            
            # Wind velocity wind direction and sensitivity coefficients
            for i in range(len(Lidar.optics.scanner.origin)):
                U_Vlos['V{}_MCM'.format(i+1)].append(np.concatenate(U_Vlos_MCM['V{}'.format(i+1)],axis=0))
                U_Vlos['V{}_GUM'.format(i+1)].append(np.concatenate(U_Vlos_GUM['V{}'.format(i+1)],axis=0))
                
                SensCoeff_Vlos['V{}_theta'.format(i+1)].append(SensitivityCoeff_VLOS_GUM['V{}_theta'.format(i+1)])
                SensCoeff_Vlos['V{}_psi'.format(i+1)].append(SensitivityCoeff_VLOS_GUM['V{}_psi'.format(i+1)])
                SensCoeff_Vlos['V{}_rho'.format(i+1)].append(SensitivityCoeff_VLOS_GUM['V{}_rho'.format(i+1)])
            
            
            # Store Vh
            Vh_['V{}_MCM'.format(ind_alpha+1)].append(Vh_MCM)
            Vh_['V{}_GUM'.format(ind_alpha+1)].append(Vh_GUM)
            Vh_['V{}_MCM_mean'.format(ind_alpha+1)].append(Vh_MCM_mean)
            U_Vh_MCM_mean.append(U_Vh_MCM)
            U_Vh_GUM_T.append(U_Vh_GUM) 

            
            # Store wind direction
            WindDirection_['V{}_MCM'.format(ind_alpha+1)].append(WindDir_MCM)
            WindDirection_['V{}_GUM'.format(ind_alpha+1)].append(WindDir_GUM)
            WindDirection_['V{}_MCM_mean'.format(ind_alpha+1)].append(WindDirect_mean)
 
            U_Wind_direction_MCM.append(U_WindDir_MCM)
            U_Wind_direction_GUM.append(U_Wind_direction_GUM0)


            # Sensitivity coefficients  
            SensCoeffVh.append(Sensitivity_Coefficients_Vh)
            SensCoeff_Vlos['W1'].append(dWinDir_Vlos1) 
            SensCoeff_Vlos['W2'].append(dWinDir_Vlos2)
            SensCoeff_Vlos['W1W2'].append(dWinDir_Vlos12)
            SensCoeff_Vlos['W4'].append(dWinDir_Vlos4T)
            SensCoeff_Vlos['W5'].append(dWinDir_Vlos5T)
            SensCoeff_Vlos['W6'].append(dWinDir_Vlos6T)            
            #Correlations
            Correlation_coeff['V12_GUM'].append(Correlation_coeff_GUM['V1'])
            Correlation_coeff['V13_GUM'].append(Correlation_coeff_GUM['V2'])
            Correlation_coeff['V23_GUM'].append(Correlation_coeff_GUM['V3'])
            Correlation_coeff['V12_MCM'].append(Correlation_coeff_MCM['V12'])
            Correlation_coeff['V13_MCM'].append(Correlation_coeff_MCM['V13'])
            Correlation_coeff['V23_MCM'].append(Correlation_coeff_MCM['V23'])

            # Sensitivity analysis of the input quantities
            U_VLOS_T_MC_rho_T.append(U_VLOS_T_MC_rho)
            U_VLOS_T_GUM_rho_T.append(U_VLOS_T_GUM_rho[0])                       
            U_VLOS_T_MC_theta_T.append(U_VLOS_T_MC_theta)
            U_VLOS_T_GUM_theta_T.append(U_VLOS_T_GUM_theta[0])
            U_VLOS_T_MC_psi_T.append(U_VLOS_T_MC_psi)
            U_VLOS_T_GUM_psi_T.append(U_VLOS_T_GUM_psi[0])

            # Add test coordinates to lidars dict
            lidars['Coord_Test']={'TESTr':np.array([rho_TESTr]),'TESTt':np.array([theta_TESTt]),'TESTp':np.array([psi_TESTp])}
        # pdb.set_trace()
    
    
    #%% CI calculations
    k = Qlunc_yaml_inputs['Components']['Scanner']['k']
    wl_alpha=1
    
    CI_VLOS_L_GUM, CI_VLOS_H_GUM, CI_VLOS_L_MC, CI_VLOS_H_MC, CI_Vh_L_GUM, CI_Vh_H_GUM,CI_Vh_L_MC, CI_Vh_H_MC,CI_L_GUM_WindDir,CI_H_GUM_WindDir,CI_L_MC_WindDir,CI_H_MC_WindDir,prob = SA.CI(wl_alpha,k,U_Vlos_GUM,U_Vlos_MCM,Vlos_GUM,Mult_param, U_Vh_GUM_T, U_Vh_MCM_mean, Vh_,U_Wind_direction_GUM,U_Wind_direction_MCM,WindDirection_)
    
    
    # MCM validation
    
    #Tolerance and lower/upper limits calculated as in JCGM 101:2008 Section 8
    delta_V = .5*1e-2
    delta_D = .5*1e-1    
    
    dlow_Vh       = [ abs(Vh_['V{}_GUM'.format(wl_alpha)][0][ilow]  - U_Vh_GUM_T[wl_alpha-1][ilow]  - CI_Vh_L_MC[ilow])  for ilow  in range( len( Vh_['V1_GUM'][0] ) ) ]
    dhigh_Vh      = [ abs(Vh_['V{}_GUM'.format(wl_alpha)][0][ihigh] + U_Vh_GUM_T[wl_alpha-1][ihigh] - CI_Vh_H_MC[ihigh]) for ihigh in range( len( Vh_['V1_GUM'][0] ) ) ]    

    dlow_WindDir  = [ abs((WindDirection_['V{}_GUM'.format(wl_alpha)][0][ilow2])  - U_Wind_direction_GUM[wl_alpha-1][ilow2]  - CI_L_MC_WindDir[ilow2])  for ilow2  in range( len( WindDirection_['V1_GUM'][0] ) ) ]
    dhigh_WindDir = [ abs((WindDirection_['V{}_GUM'.format(wl_alpha)][0][ihigh2]) + U_Wind_direction_GUM[wl_alpha-1][ihigh2] - CI_H_MC_WindDir[ihigh2]) for ihigh2 in range( len( WindDirection_['V1_GUM'][0] ) ) ]    

    #%% Store Data
    VLOS_Unc    =  {'VLOS1 Uncertainty MC [m/s]':      U_Vlos['V1_MCM'],      'VLOS1 Uncertainty GUM [m/s]':      U_Vlos['V1_GUM'],
                    'VLOS2 Uncertainty MC [m/s]':      U_Vlos['V2_MCM'],      'VLOS2 Uncertainty GUM [m/s]':      U_Vlos['V2_GUM'],
                    'VLOS3 Uncertainty MC [m/s]':      U_Vlos['V3_MCM'],      'VLOS3 Uncertainty GUM [m/s]':      U_Vlos['V3_GUM'],
                    'VLOS Uncertainty MC rho [m/s]':   U_VLOS_T_MC_rho_T,     'VLOS Uncertainty GUM rho [m/s]':   U_VLOS_T_GUM_rho_T,
                    'VLOS Uncertainty MC psi [m/s]':   U_VLOS_T_MC_psi_T,     'VLOS Uncertainty GUM psi [m/s]':   U_VLOS_T_GUM_psi_T,
                    'VLOS Uncertainty MC theta [m/s]': U_VLOS_T_MC_theta_T,   'VLOS Uncertainty GUM theta [m/s]': U_VLOS_T_GUM_theta_T}
        
    Vh_Unc      =  {'Uncertainty Vh MCM':             U_Vh_MCM_mean,            'Uncertainty Vh GUM':             U_Vh_GUM_T}   
    WinDir_Unc  =  {'Uncertainty wind direction GUM': U_Wind_direction_GUM,  'Uncertainty wind direction MCM': U_Wind_direction_MCM}
    
    
    Final_Output_UQ_Scanner = {'lidars'          : lidars,
                               'wind direction'  : wind_direction,
                               'STDVs'           : [Lidar.optics.scanner.stdv_cone_angle,Lidar.optics.scanner.stdv_azimuth,Lidar.optics.scanner.stdv_focus_dist],
                               'VLOS Unc [m/s]'  : VLOS_Unc,
                               'Vh Unc [m/s]'    : Vh_Unc,
                               'WinDir Unc [°]'  : WinDir_Unc,
                               'Sens coeff Vh'   : SensCoeffVh,
                               'Sens coeff Vlos' : SensCoeff_Vlos,
                               'Correlations'    : Correlation_coeff,
                               'Mult param'      : Mult_param,
                               'Vlos_GUM'        : Vlos_GUM,
                               'Vh'              : Vh_,  
                               'Wind direction'  : WindDirection_,                           
                               'CI'              : [CI_VLOS_L_GUM, CI_VLOS_H_GUM, CI_VLOS_L_MC, CI_VLOS_H_MC, CI_Vh_L_GUM, CI_Vh_H_GUM,CI_Vh_L_MC, CI_Vh_H_MC,CI_L_GUM_WindDir,CI_H_GUM_WindDir,CI_L_MC_WindDir,CI_H_MC_WindDir,prob],
                               'Tolerance'       : [delta_V,delta_D, dlow_Vh, dhigh_Vh,dlow_WindDir,dhigh_WindDir,wl_alpha],
                               'Conditional M'   : M}
    # Lidar.lidar_inputs.dataframe['Scanner'] = {'Focus distance':Final_Output_UQ_Scanner['lidars'][0],'Elevation angle':Final_Output_UQ_Scanner['Elevation angle'][0],'Azimuth':Final_Output_UQ_Scanner['Azimuth'][0]}
    DataFrame['Uncertainty Scanner']=Final_Output_UQ_Scanner    
    # pdb.set_trace()
   
    #%% 7) Plotting data
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_Scanner,False,Qlunc_yaml_inputs['Flags']['Wind direction uncertainty'],Qlunc_yaml_inputs['Flags']['Wind velocity uncertainty'],Qlunc_yaml_inputs['Flags']['Line of sight velocity uncertainty'],Qlunc_yaml_inputs['Flags']['PDFs'],Qlunc_yaml_inputs['Flags']['Coverage interval'])  
    
    return DataFrame


#%% Sum of uncertainties in `optics` module: 
def sum_unc_optics(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,DataFrame):
    List_Unc_optics = []
    # Each try/except evaluates wether the component is included in the module and therefore in the calculations   
    # Scanner
    if Lidar.optics.scanner != None:
       
                   
        DataFrame=Lidar.optics.scanner.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,DataFrame)
            
         
    else:
        print (colored('You didn´t include a head scanner in the lidar.','cyan', attrs=['bold']))       
    return DataFrame

