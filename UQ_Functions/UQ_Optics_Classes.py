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
    X,Y,Z,R=[],[],[],[]
    Scan_unc=[]
    rho_noisy0,theta_noisy0,psi_noisy0,rho_noisy,theta_noisy1,theta_noisy2,psi_noisy,rho_noisy1,rho_noisy2, theta_noisy,psi_noisy1,psi_noisy2,wind_direction_TEST ,wind_tilt_TEST   = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
    # pdb.set_trace() 
    Href  = Qlunc_yaml_inputs['Components']['Scanner']['Href'],
    Vref  = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']
    alpha = Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']    
    Hg    = Qlunc_yaml_inputs['Atmospheric_inputs']['Height ground']
    Hl    = [Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0][2],Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1][2]]    
    N_MC = Lidar.optics.scanner.N_MC
    # R: Implement error in deployment of the tripod as a rotation over yaw, pitch and roll
    stdv_yaw    = np.array(np.radians(Lidar.lidar_inputs.yaw_error_dep))
    stdv_pitch  = np.array(np.radians(Lidar.lidar_inputs.pitch_error_dep))
    stdv_roll   = np.array(np.radians(Lidar.lidar_inputs.roll_error_dep))
    
    if Lidar.optics.scanner.pattern=='lissajous':
        xx,yy,zz=SP.lissajous_pattern(Lidar,Lidar.optics.scanner.lissajous_param[0],Lidar.optics.scanner.lissajous_param[1],Lidar.optics.scanner.lissajous_param[2],Lidar.optics.scanner.lissajous_param[3],Lidar.optics.scanner.lissajous_param[4])
        L=len(xx)
    else:
        L=len(Lidar.optics.scanner.focus_dist)

    # Loop for the points in the pattern
    for meas_param in range(L):
        # LOVE U MAMA!!
        
        #%% 1) Rho, theta and psi values OF THE MEASURING POINT   
        rho0           = [Lidar.optics.scanner.focus_dist[meas_param]]  
        theta0         = [np.radians(Lidar.optics.scanner.cone_angle[meas_param])]  # %(360) converts negative angles to [0,360] positive angles
        psi0           = [np.radians(Lidar.optics.scanner.azimuth[meas_param])] 
        wind_direction = np.radians(np.linspace(Atmospheric_Scenario.wind_direction[0],Atmospheric_Scenario.wind_direction[1],360))
        
        
        #%% 2) Lidars' position and measuring angles. The measuring angles are calculated based on the position of the lidars and the measuring points
        
        # Measurement point in cartesian coordinates before applying lidar position
        if Lidar.optics.scanner.pattern=='lissajous':
            x=np.array([xx[meas_param]])
            y=np.array([yy[meas_param]])
            z=np.array([zz[meas_param]])
            
        else:
            x,y,z=SA.sph2cart(rho0,theta0,psi0)
        

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
        
        # Store lidar positionning   
        lidars={}       
        # pdb.set_trace()
        for n_lidars in range(len(Qlunc_yaml_inputs['Components']['Scanner']['Origin'])):
            lidars['Lidar{}_Rectangular'.format(n_lidars)]={'x':(lidar_coor.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][n_lidars][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][n_lidars][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][n_lidars][2])[1]),
                                         'y':(lidar_coor.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][n_lidars][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][n_lidars][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][n_lidars][2])[2]),
                                         'z':(lidar_coor.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][n_lidars][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][n_lidars][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][n_lidars][2])[3])}
            
            lidars['Lidar{}_Spherical'.format(n_lidars)]={'rho':   np.round((lidar_coor.Cart2Sph(lidars['Lidar{}_Rectangular'.format(n_lidars)]['x'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['y'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['z']))[1],4),
                                                          'theta': np.round((lidar_coor.Cart2Sph(lidars['Lidar{}_Rectangular'.format(n_lidars)]['x'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['y'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['z']))[2],4),
                                                          'psi':   np.round((lidar_coor.Cart2Sph(lidars['Lidar{}_Rectangular'.format(n_lidars)]['x'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['y'],lidars['Lidar{}_Rectangular'.format(n_lidars)]['z']))[3],4)}
        
             
        # Rho, theta and psi lidar inputs and their uncertainties
        
        # Lidar 1
        theta1,U_theta1 = lidars['Lidar0_Spherical']['theta']%np.radians(360) ,np.radians(Lidar.optics.scanner.stdv_cone_angle[0][0])
        psi1  ,U_psi1   = lidars['Lidar0_Spherical']['psi']%np.radians(360)   ,np.radians(Lidar.optics.scanner.stdv_azimuth[0][0])
        rho1  ,U_rho1   = lidars['Lidar0_Spherical']['rho']                 ,Lidar.optics.scanner.stdv_focus_dist [0][0]
        # Lidar 2 
        theta2,U_theta2 = lidars['Lidar1_Spherical']['theta']%np.radians(360) ,np.radians(Lidar.optics.scanner.stdv_cone_angle[1][0])
        psi2  ,U_psi2   = lidars['Lidar1_Spherical']['psi']%np.radians(360)   ,np.radians(Lidar.optics.scanner.stdv_azimuth[1][0])
        rho2  ,U_rho2   = lidars['Lidar1_Spherical']['rho']                 ,Lidar.optics.scanner.stdv_focus_dist [1][0]
        

        #%% Uncertainty in the probe volume (This call needs to be changed!)
        # Probe_param = Lidar.probe_volume.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,lidars)
        # Lidar.lidar_inputs.dataframe['Probe Volume'] = Probe_param
        
        #%% 3) State the correlations
        psi1_psi2_corr     = Lidar.optics.scanner.correlations[0]  # correlation between psi1 and psi2
        theta1_theta2_corr = Lidar.optics.scanner.correlations[1] # correlation Theta1 and Theta2
        rho1_rho2_corr     = Lidar.optics.scanner.correlations[2]  # correlation Rho1 and Rho2
        
        # Cross correlations:
        psi1_theta1_corr  = Lidar.optics.scanner.correlations[3]
        psi1_theta2_corr  = Lidar.optics.scanner.correlations[4]
        psi2_theta1_corr  = Lidar.optics.scanner.correlations[5]
        psi2_theta2_corr  = Lidar.optics.scanner.correlations[6]
        Vlos1_Vlos2_corr  = Lidar.optics.scanner.correlations[7]
        
        # There is NO correlation between range and angles since the range is determined by the AOM (at least in pulsed lidars) and the angles accuracy is related to the alignment of the telescope mirrors,
        # to the position of the lense and also to the servos orienting the scanner
        psi1_rho1_corr    = 0
        psi1_rho2_corr    = 0
        psi2_rho1_corr    = 0
        psi2_rho2_corr    = 0
        
        theta1_rho1_corr  = 0 
        theta1_rho2_corr  = 0
        theta2_rho1_corr  = 0
        theta2_rho2_corr  = 0
       
        
        #%% 4) Wind velocity uncertainy estimation

        # 4.1) Store necessary parameters
        nec_par=[wind_direction,Href,Vref,alpha,Hg,Hl,N_MC,theta1[0],U_theta1,psi1 [0] ,U_psi1,rho1[0],U_rho1,theta2[0],U_theta2,psi2[0],U_psi2,
                 rho2[0],U_rho2,psi1_psi2_corr,theta1_theta2_corr,rho1_rho2_corr,psi1_theta1_corr,psi1_theta2_corr,psi2_theta1_corr,psi2_theta2_corr,
                 Vlos1_Vlos2_corr,psi1_rho1_corr,psi1_rho2_corr,psi2_rho1_corr,psi2_rho2_corr,theta1_rho1_corr,theta1_rho2_corr,theta2_rho1_corr,theta2_rho2_corr]
        # Uncertainty u,v
        U_Vlos1_MCM,U_Vlos2_MCM,u,v,U_u_MC,U_v_MC,Mult_param = SA.MCM_uv_lidar_uncertainty(*nec_par)

        # pdb.set_trace()
        # 4.2) Uncertainty MCM for Vh
        U_Vh_MCM = SA.MCM_Vh_lidar_uncertainty(wind_direction,u,v,*Mult_param)
        
        
        # 4.3) Uncertainty GUM method
        Vlos1_GUM,Vlos2_GUM,U_Vlos1_GUM,U_Vlos2_GUM,u_GUM,v_GUM,U_u_GUM,U_v_GUM,CorrCoef_uv= SA.GUM_uv_lidar_uncertainty(*nec_par)
 
        # 4.4) Vh Uncertainty GUM method
        U_Vh_GUM=SA.GUM_Vh_lidar_uncertainty(u_GUM,v_GUM,U_u_GUM,U_v_GUM,CorrCoef_uv,Vlos1_GUM,Vlos2_GUM,U_Vlos1_GUM,U_Vlos2_GUM,*nec_par)
        
        
        #%% 5) Method for uncertainty when varying theta, psi OR rho   
        
        U_VLOS_T_MC_rho,U_VLOS_THomo_MC_rho,U_VLOS_T_GUM_rho,U_VLOS_THomo_GUM_rho,rho_TESTr,thata_TESTr,psi_TESTr          =  SA.VLOS_param(np.linspace(1000,5000,100),theta1,psi1,0,0,U_rho1,Lidar.optics.scanner.N_MC,U_Vlos1_GUM,Hl[0],Vref,Href,alpha,wind_direction_TEST,0,[0,0,0])
        U_VLOS_T_MC_theta,U_VLOS_THomo_MC_theta,U_VLOS_T_GUM_theta,U_VLOS_THomo_GUM_theta,rho_TESTt,theta_TESTt,psi_TESTt  =  SA.VLOS_param(rho1,np.radians(np.linspace(1,89,100)),psi1,U_theta1,0,0,Lidar.optics.scanner.N_MC,U_Vlos1_GUM,Hl[0],Vref,Href,alpha,wind_direction_TEST,0,[0,0,0])    
        U_VLOS_T_MC_psi,U_VLOS_THomo_MC_psi,U_VLOS_T_GUM_psi,U_VLOS_THomo_GUM_psi,rho_TESTp,theta_TESTp,psi_TESTp          =  SA.VLOS_param(rho1,theta1,np.radians(np.linspace(-90,90,100)),0,U_psi1,0,Lidar.optics.scanner.N_MC,U_Vlos1_GUM,Hl[0],Vref,Href,alpha,wind_direction_TEST,0,[0,0,0])
                
        #%% Wind direction uncertainty estimation
        U_Wind_direction_MC=SA.U_WindDir_MC(wind_direction,u,v,Mult_param)
        
        U_Wind_direction_GUM=SA.U_WindDir_GUM(Vlos1_GUM,Vlos2_GUM,U_Vlos1_GUM,U_Vlos2_GUM,*nec_par)
        
        
        
        # plt.plot(U_Wind_direction_MC,'*m')
        # plt.plot(U_Wind_direction_GUM)
        
        # pdb.set_trace()
        
        
        
        
        
        
        
        #%% 6) Storing data
        Final_Output_UQ_Scanner                 = {'Uncertainty u wind component MCM':U_u_MC,'Uncertainty v wind component MCM':U_v_MC,'Uncertainty u wind component GUM':U_u_GUM,'Uncertainty v wind component GUM':U_v_GUM,'Uncertainty wind direction MCM':U_Wind_direction_MC,'Uncertainty wind direction GUM':U_Wind_direction_GUM,'VLOS1 Uncertainty MC [m/s]':U_Vlos1_MCM,'VLOS1 Uncertainty GUM [m/s]':U_Vlos1_GUM,'Uncertainty Vh GUM':U_Vh_GUM,'Uncertainty Vh MCM':U_Vh_MCM,
                                                   'Vr Uncertainty homo MC rho [m/s]':U_VLOS_THomo_MC_rho,'Vr Uncertainty homo GUM rho [m/s]':U_VLOS_THomo_GUM_rho,'Vr Uncertainty MC rho [m/s]':U_VLOS_T_MC_rho,'Vr Uncertainty GUM rho [m/s]':U_VLOS_T_GUM_rho,
                                                   'Vr Uncertainty homo MC theta [m/s]':U_VLOS_THomo_MC_theta,'Vr Uncertainty homo GUM theta [m/s]':U_VLOS_THomo_GUM_theta,'Vr Uncertainty MC theta [m/s]':U_VLOS_T_MC_theta,'Vr Uncertainty GUM theta [m/s]':U_VLOS_T_GUM_theta,
                                                   'Vr Uncertainty homo MC psi [m/s]':U_VLOS_THomo_MC_psi,'Vr Uncertainty homo GUM psi [m/s]':U_VLOS_THomo_GUM_psi,'Vr Uncertainty MC psi [m/s]':U_VLOS_T_MC_psi,'Vr Uncertainty GUM psi [m/s]':U_VLOS_T_GUM_psi,
                                                   'x':x,'y':y,'z':z,'rho':rho_TESTr,'theta':theta_TESTt,'psi':psi_TESTp,'wind direction':wind_direction,'Focus distance':rho1,'Elevation angle':theta1,'Azimuth':psi1,'STDVs':[U_theta1,U_psi1,U_rho1],
                                                   'Lidar position':[Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0][0],Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0][1],Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0][2]]} #, 'Rayleigh length':Probe_param['Rayleigh Length'],'Rayleigh length uncertainty':Probe_param['Rayleigh Length uncertainty']}
        Lidar.lidar_inputs.dataframe['Scanner'] = {'Focus distance':Final_Output_UQ_Scanner['Focus distance'][0],'Elevation angle':Final_Output_UQ_Scanner['Elevation angle'][0],'Azimuth':Final_Output_UQ_Scanner['Azimuth'][0]}
        Scan_unc.append(Final_Output_UQ_Scanner)
        
        
        #%% 7)Plotting data
        QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_Scanner,Qlunc_yaml_inputs['Flags']['Line of sight Velocity Uncertainty'],False,False,False,False,False)  #Qlunc_yaml_inputs['Flags']['Scanning Pattern']  
        
    return Scan_unc,Lidar.lidar_inputs.dataframe

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
    
    #  If the insertion loss is expressed in % (X% losses):    
    # Optical_Circulator_losses = 10*np.log10(1-(X/100)) # output in dB
    
    # If we assume an SNR:
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
     UQ_telescope=[-100]
     Final_Output_UQ_Telescope={'Telescope_Uncertainty':UQ_telescope}
     Lidar.lidar_inputs.dataframe['Telescope']=Final_Output_UQ_Telescope['Telescope_Uncertainty']*np.linspace(1,1,len(Atmospheric_Scenario.temperature)) # linspace to create the appropriate length for the xarray. 
     return Final_Output_UQ_Telescope,Lidar.lidar_inputs.dataframe

#%% Sum of uncertainties in `optics` module: 
def sum_unc_optics(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    List_Unc_optics = []
    # Each try/except evaluates wether the component is included in the module and therefore in the calculations   
    # Scanner
    if Lidar.optics.scanner != None:
        try:                  
            if Lidar.wfr_model.reconstruction_model != 'None':
                   
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
    
    # Final_Output_UQ_Optics = {'Uncertainty_Optics':Uncertainty_Optics_Module, 'Uncertainty_WFR':WFR_Uncertainty['WFR_Uncertainty'],'Mean_error_PointingAccuracy':Scanner_Uncertainty['Simu_Mean_Distance_Error'],'Stdv_PointingAccuracy':Scanner_Uncertainty['STDV_Distance'], 'Rayleigh length':Scanner_Uncertainty['Rayleigh length'],'Rayleigh length uncertainty':Scanner_Uncertainty['Rayleigh length uncertainty']}
    Final_Output_UQ_Optics = {'Uncertainty_Optics':Uncertainty_Optics_Module,'Uncertainty': Scanner_Uncertainty}
    
    Lidar.lidar_inputs.dataframe['Optics Module']=Final_Output_UQ_Optics['Uncertainty_Optics']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))  # linspace to create the appropiate length for the xarray. 
    return Final_Output_UQ_Optics,Lidar.lidar_inputs.dataframe

