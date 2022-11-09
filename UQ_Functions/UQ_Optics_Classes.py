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


    # R: Implement error in deployment of the tripod as a rotation over yaw, pitch and roll
    stdv_yaw    = np.array(np.radians(Lidar.lidar_inputs.yaw_error_dep))
    stdv_pitch  = np.array(np.radians(Lidar.lidar_inputs.pitch_error_dep))
    stdv_roll   = np.array(np.radians(Lidar.lidar_inputs.roll_error_dep))
    
   
    # Rho, theta and psi values of the measuring point    
    rho0           = [Lidar.optics.scanner.focus_dist]  
    theta0         = [np.radians(Lidar.optics.scanner.cone_angle)]
    psi0           = [np.radians(Lidar.optics.scanner.azimuth)]
    
    wind_direction = np.radians(np.linspace(Atmospheric_Scenario.wind_direction[0],Atmospheric_Scenario.wind_direction[1],360))
    
   
    # Measurement point in cartesian coordinates before applying lidar position
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
    
    # Store lidar positionning    
    lidars={}       
    lidars['Lidar_Rectangular']={'x':(lidar_coor.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2])[1]),
                                                    'y':(lidar_coor.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2])[2]),
                                                    'z':(lidar_coor.vector_pos(x,y,z,x_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0],y_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1],z_Lidar=Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2])[3])}
    lidars['Lidar_Spherical']={'rho':np.round((lidar_coor.Cart2Sph(lidars['Lidar_Rectangular']['x'],lidars['Lidar_Rectangular']['y'],lidars['Lidar_Rectangular']['z']))[1],4),
                                                  'theta':np.round((lidar_coor.Cart2Sph(lidars['Lidar_Rectangular']['x'],lidars['Lidar_Rectangular']['y'],lidars['Lidar_Rectangular']['z']))[2],4),
                                                   'psi':np.round((lidar_coor.Cart2Sph(lidars['Lidar_Rectangular']['x'],lidars['Lidar_Rectangular']['y'],lidars['Lidar_Rectangular']['z']))[3],4)}
    pdb.set_trace()

    
    # Rho, theta and psi lidar inputs and their uncertainties
    theta1,U_theta1 = lidars['Lidar_Spherical']['theta'],np.radians(Lidar.optics.scanner.stdv_cone_angle[0])
    psi1  ,U_psi1   = lidars['Lidar_Spherical']['psi'],np.radians(Lidar.optics.scanner.stdv_azimuth[0])
    rho1  ,U_rho1   = lidars['Lidar_Spherical']['rho'],Lidar.optics.scanner.stdv_focus_dist [0]
    
    #Uncertainty in the probe volume (This call needs to be changed!)
    Probe_param = Lidar.probe_volume.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,lidars)
    Lidar.lidar_inputs.dataframe['Probe Volume'] = Probe_param
    pdb.set_trace()
    
    #%% 1) State the correlations
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


    #%% 2) Create the noisy distributions (assume normal distributions):
    Uncertainty_V,Uncertainty_U,Uncertainty_Vh_MC,Uncertainty_Vh_GUM=[],[],[],[]
    u_wind_GUM, v_wind_GUM=[],[]
    Vwind_MC,Uwind_MC=[],[]
    CORR_COEF_uv,CORR_COEF_VLOS=[],[]
    U_VLOS1_GUM,U_VLOS1_MC,VLOS1_list=[],[],[]
 
    
    for ind_wind_dir in range(len(wind_direction)):  
        theta1_noisy = np.random.normal(theta1[0],U_theta1,Lidar.optics.scanner.N_MC)
        psi1_noisy   = np.random.normal(psi1[0],U_psi1,Lidar.optics.scanner.N_MC)
        rho1_noisy   = np.random.normal(rho1[0],U_rho1,Lidar.optics.scanner.N_MC)

        
        
        
    #%% 3) Obtain the Correlated distributions:
        
        theta_means = [theta1_noisy.mean()]
        theta_stds  = [theta1_noisy.std()]
        
        psi_means = [psi1_noisy.mean()]  
        psi_stds  = [psi1_noisy.std()]
        
        rho_means = [rho1_noisy.mean()] 
        rho_stds  = [rho1_noisy.std()]
        
        # Covariance Matrix:
        cov_MAT=[[              theta_stds[0]**2,                        psi_stds[0]*theta_stds[0]*psi1_theta1_corr_n,     rho_stds[0]*theta_stds[0]*theta1_rho1_corr_n  ],
                  
                  [theta_stds[0]*psi_stds[0]*psi1_theta1_corr_n ,                        psi_stds[0]**2,                   rho_stds[0]*psi_stds[0]*psi1_rho1_corr_n],
                  
                  [theta_stds[0]*rho_stds[0]*theta1_rho1_corr_n,         psi_stds[0]*rho_stds[0]*psi1_rho1_corr_n,                     rho_stds[0]**2]]
   
     
        # Multivariate distributions:
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
        
        
        # Cross correlations
        Corr_coef_theta1_psi1 = np.corrcoef(Theta1_cr,Psi1_cr)
        Corr_coef_theta1_rho1 = np.corrcoef(Theta1_cr,Rho1_cr)
        Corr_coef_rho1_psi1   = np.corrcoef(Rho1_cr,Psi1_cr)

        
        # Cross correlations
        CROS_CORR = [psi1_theta1_corr_n,theta1_rho1_corr_n,psi1_rho1_corr_n]

        
    #%% 4) VLOS uncertainty
        # Function calculating the uncertainties in VLOS following Montecarlo simulation:
        VLOS01,U_VLOS01,VLOS1_list = SA.U_VLOS_MC(Theta1_cr,Psi1_cr,Rho1_cr,Hl,Href,alpha,wind_direction,Vref,ind_wind_dir,VLOS1_list)
        U_VLOS1_MC.append(U_VLOS01)
        
        # Function calculating the uncertainties in VLOS following GUM:
        U_VLOS1 = SA.U_VLOS_GUM (theta1,psi1,rho1,U_theta1,U_psi1,U_rho1,U_VLOS01,Hl,Vref,Href,alpha,wind_direction,ind_wind_dir,CROS_CORR)
        U_VLOS1_GUM.append(U_VLOS1[0])
    
       
    #%% 5) Method for uncertainty when varying theta, psi OR rho   
    pdb.set_trace()
    # Want to vary rho
    U_VLOS_T_MC_rho,U_VLOS_THomo_MC_rho,U_VLOS_T_GUM_rho,U_VLOS_THomo_GUM_rho,rho_TESTr,thata_TESTr,psi_TESTr          =  SA.VLOS_param(np.linspace(1000,5000,500),theta1,psi1,U_theta1,U_psi1,U_rho1,Lidar.optics.scanner.N_MC,U_VLOS1,Hl,Vref,Href,alpha,wind_direction_TEST,0,[0,0,0])
    U_VLOS_T_MC_theta,U_VLOS_THomo_MC_theta,U_VLOS_T_GUM_theta,U_VLOS_THomo_GUM_theta,rho_TESTt,theta_TESTt,psi_TESTt  =  SA.VLOS_param(rho1,np.radians(np.linspace(1,89,500)),psi1,U_theta1,U_psi1,U_rho1,Lidar.optics.scanner.N_MC,U_VLOS1,Hl,Vref,Href,alpha,wind_direction_TEST,0,[0,0,0])    
    U_VLOS_T_MC_psi,U_VLOS_THomo_MC_psi,U_VLOS_T_GUM_psi,U_VLOS_THomo_GUM_psi,rho_TESTp,theta_TESTp,psi_TESTp          =  SA.VLOS_param(rho1,theta1,np.radians(np.linspace(-90,90,500)),U_theta1,U_psi1,U_rho1,Lidar.optics.scanner.N_MC,U_VLOS1,Hl,Vref,Href,alpha,wind_direction_TEST,0,[0,0,0])
    
    
    #%% Storing data
    Final_Output_UQ_Scanner                 = {'VLOS1 Uncertainty MC [m/s]':U_VLOS1_MC,'VLOS1 Uncertainty GUM [m/s]':U_VLOS1_GUM,
                                               'Vr Uncertainty homo MC rho [m/s]':U_VLOS_THomo_MC_rho,'Vr Uncertainty homo GUM rho [m/s]':U_VLOS_THomo_GUM_rho,'Vr Uncertainty MC rho [m/s]':U_VLOS_T_MC_rho,'Vr Uncertainty GUM rho [m/s]':U_VLOS_T_GUM_rho,
                                               'Vr Uncertainty homo MC theta [m/s]':U_VLOS_THomo_MC_theta,'Vr Uncertainty homo GUM theta [m/s]':U_VLOS_THomo_GUM_theta,'Vr Uncertainty MC theta [m/s]':U_VLOS_T_MC_theta,'Vr Uncertainty GUM theta [m/s]':U_VLOS_T_GUM_theta,
                                               'Vr Uncertainty homo MC psi [m/s]':U_VLOS_THomo_MC_psi,'Vr Uncertainty homo GUM psi [m/s]':U_VLOS_THomo_GUM_psi,'Vr Uncertainty MC psi [m/s]':U_VLOS_T_MC_psi,'Vr Uncertainty GUM psi [m/s]':U_VLOS_T_GUM_psi,
                                               'x':x,'y':y,'z':z,'rho':rho_TESTr,'theta':theta_TESTt,'psi':psi_TESTp,'wind direction':wind_direction,'Focus distance':rho1,'Elevation angle':theta1,'Azimuth':psi1,'STDVs':[U_theta1,U_psi1,U_rho1]} #, 'Rayleigh length':Probe_param['Rayleigh Length'],'Rayleigh length uncertainty':Probe_param['Rayleigh Length uncertainty']}
    
    Lidar.lidar_inputs.dataframe['Scanner'] = {'Focus distance':Final_Output_UQ_Scanner['Focus distance'][0],'Elevation angle':Final_Output_UQ_Scanner['Elevation angle'][0],'Azimuth':Final_Output_UQ_Scanner['Azimuth'][0]}
    pdb.set_trace()
    # Plotting
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_Scanner,Qlunc_yaml_inputs['Flags']['Line of sight Velocity Uncertainty'],False,False,False,False)  #Qlunc_yaml_inputs['Flags']['Scanning Pattern']  
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

