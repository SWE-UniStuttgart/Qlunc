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
# from Functions import UQ_ProbeVolume_Classes as upbc
import numpy as np
import pdb

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
    X,Y,Z,X0,Y0,Z0=[],[],[],[],[],[]
    Noisy_Coord=[]
    NoisyX=[]
    NoisyY=[]
    NoisyZ=[]
    coun=0
    sample_rate_count=0

    # #Call probe volume uncertainty function. 
    # Probe_param = Lidar.probe_volume.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)

    
    # R: Implement error in deployment of the tripod as a rotation over yaw, pitch and roll
    stdv_yaw    = np.array(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))
    stdv_pitch  = np.array(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))
    stdv_roll   = np.array(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))
    
    # stdv focus distance, cone angle and azimuth:
    # stdv_param1 = Probe_param['Focus Distance uncertainty']
    stdv_param1 = Lidar.optics.scanner.stdv_focus_dist    
    stdv_param2 = np.deg2rad(Lidar.optics.scanner.stdv_cone_angle)
    stdv_param3 = np.deg2rad(Lidar.optics.scanner.stdv_azimuth)
    
    # Differentiate between 'VAD' or 'Scanning' lidar depending on user's choice:
    if Qlunc_yaml_inputs['Components']['Scanner']['Type']=='VAD':
        param1=Lidar.optics.scanner.focus_dist
        # param1 = [np.array(Probe_param['Focus Distance'])]
        param2 = np.deg2rad(Lidar.optics.scanner.cone_angle)
        param3 = np.deg2rad(Lidar.optics.scanner.azimuth)
        
    elif Qlunc_yaml_inputs['Components']['Scanner']['Type']=='SCAN':
        
        # 'Transform coordinates from cartesians to spherical'
        param1=[]
        param2=[]
        param3=[]
        
        # When SCAN is selected user can choose specific patterns already implemented (./Qlunc/Utils/Scanning_patterns.py)
        if Qlunc_yaml_inputs['Components']['Scanner']['Pattern']=='lissajous':
            # x_init,y_init,z_init = SP.lissajous_pattern(Lidar.optics.scanner.lissajous_param[0],Lidar.optics.scanner.lissajous_param[1],Lidar.optics.scanner.lissajous_param[2],Lidar.optics.scanner.lissajous_param[3],Lidar.optics.scanner.lissajous_param[4])
            
            # x_init =np.array( [Probe_param['Focus Distance']])
            x_init,y_init,z_init = SP.lissajous_pattern(Lidar,Lidar.optics.scanner.lissajous_param[0],Lidar.optics.scanner.lissajous_param[1],Lidar.optics.scanner.lissajous_param[2],Lidar.optics.scanner.lissajous_param[3],Lidar.optics.scanner.lissajous_param[4])
        
        elif Qlunc_yaml_inputs['Components']['Scanner']['Pattern']=='None':
            x_init = Lidar.optics.scanner.x
            # x_init = np.array([Probe_param['Focus Distance']]) # This needs to be changed
            y_init = Lidar.optics.scanner.y
            z_init = Lidar.optics.scanner.z
            
        # Calculating parameter1, parameter2 and parameter3 depending on the quadrant (https://es.wikipedia.org/wiki/Coordenadas_esf%C3%A9ricas):           
        param1,param3,param2=SA.cart2sph(x_init,y_init,z_init)
        # xc,yc,zc=SA.sph2cart(param1,param3,param2)
    for param1_or,param2_or,param3_or in zip(param1,param2,param3):# Take coordinates from inputs
        Mean_DISTANCE=[]
        DISTANCE=[]        
        stdv_DISTANCE=[]  
        
        # Coordinates of the original points:
        # Calculating the theoretical point coordinate transformation + lidar origin + sample rate  (conversion from spherical to cartesians):
        x0 = (param1_or)*np.cos(param3_or)*np.sin(param2_or) + Lidar.optics.scanner.origin[0]
        y0 = (param1_or)*np.sin(param3_or)*np.sin(param2_or) + Lidar.optics.scanner.origin[1]
        z0 = (param1_or)*np.cos(param2_or) + Lidar.optics.scanner.origin[2] + sample_rate_count
        # Storing coordinates
        X0.append(x0)
        Y0.append(y0)
        Z0.append(z0)
        
        for trial in range(0,10):
            
            # Create white noise with stdv selected by user:
            n=10000 # Number of cases to combine           
            # Position, due to pointing accuracy
            del_param1 = np.array(np.random.normal(0,stdv_param1,n)) # why a normal distribution??Does it have sense, can be completely random?
            del_param2 = np.array(np.random.normal(0,stdv_param2,n))
            del_param3 = np.array(np.random.normal(0,stdv_param3,n))                        
            
            # Adding noise to the theoretical position:
            noisy_param1 = param1_or + del_param1
            noisy_param2 = param2_or + del_param2 
            noisy_param3 = param3_or + del_param3 
            # Coordinates of the noisy points:            
            x = noisy_param1*np.cos(noisy_param3)*np.sin(noisy_param2)
            y = noisy_param1*np.sin(noisy_param3)*np.sin(noisy_param2) 
            z = noisy_param1*np.cos(noisy_param2) + sample_rate_count
            
            # Apply error in inclinometers   
            # Rotation, due to inclinometers
            del_yaw     = np.random.normal(0,stdv_yaw,n)
            del_pitch   = np.random.normal(0,stdv_pitch,n)
            del_roll    = np.random.normal(0,stdv_roll,n)
            
            # Adding noise to the inclinometer stdv
            noisy_yaw   = stdv_yaw + del_yaw
            noisy_pitch = stdv_pitch + del_pitch
            noisy_roll  = stdv_roll + del_roll
            
            R = SA.sum_mat(noisy_yaw,noisy_pitch,noisy_roll)
                           
            xfinal = np.matmul(R,[x,y,z])[0] + Lidar.optics.scanner.origin[0] # Rotation
            yfinal = np.matmul(R,[x,y,z])[1] + Lidar.optics.scanner.origin[1]
            zfinal = np.matmul(R,[x,y,z])[2] + Lidar.optics.scanner.origin[2]

            # Distance between theoretical measured points and noisy points:
            DISTANCE.append(np.sqrt((xfinal-x0)**2+(yfinal-y0)**2+(zfinal-z0)**2))
            Mean_DISTANCE.append(np.mean(DISTANCE[trial]))    
            stdv_DISTANCE.append(np.std(DISTANCE[trial]))

        sample_rate_count+=Lidar.optics.scanner.sample_rate    
        SimMean_DISTANCE.append(np.mean(Mean_DISTANCE))        # Mean error distance of each point in the pattern  
        StdvMean_DISTANCE.append(np.mean(stdv_DISTANCE)) # Mean error distance stdv for each point in the pattern
        
        # Storing coordinates:
        X.append(xfinal)
        Y.append(yfinal)
        Z.append(zfinal)
        NoisyX.append(X[coun][0])
        NoisyY.append(Y[coun][0])
        NoisyZ.append(Z[coun][0])
        coun+=1
    Noisy_Coord=[NoisyX,NoisyY,NoisyZ]
    Coord=[X0,Y0,Z0]
    
    #Call probe volume uncertainty function
    Probe_param = Lidar.probe_volume.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,param1)

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
        
    Final_Output_UQ_Scanner                 = {'Simu_Mean_Distance_Error':SimMean_DISTANCE,'STDV_Distance':StdvMean_DISTANCE,'MeasPoint_Coordinates':Coord,'NoisyMeasPoint_Coordinates':Noisy_Coord,'Rayleigh length':Probe_param['Rayleigh Length'],'Rayleigh length uncertainty':Probe_param['Rayleigh Length uncertainty']}
    Lidar.lidar_inputs.dataframe['Scanner'] = ([np.mean(Final_Output_UQ_Scanner['Simu_Mean_Distance_Error'])])*len(Atmospheric_Scenario.temperature)  

    # Plotting
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_Scanner,Qlunc_yaml_inputs['Flags']['Scanning Pattern'],False,False,False)
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
                WFR_Uncertainty=Lidar.wfr_model.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,Scanner_Uncertainty)            
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
    # pdb.set_trace()
    Lidar.lidar_inputs.dataframe['Optics Module']=Final_Output_UQ_Optics['Uncertainty_Optics']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))  # linspace to create the appropiate length for the xarray. 
    return Final_Output_UQ_Optics,Lidar.lidar_inputs.dataframe

