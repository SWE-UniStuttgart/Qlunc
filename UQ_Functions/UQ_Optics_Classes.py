# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:58:24 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c)

Here we calculate the uncertainties related with components in the `optics`
module. 

    
   - noise definintions (reference in literature)
   
 
"""
import os
os.chdir('../Utils')
import Scanning_patterns as SP
from Qlunc_Help_standAlone import *
os.chdir('../Main')
import pdb
#%% TELESCOPE:
# NOT IMPLEMENTED
#==============================================================================
# def UQ_Telescope(Lidar, Atmospheric_Scenario,cts):
#     UQ_telescope=[(temp*0.5+hum*0.1+curvature_lens*0.1+aberration+o_c_tele) \
#                   for temp           in inputs.atm_inp.Atmospheric_inputs['temperature']\
#                   for hum            in inputs.atm_inp.Atmospheric_inputs['humidity']\
#                   for curvature_lens in inputs.optics_inp.Telescope_uncertainty_inputs['curvature_lens'] \
#                   for aberration     in inputs.optics_inp.Telescope_uncertainty_inputs['aberration'] \
#                   for o_c_tele       in inputs.optics_inp.Telescope_uncertainty_inputs['OtherChanges_tele']]
#     Telescope_Losses =inputs.optics_inp.Telescope_uncertainty_inputs['losses']
#     UQ_telescope=[round(UQ_telescope[i_dec],3) for i_dec in range(len(UQ_telescope))]
#     Final_Output_UQ_Telescope={'Uncertainty_Telescope':UQ_telescope}
#     return Final_Output_UQ_Telescope
#==============================================================================

#%% SCANNER:
def UQ_Scanner(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
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
                
    # R: Implement error in deployment of the tripod as a rotation over yaw, pitch and roll
    R=[[np.cos(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep)),  np.cos(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))-np.sin(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.roll_error_dep)),  np.cos(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))+np.sin(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))],
      [np.sin(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep)),  np.sin(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))+np.cos(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.roll_error_dep)),  np.sin(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))-np.cos(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))],
      [       -np.sin(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))                                             ,                      np.cos(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))                                                                                                                                            ,                                                                np.cos(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))]]
    stdv_param1=Lidar.optics.scanner.stdv_focus_dist
    stdv_param2=Lidar.optics.scanner.stdv_cone_angle
    stdv_param3=Lidar.optics.scanner.stdv_azimuth
    
    # Differentiate between 'VAD' or 'Scanning' lidar depending on user's choice:
    if Qlunc_yaml_inputs['Components']['Scanner']['Type']=='VAD':
        param1=Lidar.optics.scanner.focus_dist
        param2=Lidar.optics.scanner.cone_angle
        param3=Lidar.optics.scanner.azimuth      

    elif Qlunc_yaml_inputs['Components']['Scanner']['Type']=='SCAN':
        # 'Transform coordinates from cartesians to spherical'
        param1=[]
        param2=[]
        param3=[]
        # When SCAN is selected user can choose specific patterns already implemented (./Qlunc/Utils/Scanning_patterns.py)
        if Qlunc_yaml_inputs['Components']['Scanner']['Pattern']=='lissajous':
            x_init,y_init,z_init = SP.lissajous_pattern(20,20,20,2,2)
        elif Qlunc_yaml_inputs['Components']['Scanner']['Pattern']=='None':
            x_init = Lidar.optics.scanner.x
            y_init = Lidar.optics.scanner.y
            z_init = Lidar.optics.scanner.z
        
        # pdb.set_trace()  
        # Calculating parameter1, parameter2 and parameter3 depending on the quadrant (https://es.wikipedia.org/wiki/Coordenadas_esf%C3%A9ricas):
        param1=np.array(np.sqrt(x_init**2+y_init**2+z_init**2)) 
        for ind in range(len(z_init)):
            
            # Tolerance:
            np.around(z_init[ind],decimals=10)
            # np.around(x_init[ind],decimals=10)
            # np.around(y_init[ind],decimals=10)
            
            #Parameter2
            if int(z_init[ind])>0:
                param2.append(np.arctan(np.sqrt(x_init[ind]**2+y_init[ind]**2)/z_init[ind]))
            elif int(z_init[ind])==0:
                param2.append(np.array(np.pi/2))
            elif int(z_init[ind])<0:
                param2.append((np.pi)+(np.arctan(np.sqrt(x_init[ind]**2+y_init[ind]**2)/z_init[ind])))
            
            #Parameter3
            if int(x_init[ind])>0 and int(y_init[ind])>0:
                param3.append(np.array(np.arctan(y_init[ind]/x_init[ind])))            
            elif int(x_init[ind])>0 and int(y_init[ind])<0:
                param3.append((2*np.pi)+(np.arctan(y_init[ind]/x_init[ind])))           
            elif int(x_init[ind])<0:
                param3.append((np.pi)+(np.arctan(y_init[ind]/x_init[ind])))            
            elif int(x_init[ind])==0:
                param3.append(np.pi/2*np.array(np.sign(y_init[ind])))
    
    # pdb.set_trace()
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
            xfinal = np.matmul(R,[x,y,z])[0] + Lidar.optics.scanner.origin[0] # Rotation
            yfinal = np.matmul(R,[x,y,z])[1] + Lidar.optics.scanner.origin[1]
            zfinal = np.matmul(R,[x,y,z])[2] + Lidar.optics.scanner.origin[2]

            # Distance between theoretical measured points and noisy points:
            DISTANCE.append(np.sqrt((xfinal-x0)**2+(yfinal-y0)**2+(zfinal-z0)**2))
            Mean_DISTANCE.append(np.mean(DISTANCE[trial]))    
            stdv_DISTANCE.append(np.std(DISTANCE[trial]))

        sample_rate_count+=Lidar.optics.scanner.sample_rate    
        SimMean_DISTANCE.append(np.mean(DISTANCE))        # Mean error distance of each point in the pattern  
        StdvMean_DISTANCE.append(np.mean(stdv_DISTANCE)) # Mean error distance stdv for each point in the pattern

        # Storing coordinates:
        X.append(xfinal)
        Y.append(yfinal)
        Z.append(zfinal)
        NoisyX.append(X[coun][0])
        NoisyY.append(Y[coun][0])
        NoisyZ.append(Z[coun][0])
        coun+=1
    # pdb.set_trace()
    Noisy_Coord=[NoisyX,NoisyY,NoisyZ]
    Coord=[X0,Y0,Z0]
    
    # Svaing coordenates to a file in desktop
    file=open('C:/Users/fcosta/Desktop/data_'+Qlunc_yaml_inputs['Components']['Scanner']['Type']+'.txt','w')
    XX=repr(X0)
    YY=repr(Y0)
    ZZ=repr(Z0)
    XX_noisy=repr(NoisyX)
    Y_noisy=repr(NoisyY)
    ZZ_noisy=repr(NoisyZ)    

    file.write('\n'+Qlunc_yaml_inputs['Components']['Scanner']['Type'] +'\nX:'+XX+"\n"+'\nY:'+YY+"\n"+'\nZ:'+ZZ+"\n")
    file.close()   
        
    Final_Output_UQ_Scanner={'Simu_Mean_Distance':SimMean_DISTANCE,'STDV_Distance':StdvMean_DISTANCE,'MeasPoint_Coordinates':Coord,'NoisyMeasPoint_Coordinates':Noisy_Coord}
    return Final_Output_UQ_Scanner

#%% Optical circulator:
def UQ_OpticalCirculator(Lidar,Atmospheric_Scenario,cts):
    Optical_Circulator_Uncertainty = [Lidar.optics.optical_circulator.insertion_loss]
    Final_Output_UQ_Optical_Circulator={'Optical_Circulator_Uncertainty':Optical_Circulator_Uncertainty}
    return Final_Output_UQ_Optical_Circulator

#%% Sum of uncertainties in `optics` module: 
def sum_unc_optics(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    List_Unc_optics = []
    try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations
        Scanner_Uncertainty=Lidar.optics.scanner.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)        
    except:
        Scanner_Uncertainty=None
        print('No scanner in calculations!')
    try:
        Telescope_Uncertainty=Lidar.optics.telescope.Uncertainty(Lidar,Atmospheric_Scenario,cts)
    except:
        Telescope_Uncertainty=None
        print('No telescope in calculations!')
    try:
        Optical_circulator_Uncertainty = Lidar.optics.optical_circulator.Uncertainty(Lidar,Atmospheric_Scenario,cts)
        List_Unc_optics.append(Optical_circulator_Uncertainty['Optical_Circulator_Uncertainty'])       
    except:
        Optical_circulator_Uncertainty = None
        print('No optical circulator in calculations!')
           
    Uncertainty_Optics_Module=unc_comb(List_Unc_optics)
    Final_Output_UQ_Optics = {'Uncertainty_Optics':Uncertainty_Optics_Module}
    return Final_Output_UQ_Optics