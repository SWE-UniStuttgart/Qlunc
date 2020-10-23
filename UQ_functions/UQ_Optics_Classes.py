# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:58:24 2020

@author: fcosta
"""

#import LiUQ_inputs

from Qlunc_ImportModules import *
import Qlunc_Help_standAlone as SA
#import pandas as pd
#import scipy.interpolate as itp
#import pdb

def UQ_Telescope(Lidar, Atmospheric_Scenario,cts):
#    toreturn={}
    UQ_telescope=[(temp*0.5+hum*0.1+curvature_lens*0.1+aberration+o_c_tele) \
                  for temp           in inputs.atm_inp.Atmospheric_inputs['temperature']\
                  for hum            in inputs.atm_inp.Atmospheric_inputs['humidity']\
                  for curvature_lens in inputs.optics_inp.Telescope_uncertainty_inputs['curvature_lens'] \
                  for aberration     in inputs.optics_inp.Telescope_uncertainty_inputs['aberration'] \
                  for o_c_tele       in inputs.optics_inp.Telescope_uncertainty_inputs['OtherChanges_tele']]
    Telescope_Losses =inputs.optics_inp.Telescope_uncertainty_inputs['losses']
    UQ_telescope=[round(UQ_telescope[i_dec],3) for i_dec in range(len(UQ_telescope))]
#    toreturn['telescope_atm_unc']=UQ_telescope
#    toreturn['telescope_losses']=Telescope_Losses
    Final_Output_UQ_Telescope={'Uncertainty_Telescope':UQ_telescope}
    return Final_Output_UQ_Telescope
    
def UQ_Scanner(Lidar, Atmospheric_Scenario,cts):
    Coord=[]
    Mean_Stdv_DISTANCE=[]  
    SimMean_DISTANCE=[]
    X,Y,Z,X0,Y0,Z0=[],[],[],[],[],[]
    Noisy_Coord=[]
    NoisyX=[]
    NoisyY=[]
    NoisyZ=[]
    coun=0
    sample_rate_count=0
    for fd_or,cone_angle_or,azimuth_or in zip(Lidar.optics.scanner.focus_dist,Lidar.optics.scanner.cone_angle,Lidar.optics.scanner.azimuth):# Take coordinates from inputs
        Mean_DISTANCE=[]
        DISTANCE=[]        
        stdv_DISTANCE=[]  
        sample_rate_count+=Lidar.optics.scanner.sample_rate
        
        #Calculating the theoretical point coordinate transformation (conversion from spherical to cartesians):
        x0=(fd_or)*np.cos(np.deg2rad(azimuth_or))*np.sin(np.deg2rad(cone_angle_or))+Lidar.optics.scanner.origin[0]
        y0=(fd_or)*np.sin(np.deg2rad(azimuth_or))*np.sin(np.deg2rad(cone_angle_or))+ Lidar.optics.scanner.origin[1]
        z0=(fd_or)*np.cos(np.deg2rad(cone_angle_or))+Lidar.optics.scanner.origin[2]+sample_rate_count
        #Storing coordinates
        X0.append(x0)
        Y0.append(y0)
        Z0.append(z0)

        for trial in range(0,100):
            
        # Create white noise with stdv selected by user for each pointing input
            n=10000 # Number of cases to combine
            del_focus_dist = np.array(np.random.normal(0 ,Lidar.optics.scanner.stdv_focus_dist,n)) # why a normal distribution??Does it have sense, can be completely random?
            del_cone_angle = np.array(np.random.normal(0,Lidar.optics.scanner.stdv_cone_angle,n))
            del_azimuth    = np.array(np.random.normal(0,Lidar.optics.scanner.stdv_azimuth,n))
            
    #        Adding noise to the theoretical position:
            noisy_fd         = fd_or    + del_focus_dist
            noisy_cone_angle = cone_angle_or + del_cone_angle 
            noisy_azimuth    = azimuth_or   + del_azimuth 
            
#            Cartesian coordinates of the noisy points:            
            x=noisy_fd*np.cos(np.deg2rad(noisy_azimuth))*np.sin(np.deg2rad(noisy_cone_angle))
            y=noisy_fd*np.sin(np.deg2rad(noisy_azimuth))*np.sin(np.deg2rad(noisy_cone_angle)) 
            z=noisy_fd*np.cos(np.deg2rad(noisy_cone_angle)) + sample_rate_count
            
            # Implement error in deployment of the tripod as a rotation over yaw, pitch and roll

            R=[[np.cos(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep)),  np.cos(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))-np.sin(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.roll_error_dep)),  np.cos(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))+np.sin(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))],
               [np.sin(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep)),  np.sin(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))+np.cos(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.roll_error_dep)),  np.sin(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))-np.cos(np.deg2rad(Lidar.lidar_inputs.yaw_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))],
               [       -np.sin(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))                                             ,                      np.cos(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.sin(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))                                                                                                                                            ,                                                                np.cos(np.deg2rad(Lidar.lidar_inputs.pitch_error_dep))*np.cos(np.deg2rad(Lidar.lidar_inputs.roll_error_dep))]]
            
            xfinal = np.matmul(R,[x,y,z])[0]+Lidar.optics.scanner.origin[0]
            yfinal = np.matmul(R,[x,y,z])[1]+Lidar.optics.scanner.origin[1]
            zfinal = np.matmul(R,[x,y,z])[2]+Lidar.optics.scanner.origin[2]

#            pdb.set_trace()
            # Distance between theoretical measured points and noisy points:
            DISTANCE.append(np.sqrt((xfinal-x0)**2+(yfinal-y0)**2+(zfinal-z0)**2))
            Mean_DISTANCE.append(np.mean(DISTANCE[trial]))    
            stdv_DISTANCE.append(np.std(DISTANCE[trial]))
            
        SimMean_DISTANCE.append(np.mean(DISTANCE))   # Mean error distance of each point in the pattern  
        Mean_Stdv_DISTANCE.append(np.mean(stdv_DISTANCE)) # Mean error distance stdv for each point in the pattern
        # Want to create a noise to add to the theoretical position to simulate the error in measurements
        #Storing coordinates
        X.append(xfinal)
        Y.append(yfinal)
        Z.append(zfinal)
        NoisyX.append(X[coun][0])
        NoisyY.append(Y[coun][0])
        NoisyZ.append(Z[coun][0])
        Noisy_Coord=[NoisyX,NoisyY,NoisyZ]
        Coord=[X0,Y0,Z0]
        coun+=1
#        Xn,Yn,Zn=X0,Y0
#        Coor={'Xn':Xn,'Yn':Yn,'Zn':Zn,'X0':X0,'Y0':Y0,'Z0':Z0}
#    plot_dist=np.mean(SimMean_DISTANCE)
#    plot_stdv_dist=np.mean(Mean_Stdv_DISTANCE)
#    noiss=np.array(np.random.normal(plot_dist,plot_stdv_dist,1000))
#    z=(noiss-np.mean(SimMean_DISTANCE))/np.mean(Mean_Stdv_DISTANCE)
#    plt.figure,
#    plt.hist(noiss,bins=35)
#    plt.show()
#    plt.figure,
#    plt.hist(z,bins=35)
#    plt.show()
#    print('STDV_Z= {}'.format(np.std(z)))
#    print('STDV_Dist= {}'.format(plot_stdv_dist))
#    DistancePointMean.append(np.mean(SimMean_DISTANCE)) #We can use that for computing the total pattern mean error distance
#    stdvPointMean.append(np.mean(Mean_Stdv_DISTANCE))   #We can use that for computing the total pattern stdv error distance
#    
#    pdb.set_trace()
    Final_Output_UQ_Scanner={'Simu_Mean_Distance':SimMean_DISTANCE,'STDV_Distance':Mean_Stdv_DISTANCE,'MeasPoint_Coordinates':Coord,'NoisyMeasPoint_Coordinates':Noisy_Coord}
    return Final_Output_UQ_Scanner#,Coor #plot_dist,plot_stdv_dist
    
def UQ_OpticalCirculator(Lidar,Atmospheric_Scenario,cts):
    Optical_Circulator_Uncertainty = [Lidar.optics.optical_circulator.insertion_loss]
    Final_Output_UQ_Optical_Circulator={'Optical_Circulator_Uncertainty':Optical_Circulator_Uncertainty}
    return Final_Output_UQ_Optical_Circulator
#%% Sum of uncertainty components in optics module: 
def sum_unc_optics(Lidar,Atmospheric_Scenario,cts):
    List_Unc_optics = []
    try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations
        Scanner_Uncertainty=Lidar.optics.scanner.Uncertainty(Lidar,Atmospheric_Scenario,cts)
        
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

            
#    pdb.set_trace()   
    Uncertainty_Optics_Module=SA.unc_comb(List_Unc_optics)
    Final_Output_UQ_Optics = {'Uncertainty_Optics':Uncertainty_Optics_Module}
    return Final_Output_UQ_Optics

#    return list(SA.flatten(Uncertainty_Optics_Module))
