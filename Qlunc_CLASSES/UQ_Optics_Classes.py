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
    return UQ_telescope
    
def UQ_Scanner(Lidar, Atmospheric_Scenario,cts):

    Mean_Stdv_DISTANCE=[]
    stdvPointMean=[]
    DistancePointMean=[]  
    SimMean_DISTANCE=[]
    for fd_or,theta_or,phi_or in zip(Lidar.optics.scanner.focus_dist,Lidar.optics.scanner.theta,Lidar.optics.scanner.phi):
        Mean_DISTANCE=[]
        DISTANCE=[]        
        stdv_DISTANCE=[]  
        x0=fd_or*np.cos(np.deg2rad(phi_or))*np.sin(np.deg2rad(theta_or))
        y0=fd_or*np.sin(np.deg2rad(phi_or))*np.sin(np.deg2rad(theta_or)) 
        z0=fd_or*np.cos(np.deg2rad(theta_or))
        for trial in range(0,100):
            
        # Create white noise with stdv selected by user for each pointing input
            n=1000 # Number of cases to combine
            del_focus_dist = np.array(np.random.normal(0 ,Lidar.optics.scanner.stdv_focus_dist,n))
            del_theta      = np.array(np.random.normal(0,Lidar.optics.scanner.stdv_theta,n))
            del_phi        = np.array(np.random.normal(0,Lidar.optics.scanner.stdv_phi,n))
            
    #        Adding noise to the theoretical position:
            noisy_fd    = fd_or    + del_focus_dist
            noisy_theta = theta_or + del_theta 
            noisy_phi   = phi_or   + del_phi 
            
#            Cartesian coordinates of the noisy points:
            x=noisy_fd*np.cos(np.deg2rad(noisy_phi))*np.sin(np.deg2rad(noisy_theta))
            y=noisy_fd*np.sin(np.deg2rad(noisy_phi))*np.sin(np.deg2rad(noisy_theta)) 
            z=noisy_fd*np.cos(np.deg2rad(noisy_theta)) 
            
            # Distance between theoretical measured points and noisy points:
            DISTANCE.append(np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2))
            Mean_DISTANCE.append(np.mean(DISTANCE[trial]))    
            stdv_DISTANCE.append(np.std(DISTANCE[trial]))
#        pdb.set_trace()
        SimMean_DISTANCE.append(np.mean(DISTANCE))   # Mean distance of each trial    
        Mean_Stdv_DISTANCE.append(np.mean(stdv_DISTANCE)) # Mean distance stdv of each trial
#    DistancePointMean.append(np.mean(SimMean_DISTANCE)) #We can use that for computing the total pattern mean error distance
#    stdvPointMean.append(np.mean(Mean_Stdv_DISTANCE))   #We can use that for computing the total pattern stdv error distance
    

    return SimMean_DISTANCE,Mean_Stdv_DISTANCE
    
    
#%% Sum of uncertainty components in optics module: 
def sum_unc_optics(Lidar,Atmospheric_Scenario,cts): 
    try: # ecah try/except evaluates wether the component is included in the module, therefore in the calculations
#        if Photodetector_Uncertainty not in locals():
        Scanner_Uncertainty=Lidar.optics.scanner.Uncertainty(Lidar,Atmospheric_Scenario,cts)
    except:
        Scanner_Uncertainty=None
        print('No scanner in calculations!')
    try:
        Telescope_Uncertainty=Lidar.optics.telescope.Uncertainty(Lidar,Atmospheric_Scenario,cts)
    except:
        Telescope_Uncertainty=None
        print('No telescope in calculations!')
    pdb.set_trace()
#    ##########################
    Uncertainty_Optics_Module=2 # this is just for test
    ############################
#    List_Unc_optics1=[]
#    List_Unc_optics0=[Scanner_Uncertainty,Telescope_Uncertainty]
#    for x in List_Unc_optics0:
#        
#        if isinstance(x,list):
#           
#            List_Unc_optics0=([10**(i/10) for i in x]) # Make the list without None values and convert in watts(necessary for SA.unc_comb)
#            List_Unc_optics1.append([List_Unc_optics0]) # Make a list suitable for unc.comb function

#    Uncertainty_Optics_Module=SA.unc_comb(List_Unc_optics1)
    print('Done')
    return Uncertainty_Optics_Module

#    return list(SA.flatten(Uncertainty_Optics_Module))
