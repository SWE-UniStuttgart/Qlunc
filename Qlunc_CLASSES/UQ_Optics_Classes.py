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

def UQ_Scanner(Lidar, Atmospheric_Scenario,cts):
    
#    rho             = Lidar.optics.scanner.focus_dist/np.cos(np.deg2rad(Lidar.optics.scanner.theta)) # beam length   
#   Estimating the error in theta and phi, for each measured height, by 'equivalent triangles' method
    vec_stdv_thetas = Lidar.optics.scanner.focus_dist*Lidar.optics.scanner.stdv_theta/np.max(Lidar.optics.scanner.focus_dist)
    vec_stdv_phis   = Lidar.optics.scanner.focus_dist*Lidar.optics.scanner.stdv_phi/np.max(Lidar.optics.scanner.focus_dist)    
    
#    # Pointing accuracy:
#    # Jacobian Method:#######################
    dxdrho   = np.cos(np.deg2rad(Lidar.optics.scanner.phi))*np.sin(np.deg2rad(Lidar.optics.scanner.theta))
    dxdtheta = Lidar.optics.scanner.focus_dist*np.cos(np.deg2rad(Lidar.optics.scanner.phi))*np.cos(np.deg2rad(Lidar.optics.scanner.theta))
    dxdphi   = -(Lidar.optics.scanner.focus_dist*np.sin(np.deg2rad(Lidar.optics.scanner.phi))*np.sin(np.deg2rad(Lidar.optics.scanner.theta)))
    
    dydrho   = np.sin(np.deg2rad(Lidar.optics.scanner.phi))*np.sin(np.deg2rad(Lidar.optics.scanner.theta))
    dydtheta = Lidar.optics.scanner.focus_dist*np.sin(np.deg2rad(Lidar.optics.scanner.phi))*np.cos(np.deg2rad(Lidar.optics.scanner.theta))
    dydphi   = Lidar.optics.scanner.focus_dist*np.cos(np.deg2rad(Lidar.optics.scanner.phi))*np.sin(np.deg2rad(Lidar.optics.scanner.theta))
    
    dzdrho   = np.cos(np.deg2rad(Lidar.optics.scanner.theta))
    dzdtheta = -(Lidar.optics.scanner.focus_dist*np.sin(np.deg2rad(Lidar.optics.scanner.theta)))
    dzdphi   = 0
    
    J=np.array([[dxdrho[0][0],dxdtheta[0][0],dxdphi[0][0]],
                [dydrho[0][0],dydtheta[0][0],dydphi[0][0]],
                [dzdrho[0],dzdtheta[0],float(dzdphi)]])
    inv_J=np.linalg.inv(J)
#    pdb.set_trace()
#    #########################
#    
    varianceSph = np.array([Lidar.optics.scanner.stdv_focus_dist,vec_stdv_thetas[0],vec_stdv_phis[0]])**2
#    
    Cart_errorPoints=np.diag((J.dot(varianceSph)).dot(inv_J))  # Error in transformation
    ErrorJx=np.sqrt((dxdrho*Lidar.optics.scanner.stdv_focus_dist)**2+(dxdtheta*Lidar.optics.scanner.stdv_theta)**2+(dxdphi*Lidar.optics.scanner.stdv_phi)**2)
    ErrorJy=np.sqrt((dydrho*Lidar.optics.scanner.stdv_focus_dist)**2+(dydtheta*Lidar.optics.scanner.stdv_theta)**2+(dydphi*Lidar.optics.scanner.stdv_phi)**2)
    ErrorJz=np.sqrt((dzdrho*Lidar.optics.scanner.stdv_focus_dist)**2+(dzdtheta*Lidar.optics.scanner.stdv_theta)**2+(dzdphi*Lidar.optics.scanner.stdv_phi)**2)
   
    # Coordinates transformation:  
    
    x,y,z = SA.sph2cart(Lidar)

    # Variance: (Formula from QUAM_2012)
    var_x = (x*(np.sqrt((Lidar.optics.scanner.stdv_focus_dist/Lidar.optics.scanner.focus_dist)**2+(Lidar.optics.scanner.stdv_theta/(Lidar.optics.scanner.theta))**2+(Lidar.optics.scanner.stdv_phi/Lidar.optics.scanner.phi)**2)))**2
    var_y = (y*(np.sqrt((Lidar.optics.scanner.stdv_focus_dist/Lidar.optics.scanner.focus_dist)**2+(Lidar.optics.scanner.stdv_theta/(Lidar.optics.scanner.theta))**2+(Lidar.optics.scanner.stdv_phi/Lidar.optics.scanner.phi)**2)))**2
    var_z = (z*(np.sqrt((Lidar.optics.scanner.stdv_focus_dist/Lidar.optics.scanner.focus_dist)**2+(Lidar.optics.scanner.stdv_theta/(Lidar.optics.scanner.theta))**2+(0*Lidar.optics.scanner.stdv_phi/Lidar.optics.scanner.phi)**2)))**2
    Variance_PointingAccuraccy=var_x+var_y+var_z
    STDV_PointingAccuraccy=np.sqrt(Variance_PointingAccuraccy)
    
    
    mean_stdv_x = np.sqrt(np.mean(var_x[0,1:])) ## FIX here
    mean_stdv_y = np.sqrt(np.mean(var_y[0,1:]))
    mean_stdv_z = np.sqrt(np.mean(var_z[0,1:]))
    stdv_total=np.mean(np.sqrt((var_x)+(var_y)+(var_z)))
    stdv_p_accuracy=np.mean(np.sqrt(mean_stdv_x**2+mean_stdv_y**2+mean_stdv_z**2))
    
    # White Noise:
    del_x = np.array(np.random.normal(0,mean_stdv_x,len(Lidar.optics.scanner.phi[0])))
    del_y = np.array(np.random.normal(0,mean_stdv_y,len(Lidar.optics.scanner.phi[0])))
    del_z = np.array(np.random.normal(0,mean_stdv_z,len(Lidar.optics.scanner.phi[0]))) #np.random.normal(0,0.031,len(phi))
    X     = x + del_x
    Y     = y + del_y
    Z     = z + del_z
    
    pdb.set_trace()
    Pointing_accuracy=np.sqrt((Lidar.optics.scanner.stdv_focus_dist)**2+(Lidar.optics.scanner.stdv_theta)**2+(Lidar.optics.scanner.stdv_phi)**2)
    return X,Y,Z,stdv_p_accuracy


    UQ_telescope=[round(UQ_telescope[i_dec],3) for i_dec in range(len(UQ_telescope))]
#    toreturn['telescope_atm_unc']=UQ_telescope
#    toreturn['telescope_losses']=Telescope_Losses
    return UQ_telescope

