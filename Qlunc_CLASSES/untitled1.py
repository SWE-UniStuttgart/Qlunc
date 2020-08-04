# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:40:24 2020

@author: fcosta
"""
import numpy as np

rho0=80
theta0=0
phi0=0

#Create white noise 
stdv_rho   = 1
stdv_theta = 2
stdv_phi   = 1

# Create white noise with stdv selected by user
n=10000000
del_rho   = np.array(np.random.normal(0 ,stdv_rho,n))
del_theta = np.array(np.random.normal(0,stdv_theta,n))
del_phi   = np.array(np.random.normal(0,stdv_phi,n))
#del_phi   = np.repeat(np.array(stdv_phi),n,axis=0)
#del_theta   = np.repeat(np.array(stdv_theta),n,axis=0)
noisy_rho   = rho0   + del_rho#stdv_rho
noisy_theta = theta0 + del_theta#stdv_theta
noisy_phi   = phi0   + del_phi#stdv_phi

# Combined uncertainty (stdv)
stdv_noisy_rho=np.std(noisy_rho)
stdv_noisy_theta=np.std(noisy_theta)
stdv_noisy_phi=np.std(noisy_phi)

#stdv_total_sph=np.sqrt((stdv_rho)**2+(stdv_theta)**2+(stdv_phi)**2)

# Coordinate system Transformation:
x0=rho0*np.cos(np.deg2rad(phi0))*np.sin(np.deg2rad(theta0))
y0=rho0*np.sin(np.deg2rad(phi0))*np.sin(np.deg2rad(theta0)) 
z0=rho0*np.cos(np.deg2rad(theta0)) 

x=noisy_rho*np.cos(np.deg2rad(noisy_phi))*np.sin(np.deg2rad(noisy_theta))
y=noisy_rho*np.sin(np.deg2rad(noisy_phi))*np.sin(np.deg2rad(noisy_theta)) 
z=noisy_rho*np.cos(np.deg2rad(noisy_theta)) 


#del_x = np.array(np.random.normal(0,mean_stdv_x,len(Lidar.optics.scanner.phi[0])))
#del_y = np.array(np.random.normal(0,mean_stdv_y,len(Lidar.optics.scanner.phi[0])))
#del_z = np.array(np.random.normal(0,mean_stdv_z,len(Lidar.optics.scanner.phi[0]))) #np.random.normal(0,0.031,len(phi))


DISTANCE=(np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2))
stdv_DISTANCE=np.std(DISTANCE)
print('DISTANCE= {}'.format(np.mean(DISTANCE)))
print('stdv_DISTANCE= {}'.format(np.mean(stdv_DISTANCE)))