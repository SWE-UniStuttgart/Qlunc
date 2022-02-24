# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:34:29 2022

@author: fcosta
"""
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 21 08:45:23 2022

@author: fcosta

"""

import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm
GUM=1
MC=1
#%% Define inputs
# theta =np.linspace(0,86,100)
# stdv_theta = np.radians(.0573)
# alpha =np.round(np.linspace(.1,.5,5),3)
pointY=np.linspace(0,350,500)
pointX=100*np.ones(len(pointY))
alpha=np.array([.1,.15,.2,.25]) # shear exponent
N=20000#number of points for the MC simultion
Vh   = 12.5


stdv_X = 0
stdv_Y = 1
pointX_noisy=[]
pointY_noisy=[]
distance=[]
distance_noisy=[]
theta=[]
theta_noisy=[]
for ind_points in range(len(pointX)):
    pointX_noisy.append(np.random.normal(pointX[ind_points],stdv_X,N))
    pointY_noisy.append(np.random.normal(pointY[ind_points],stdv_Y,N))
    
    # Calculate distance and theta for the original and noisy points
    # original:
    distance.append (np.sqrt(pointX[ind_points]**2+pointY[ind_points]**2))
    theta.append(np.rad2deg(math.asin(pointY[ind_points]/distance[ind_points])))
    # noisy:
    distance_noisy.append((np.sqrt(pointX_noisy[ind_points]**2+pointY_noisy[ind_points]**2)))
    argm=np.divide(pointY_noisy[ind_points],distance[ind_points])
    theta_noisy.append([(np.rad2deg(math.asin(argm[ind_arg]))) for ind_arg in range(len(argm))])


# distance = 100
stdv_theta=.01#np.round(np.mean([np.std(theta_noisy[ind_angle]) for ind_angle in range(len(theta_noisy))]),4)/(N)
# pdb.set_trace()
#%% MONTECARLO METHOD
# Define inputs
if MC==1:

    # Homogeneous flow
    
    # Calculate radial speed
    Vrad_homo = []
    Vrad_homo=([Vh*np.cos(np.radians(theta_noisy[ind_theta])) for ind_theta in range (len(theta_noisy))])
    # Vrad_homo=[Vh for ind_theta in range (len(theta_noisy))]
    # pdb.set_trace()
    # simulation to get reconstructed Vh from the simulated points
    Vh_rec_homo=[]
    for index_vrad in range(len(theta)):      
        Vh_rec_homo.append(Vrad_homo[index_vrad]/math.cos(np.deg2rad(theta[index_vrad])))
    
    # Uncertainty
    U_Vh_homo,U_Vrad_homo=[],[]
    # U_Vh_homo.append([np.std(Vh_rec_homo[ind_stdv]) for ind_stdv in range(len(Vh_rec_homo))])
    U_Vrad_homo.append([np.std(Vrad_homo[ind_stdv])  for ind_stdv in range(len(Vrad_homo))])

    # Including shear model
    
    # Calculate the hights
    H  = [np.multiply(distance[ind_mul],np.sin(np.deg2rad(theta[ind_mul]))) for ind_mul in range(len(theta_noisy)) ] # Original heights
    H2 = [np.multiply(distance[ind_mul],np.sin(np.deg2rad(theta_noisy[ind_mul]))) for ind_mul in range(len(theta_noisy))] # Noisy heights
    
    # Calculate radial speed
    Vrad_shear = []
    # Vh_rec_shear=[]
    U_Vh_shear,U_Vrad_shear=[],[]
    for ind_npoints in range(len(pointX)):
        term_sh = np.divide(H2[ind_npoints],H[ind_npoints])**alpha[0]
        term_h  = np.multiply(Vh,np.cos(np.radians(theta_noisy[ind_npoints])))
        # pdb.set_trace()
        Vrad_shear.append(np.multiply(term_h,term_sh))
        # Vh_rec_shear.append(np.divide(Vrad_shear[ind_npoints],(math.cos(np.deg2rad(theta[ind_npoints])))) )
        
    # Uncertainty            
    # U_Vh_shear.append([np.std(Vh_rec_shear[ind_stdv])*Vh for ind_stdv in range(len(Vh_rec_shear))])
    U_Vrad_shear.append([np.std(Vrad_shear[ind_stdv]) for ind_stdv in range(len(Vrad_shear))])
   
#%% GUM METHOD

if GUM==1:
   
    # Homogeneous flow
    U_Vrad,U_Vh=[],[]
    U_Vrad.append([Vh*np.cos(np.radians(theta[ind_u]))*np.tan(np.radians(theta[ind_u]))*stdv_theta for ind_u in range(len(theta))])
    # U_Vrad.append([np.tan(np.radians(theta[ind_u]))*stdv_theta for ind_u in range(len(theta))])

    # U_Vh.append([Vh*np.tan(np.radians(theta[ind_u]))*stdv_theta for ind_u in range(len(theta))])
    
    # Including shear:
    U_Vrad_sh,U_Vh_sh = [],[]
    for ind_alpha in alpha:
        U_Vrad_sh.append([Vh*np.cos(np.radians(theta[ind_u]))*stdv_theta*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])

        # U_Vrad_sh.append([Vh*np.cos(np.radians(theta[ind_u]))*stdv_theta*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
        # U_Vh_sh.append([Vh*stdv_theta*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
  
    
#%% Plot errors
if GUM==1 and MC==0:
    # color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))   
    # fig,ax1= plt.subplots()  
    # ax1.plot(theta,U_Vh[0],'b-',label='U Uniform flow GUM')
    # for ind_a in range(len(alpha)):
    #     c=next(color)
    #     ax1.plot(theta,U_Vh_sh[ind_a],'r-.',label='U Shear GUM (\u03B1 = {})'.format(alpha[ind_a]),c=c)
    
    # ax1.set_xlabel('theta [°]',fontsize=25)
    # ax1.set_ylabel('U [%]',fontsize=25)
    
    # plt.title('Uncertainty in horizontal velocity (GUM)',fontsize=29)
    # ax1.legend()
    # ax1.grid(axis='both')
    
    color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))
    fig,ax2 = plt.subplots()  
    ax2.plot(theta,U_Vrad[0],'b-',label='U Uniform flow GUM')
    for ind_a in range(len(alpha)):
        c=next(color)
        ax2.plot(theta,U_Vrad_sh[ind_a],'r-.',label='U Shear GUM (\u03B1 = {})'.format(alpha[ind_a]),c=c)
    
    ax2.set_xlabel('theta [°]',fontsize=25)
    ax2.set_ylabel('U [%]',fontsize=25)
    plt.title('Uncertainty in radial velocity (GUM)',fontsize=29)
    ax2.legend()
    ax2.grid(axis='both')
elif MC==1 and GUM==0:
    fig,ax1=plt.subplots()
    # for ind_al in range(len(alpha0)):
    # pdb.set_trace()
    # ax1.plot(theta,U_Vh_homo[0],'x-b' ,label='U Uniform flow')
    # ax1.plot(theta,U_Vh_shear[0],'+-r' ,label='U shear')
    # ax1.legend()
    # ax1.set_xlabel('Theta [°]')
    # ax1.set_ylabel('Uncertainty [%]')
    # ax1.grid(axis='both')
    # plt.title('Vh Uncertainty')
    # pdb.set_trace()
    # fig,ax2=plt.subplots()
    # ax2.plot(pointY,U_Vrad_homo[0],'-b' ,label='U_homo')
    # ax2.plot(pointY,U_Vrad_shear[0],'-r' ,label='U_shear')
    # ax2.legend()
    # ax2.set_xlabel('Height [m]')
    # ax2.set_ylabel('Uncertainty')
    # ax2.grid(axis='both')
    # plt.title('Vrad Uncertainty')
    
    fig,ax2=plt.subplots()
    ax2.plot(theta,U_Vrad_homo[0],'x-b' ,label='U Uniform flow (MC)')
    ax2.plot(theta,U_Vrad_shear[0],'+-r' ,label='U shear (MC)')
    ax2.legend()
    ax2.set_xlabel('Theta [°]')
    ax2.set_ylabel('Uncertainty [%]')
    ax2.grid(axis='both')
    plt.title('Vrad Uncertainty')
if MC==1 and GUM==1:
    # fig,ax1=plt.subplots()
    # color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))   

    # ax1.plot(theta,U_Vh_homo[0],'x-b' ,label='U uniform MC')
    # ax1.plot(theta,U_Vh_shear[0],'+-r' ,label='U shear MC')
    # ax1.plot(theta,U_Vh[0],'b-',label='Uniform flow')
    # for ind_a in range(len(alpha)):
    #     c=next(color)
    #     ax1.plot(theta,U_Vh_sh[ind_a],'r-.',label='U Shear GUM ({})'.format(alpha[ind_a]),c=c)
    
    # ax1.legend()
    # ax1.set_xlabel('Theta [°]')
    # ax1.set_ylabel('Uncertainty [%]')
    # ax1.grid(axis='both')
    # plt.title('Vh Uncertainty')
    # pdb.set_trace()
    # fig,ax2=plt.subplots()
    # ax2.plot(pointY,U_Vrad_homo[0],'-b' ,label='U_homo')
    # ax2.plot(pointY,U_Vrad_shear[0],'-r' ,label='U_shear')
    # ax2.legend()
    # ax2.set_xlabel('Height [m]')
    # ax2.set_ylabel('Uncertainty')
    # ax2.grid(axis='both')
    # plt.title('Vrad Uncertainty')
    
    fig,ax2=plt.subplots()
    ax2.plot(theta,U_Vrad[0],'b-',label='U Uniform flow')
    color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))   
    for ind_a in range(len(alpha)):
        c=next(color)
        ax2.plot(theta,U_Vrad_sh[ind_a],'r-.',label='U Shear GUM  ({})'.format(alpha[ind_a]),c=c)
    
    ax2.plot(theta,U_Vrad_homo[0],'x-b' ,label='U uniform MC')
    ax2.plot(theta,U_Vrad_shear[0],'+-r' ,label='U shear MC')
    ax2.legend()
    ax2.set_xlabel('Theta [°]')
    ax2.set_ylabel('Uncertainty [%]')
    ax2.grid(axis='both')
    plt.title('Vrad Uncertainty')