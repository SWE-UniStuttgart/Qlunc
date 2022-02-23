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
GUM=0
MC=1

#%% MONTECARLO METHOD
# Define inputs
if MC==1:
    pointY=np.linspace(100,350,1000)
    pointX=100*np.ones(len(pointY))
    # pointX=[100]
    # pointY=[10,]
    alpha0=.1 # shear exponent
    N=20000#number of points to simulate
    V_h   = 12.5 # reference horizontal velocity in [m/s]
    
    stdv_X = 0
    stdv_Y = .1
    pointX_noisy=[]
    pointY_noisy=[]
    distance=[]
    distance_noisy=[]
    angles=[]
    angles_noisy=[]
    for ind_points in range(len(pointX)):
        pointX_noisy.append(np.random.normal(pointX[ind_points],stdv_X,N))
        pointY_noisy.append(np.random.normal(pointY[ind_points],stdv_Y,N))
        
        # Calculate distance and angles for the original and noisy points
        # original:
        distance.append (np.sqrt(pointX[ind_points]**2+pointY[ind_points]**2))
        angles.append(np.rad2deg(math.asin(pointY[ind_points]/distance[ind_points])))
        # noisy:
        distance_noisy.append((np.sqrt(pointX_noisy[ind_points]**2+pointY_noisy[ind_points]**2)))
        argm=np.divide(pointY_noisy[ind_points],distance[ind_points])
        angles_noisy.append([(np.rad2deg(math.asin(argm[ind_arg]))) for ind_arg in range(len(argm))])
    
    stdv_angles=[np.std(angles_noisy[ind_angle]) for ind_angle in range(len(angles_noisy))]
    
    # Homogeneous flow
    
    # Calculate radial speed
    Vrad_homo = []
    Vrad_homo=([V_h*np.cos(np.radians(angles_noisy[ind_theta])) for ind_theta in range (len(angles_noisy))])
    
    # simulation to get reconstructed Vh from the simulated points
    Vh_rec_homo=[]
    for index_vrad in range(len(angles)):      
        Vh_rec_homo.append(Vrad_homo[index_vrad]/math.cos(np.deg2rad(angles[index_vrad])))
    
    # Uncertainty
    U_Vh_homo,U_Vrad_homo=[],[]
    U_Vh_homo.append([np.std(Vh_rec_homo[ind_stdv]) for ind_stdv in range(len(Vh_rec_homo))])
    U_Vrad_homo.append([np.std(Vrad_homo[ind_stdv]) for ind_stdv in range(len(Vrad_homo))])
    # print(['U_Vh: ', U_Vh_homo[0]])
    
    # Including shear model
    
    # Calculate the hights
    H  = [np.multiply(distance[ind_mul],np.sin(np.deg2rad(angles[ind_mul]))) for ind_mul in range(len(angles_noisy)) ] # Original heights
    H2 = [np.multiply(distance_noisy[ind_mul],np.sin(np.deg2rad(angles_noisy[ind_mul]))) for ind_mul in range(len(angles_noisy))] # Noisy heights
    
    # Calculate radial speed
    Vrad_shear = []
    Vh_rec_shear=[]
    U_Vh_shear,U_Vrad_shear=[],[]
    # for ind_alpha in alpha0:
    for ind_npoints in range(len(pointX)):
        term_sh = np.divide(H2[ind_npoints],H[ind_npoints])**alpha0
        term_h  = np.multiply(V_h,np.cos(np.radians(angles_noisy[ind_npoints])))
        # pdb.set_trace()
        Vrad_shear.append(np.multiply(term_h,term_sh))
        Vh_rec_shear.append(np.divide(Vrad_shear[ind_npoints],(math.cos(np.deg2rad(angles[ind_npoints])))) )
        
        # Uncertainty            
        U_Vh_shear.append([np.std(Vh_rec_shear[ind_stdv]) for ind_stdv in range(len(Vh_rec_shear))])
        U_Vrad_shear.append([np.std(Vrad_shear[ind_stdv]) for ind_stdv in range(len(Vh_rec_shear))])
    # print(['U_Shear: ', U_Vh_shear[0]])
    # pdb.set_trace()
    
    # Plotting errors
    
    # fig,ax=plt.subplots()
    # ax.plot(pointY,U_Vh_homo[0],'-b' ,label='U_homo')
    # ax.plot(pointY,U_Vh_shear[0],'-r' ,label='U_shear')
    # ax.legend()
    # ax.set_xlabel('Height [m]')
    # ax.set_ylabel('Uncertainty')
    # ax.grid(axis='both')
    # plt.title('Vh Uncertainty')
    # pdb.set_trace()
    fig,ax1=plt.subplots()
    # for ind_al in range(len(alpha0)):
    # pdb.set_trace()
    ax1.plot(angles,U_Vh_homo[0],'-b' ,label='U_homo')
    ax1.plot(angles,U_Vh_shear[0],'-r' ,label='U_shear')
    ax1.legend()
    ax1.set_xlabel('Theta [°]')
    ax1.set_ylabel('Uncertainty [%]')
    ax1.grid(axis='both')
    plt.title('Vh Uncertainty')
    
    # fig,ax2=plt.subplots()
    # ax2.plot(pointY,U_Vrad_homo[0],'-b' ,label='U_homo')
    # ax2.plot(pointY,U_Vrad_shear[0],'-r' ,label='U_shear')
    # ax2.legend()
    # ax2.set_xlabel('Height [m]')
    # ax2.set_ylabel('Uncertainty')
    # ax2.grid(axis='both')
    # plt.title('Vrad Uncertainty')
    
    fig,ax3=plt.subplots()
    ax3.plot(angles,U_Vrad_homo[0],'-b' ,label='U_homo')
    ax3.plot(angles,U_Vrad_shear[0],'-r' ,label='U_shear')
    ax3.legend()
    ax3.set_xlabel('Theta [°]')
    ax3.set_ylabel('Uncertainty [%]')
    ax3.grid(axis='both')
    plt.title('Vrad Uncertainty')
    

#%% GUM ASSESSMENT

# if GUM==1:
#     # Define inputs
#     theta =np.linspace(0,86,1000)
#     stdv_angle = np.radians(.0573)
#     Vh    = 12.5
#     alpha =np.round(np.linspace(.1,.5,5),3)
    
#     # Uncertainty (GUM)
    
#     # Uniform flow
#     U_Vrad,U_Vh=[],[]
#     # U_Vrad.append([Vh*np.sin(np.radians(theta[ind_u]))*stdv_angle for ind_u in range(len(theta))])
#     U_Vrad.append([100*np.cos(np.radians(theta[ind_u]))*np.tan(np.radians(theta[ind_u]))*stdv_angle for ind_u in range(len(theta))])
#     # pdb.set_trace()
#     U_Vh.append([100*np.tan(np.radians(theta[ind_u]))*stdv_angle for ind_u in range(len(theta))])
    
#     # Including shear:
#     U_Vrad_sh,U_Vh_sh = [],[]
#     for ind_alpha in alpha:
#         # 
#         U_Vrad_sh.append([100*np.cos(np.radians(theta[ind_u]))*stdv_angle*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
#         U_Vh_sh.append([100*stdv_angle*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
    
#         # U_Vh_sh.append([(Vh*np.tan(np.radians(theta[ind_u]))+ind_alpha*math.atan(np.radians(theta[ind_u])))*stdv_angle for ind_u in range(len(theta))])
#         # U_Vh_sh.append([(ind_alpha*math.atan(np.radians(theta[ind_u])))*stdv_angle for ind_u in range(len(theta))])
#         # pdb.set_trace()
#         # U_Vh_sh.append([ind_alpha*(()**alpha) for ind_u in range(len(theta))])
        
    
#     # Plot errors
#     color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))
#     fig,axs1 = plt.subplots()  
#     axs1.plot(theta,U_Vrad[0],'b-',label='Uniform flow')
#     for ind_a in range(len(alpha)):
#         c=next(color)
#         axs1.plot(theta,U_Vrad_sh[ind_a],'r-.',label='Shear ({})'.format(alpha[ind_a]),c=c)
    
#     axs1.set_xlabel('theta [°]',fontsize=25)
#     axs1.set_ylabel('U [%]',fontsize=25)
#     plt.title('Uncertainty in radial velocity',fontsize=29)
#     axs1.legend()
#     axs1.grid(axis='both')
    
#     color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))
    
#     fig,axs2 = plt.subplots()  
#     axs2.plot(theta,U_Vh[0],'b-',label='Uniform flow')
#     for ind_a in range(len(alpha)):
#         c=next(color)
#         axs2.plot(theta,U_Vh_sh[ind_a],'r-.',label='Shear ({})'.format(alpha[ind_a]),c=c)
    
#     axs2.set_xlabel('theta [°]',fontsize=25)
#     axs2.set_ylabel('U [%]',fontsize=25)
    
#     plt.title('Uncertainty in horizontal velocity',fontsize=29)
#     axs2.legend()
#     axs2.grid(axis='both')