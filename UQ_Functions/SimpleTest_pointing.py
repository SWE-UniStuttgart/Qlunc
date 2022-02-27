


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
PL=1
Linear=0
#%% Define inputs

alpha       = np.array([.15,.1,.2,.25]) # shear exponent
N           = 50000 #number of points for the MC simulation
Vh          = 8.5
rho         = np.linspace(2000,2000,300)
theta       = np.linspace(0,45,300)
psi         = np.linspace(45,45,300)
stdv_rho    = 0
stdv_theta  = 0.5
stdv_psi    = 0
rho_noisy   = []
theta_noisy = []
psi_noisy   = []
for ind_noise in range(len(rho)):
    rho_noisy.append(np.random.normal(rho[ind_noise],stdv_rho,N))
    theta_noisy.append(np.random.normal(theta[ind_noise],stdv_theta,N))
    psi_noisy.append(np.random.normal(psi[ind_noise],stdv_psi,N))

#%% MONTECARLO METHOD
# Define inputs
if MC==1:

    # Homogeneous flow
    
    # Calculate radial speed
    Vrad_homo = []
    Vrad_homo=([Vh*np.cos(np.radians(theta_noisy[ind_theta]))*np.cos(np.radians(psi_noisy[ind_theta])) for ind_theta in range (len(theta_noisy))])
    # Vrad_homo=[Vh for ind_theta in range (len(theta_noisy))]
    # pdb.set_trace()
    # simulation to get reconstructed Vh from the simulated points
    Vh_rec_homo=[]
    for index_vrad in range(len(theta)):      
        Vh_rec_homo.append(Vrad_homo[index_vrad]/(math.cos(np.deg2rad(psi[index_vrad]))*math.cos(np.deg2rad(theta[index_vrad]))))
    
    # Uncertainty
    U_Vh_homo,U_Vrad_homo=[],[]
    # U_Vh_homo.append([np.std(Vh_rec_homo[ind_stdv]) for ind_stdv in range(len(Vh_rec_homo))])
    U_Vrad_homo.append([np.std(Vrad_homo[ind_stdv])  for ind_stdv in range(len(Vrad_homo))])

    # Including shear model
    U_Vh_PL,U_Vrad_Sh=[],[]
    # Calculate the hights
    # H0 = [np.multiply(distance[ind_mul],np.sin(np.deg2rad(theta[ind_mul]))) for ind_mul in range(len(theta_noisy)) ] # Original heights
    # H  = [np.multiply(distance[ind_mul],np.sin(np.deg2rad(theta_noisy[ind_mul]))) for ind_mul in range(len(theta_noisy))] # Noisy heights
        
    if PL ==1:
        # Power Law model        
        # Calculate radial speed
        Vrad_PL,Vh_rec_shear = [],[]
        for ind_npoints in range(len(rho)):
            Vrad_PL.append (Vh*(np.cos(np.radians(psi_noisy[ind_npoints]))*np.cos(np.radians(theta_noisy[ind_npoints])))*((np.sin(np.radians(theta_noisy[ind_npoints]))*rho_noisy[ind_npoints])/(np.sin(np.radians(theta[ind_npoints]))*rho[ind_npoints]))**alpha[0])
            # Vh_rec_shear.append(np.divide(Vrad_PL[ind_npoints],(math.cos(np.deg2rad(theta[ind_npoints])))) )
            
        # Uncertainty
        U_Vrad_Sh.append([np.nanstd(Vrad_PL[ind_stdv]) for ind_stdv in range(len(Vrad_PL))])           
        # U_Vh_PL.append([np.std(Vh_rec_shear[ind_stdv])*Vh for ind_stdv in range(len(Vh_rec_shear))])
        
    elif Linear==1:
        # Linear model
        # Linear factor
        Lf = 3/(2*100)
         # Calculate radial speed
        Vrad_L = []
        # Vh_rec_shear=[]
        U_Vh_L,U_Vrad_L=[],[]
        for ind_npoints in range(len(distance)):
            term_sh = (H0[ind_npoints])*Lf
            term_h  = np.multiply(Vh,np.cos(np.radians(theta_noisy[ind_npoints])))
            # pdb.set_trace()
            Vrad_L.append(np.multiply(term_h,term_sh))
            # Vh_rec_shear.append(np.divide(Vrad_PL[ind_npoints],(math.cos(np.deg2rad(theta[ind_npoints])))) )
            
        # Uncertainty            
        # U_Vh_PL.append([np.std(Vh_rec_shear[ind_stdv])*Vh for ind_stdv in range(len(Vh_rec_shear))])
        U_Vrad_Sh.append([np.std(Vrad_L[ind_stdv]) for ind_stdv in range(len(Vrad_L))])
#%% GUM METHOD

if GUM==1:
   
    # Homogeneous flow
    U_Vrad,U_Vrad_theta,U_Vrad_psi,U_Vh,U_Vrad_range=[],[],[],[],[]
    U_Vrad_theta.append([Vh*np.cos(np.radians(psi[ind_u]))*np.sin(np.radians(theta[ind_u]))*np.radians(stdv_theta) for ind_u in range(len(theta))])
    U_Vrad_psi.append([Vh*np.cos(np.radians(theta[ind_u]))*np.sin(np.radians(psi[ind_u]))*np.radians(stdv_psi) for ind_u in range(len(theta))])
    
    U_Vrad.append([np.sqrt((U_Vrad_theta[0][ind_u])**2+(U_Vrad_psi[0][ind_u])**2) for ind_u in range(len(theta))])
    
    # U_Vrad_psi.append([Vh*np.cos(np.radians(psi[ind_u]))*np.tan(np.radians(psi[ind_u]))*stdv_theta for ind_u in range(len(theta))])
    # U_Vrad.append([np.tan(np.radians(theta[ind_u]))*stdv_theta for ind_u in range(len(theta))])

    # U_Vh.append([Vh*np.tan(np.radians(theta[ind_u]))*stdv_theta for ind_u in range(len(theta))])
    
    # Including shear:
    U_Vrad_sh_theta,U_Vrad_sh_psi,U_Vh_sh,U_Vrad_sh,U_Vrad_sh_range= [],[],[],[],[]
    if PL==1:
       
        for ind_alpha in range(len( alpha)):
            U_Vrad_sh_theta.append([Vh*np.cos(np.radians(psi[ind_u]))*np.cos(np.radians(theta[ind_u]))*np.radians(stdv_theta)*abs((alpha[ind_alpha]/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
            U_Vrad_sh_psi.append([Vh*np.cos(np.radians(theta[ind_u]))*np.sin(np.radians(psi[ind_u]))*np.radians(stdv_psi) for ind_u in range(len(psi))])            
            U_Vrad_sh_range.append([Vh*alpha*(1/rho[ind_u])*np.cos(np.radians(theta[ind_u]))*np.cos(np.radians(psi[ind_u]))*stdv_rho for ind_u in range(len(psi))])
            U_Vrad_sh.append([np.sqrt((U_Vrad_sh_theta[ind_alpha][ind_u])**2+(U_Vrad_sh_psi[ind_alpha][ind_u])**2+(U_Vrad_sh_range[ind_alpha][ind_u])**2) for ind_u in range(len(rho)) ])
            # pdb.set_trace()
            # U_Vrad_sh_theta.append([Vh*np.cos(np.radians(theta[ind_u]))*stdv_theta*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
            # U_Vh_sh.append([Vh*stdv_theta*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
    elif Linear==1:
        m=1/10000
        for ind_alpha in alpha:
                # U_Vrad_sh_theta.append([Vh*np.cos(np.radians(theta[ind_u]))*((1/(np.cos(np.radians(theta[ind_u]))*np.sin(np.radians(theta[ind_u]))))-math.tan(np.radians(theta[ind_u]))) for ind_u in range(len(theta))])
                U_Vrad_sh_theta.append([distance[ind_u]*Vh*np.cos(np.radians(theta[ind_u]))*m*np.cos(np.radians(theta[ind_u]))*stdv_theta for ind_u in range(len(theta))])

                # U_Vrad_sh_theta.append([Vh*np.cos(np.radians(theta[ind_u]))*stdv_theta*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
                # U_Vh_sh.append([Vh*stdv_theta*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
    
#%% Plot errors
# pdb.set_trace()
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
    # ax1.plot(theta,U_Vh_PL[0],'+-r' ,label='U shear')
    # ax1.legend()
    # ax1.set_xlabel('Theta [°]')
    # ax1.set_ylabel('Uncertainty [%]')
    # ax1.grid(axis='both')
    # plt.title('Vh Uncertainty')
    # pdb.set_trace()
    # fig,ax2=plt.subplots()
    # ax2.plot(pointZ,U_Vrad_homo[0],'-b' ,label='U_homo')
    # ax2.plot(pointZ,U_Vrad_PL[0],'-r' ,label='U_shear')
    # ax2.legend()
    # ax2.set_xlabel('Height [m]')
    # ax2.set_ylabel('Uncertainty')
    # ax2.grid(axis='both')
    # plt.title('Vrad Uncertainty')
    
    fig,ax2=plt.subplots()
    ax2.plot(theta,U_Vrad_homo[0],'x-b' ,label='U Uniform flow (MC)')
    ax2.plot(theta,U_Vrad_PL[0],'+-r' ,label='U shear (MC)')
    ax2.legend()
    ax2.set_xlabel('Theta [°]')
    ax2.set_ylabel('Uncertainty [m/s]')
    ax2.grid(axis='both')
    plt.title('Vrad Uncertainty')
if MC==1 and GUM==1:
    # fig,ax1=plt.subplots()
    # color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))   

    # ax1.plot(theta,U_Vh_homo[0],'x-b' ,label='U uniform MC')
    # ax1.plot(theta,U_Vh_PL[0],'+-r' ,label='U shear MC')
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
    # ax2.plot(pointZ,U_Vrad_homo[0],'-b' ,label='U_homo')
    # ax2.plot(pointZ,U_Vrad_PL[0],'-r' ,label='U_shear')
    # ax2.legend()
    # ax2.set_xlabel('Height [m]')
    # ax2.set_ylabel('Uncertainty')
    # ax2.grid(axis='both')
    # plt.title('Vrad Uncertainty')
    
    fig,ax2=plt.subplots()
    ax2.plot(theta,U_Vrad[0],'b-',label='U Uniform flow (GUM)')
    color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))   
    for ind_a in range(len(alpha)):
        c=next(color)
        ax2.plot(theta,U_Vrad_sh[ind_a],'-',label='U Shear GUM  (\u03B1 = {})'.format(alpha[ind_a]),c=c)
    
    ax2.plot(theta,U_Vrad_homo[0],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
    ax2.plot(theta,U_Vrad_Sh[0],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')
    ax2.legend()
    ax2.set_xlabel('Theta [°]')
    ax2.set_ylabel('Uncertainty [m/s]')
    ax2.grid(axis='both')
    plt.title('Vrad Uncertainty')
    
    
    
    fig,ax3=plt.subplots()
    ax3.plot(psi,U_Vrad[0],'b-',label='U Uniform flow (GUM)')
    color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))   
    for ind_a in range(len(alpha)):
        c=next(color)
        ax3.plot(psi,U_Vrad_sh[ind_a],'r-',label='U Shear GUM  (\u03B1 = {})'.format(alpha[ind_a]),c=c)
    
    ax3.plot(psi,U_Vrad_homo[0],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
    ax3.plot(psi,U_Vrad_Sh[0],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')
    ax3.legend()
    ax3.set_xlabel('Psi [°]')
    ax3.set_ylabel('Uncertainty [m/s]')
    ax3.grid(axis='both')
    plt.title('Vrad Uncertainty')