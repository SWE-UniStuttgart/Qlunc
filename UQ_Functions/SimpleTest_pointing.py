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
GUM    = 1
MC     = 1

#%% Define inputs
Hh =100
alpha       = np.array([.134]) # shear exponent
N           = 80000 #number of points for the MC simulation
Vh          = 8.5
rho         = np.linspace(2000,2000,500)
theta       = np.linspace(0,45,500)
psi         = np.linspace(45,45,500)
stdv_rho    = 0/100     #in percentage
stdv_theta  = .5/100 #in percentage
stdv_psi    = 0/100     #in percentage
rho_noisy   = []
theta_noisy = []
psi_noisy   = []
for ind_noise in range(len(rho)):
    # rho_noisy.append(np.random.normal(rho[ind_noise],stdv_rho,N))
    # theta_noisy.append(np.random.normal(theta[ind_noise],stdv_theta,N))
    # psi_noisy.append(np.random.normal(psi[ind_noise],stdv_psi,N))
    rho_noisy.append(np.random.normal(rho[ind_noise],stdv_rho*rho[ind_noise],N))
    theta_noisy.append(np.random.normal(theta[ind_noise],stdv_theta*theta[ind_noise],N))
    psi_noisy.append(np.random.normal(psi[ind_noise],stdv_psi*psi[ind_noise],N))

#%% MONTECARLO METHOD
# Define inputs
if MC==1:

    # Homogeneous flow
    
    # Calculate radial speed
    Vrad_homo = []
    Vrad_homo=([Vh*np.cos(np.radians(theta_noisy[ind_theta]))*np.cos(np.radians(psi_noisy[ind_theta])) for ind_theta in range (len(theta_noisy))])

    # simulation to get reconstructed Vh from the simulated points
    Vh_rec_homo_MC=[]
    for index_vrad in range(len(theta)):      
        Vh_rec_homo_MC.append(Vrad_homo[index_vrad]/(math.cos(np.deg2rad(psi[index_vrad]))*math.cos(np.deg2rad(theta[index_vrad]))))
    
    # Uncertainty
    U_Vh_homo,U_Vrad_homo_MC=[],[]
    # U_Vh_homo.append([np.std(Vh_rec_homo_MC[ind_stdv]) for ind_stdv in range(len(Vh_rec_homo_MC))])
    U_Vrad_homo_MC.append([np.std(Vrad_homo[ind_stdv])  for ind_stdv in range(len(Vrad_homo))])

    # Including shear model
    U_Vh_PL,U_Vrad_S_MC=[],[]
    # Calculate the hights
    # H0 = [np.multiply(distance[ind_mul],np.sin(np.deg2rad(theta[ind_mul]))) for ind_mul in range(len(theta_noisy)) ] # Original heights
    # H  = [np.multiply(distance[ind_mul],np.sin(np.deg2rad(theta_noisy[ind_mul]))) for ind_mul in range(len(theta_noisy))] # Noisy heights
        

    # Power Law model        
    # Calculate radial speed
    Vrad_PL,Vh_rec_shear = [],[]
    for ind_npoints in range(len(rho)):
        Vrad_PL.append (Vh*(np.cos(np.radians(psi_noisy[ind_npoints]))*np.cos(np.radians(theta_noisy[ind_npoints])))*(((np.sin(np.radians(theta_noisy[ind_npoints]))*rho_noisy[ind_npoints])/(np.sin(np.radians(theta[ind_npoints]))*rho[ind_npoints]))**alpha[0]))
        # Vh_rec_shear.append(np.divide(Vrad_PL[ind_npoints],(math.cos(np.deg2rad(theta[ind_npoints])))) )
        
    # Uncertainty
    U_Vrad_S_MC.append([np.nanstd(Vrad_PL[ind_stdv]) for ind_stdv in range(len(Vrad_PL))])           
    # U_Vh_PL.append([np.std(Vh_rec_shear[ind_stdv])*Vh for ind_stdv in range(len(Vh_rec_shear))])
        

#%% GUM METHOD

if GUM==1:
   
    # Homogeneous flow
    U_Vrad_homo_GUM,U_Vrad_theta,U_Vrad_psi,U_Vh,U_Vrad_range=[],[],[],[],[]
    U_Vrad_theta.append([Vh*np.cos(np.radians(psi[ind_u]))*np.sin(np.radians(theta[ind_u]))*np.radians(stdv_theta*theta[ind_u]) for ind_u in range(len(theta))])
    U_Vrad_psi.append([Vh*np.cos(np.radians(theta[ind_u]))*np.sin(np.radians(psi[ind_u]))*np.radians(stdv_psi*psi[ind_u]) for ind_u in range(len(theta))])   
    U_Vrad_homo_GUM.append([np.sqrt((U_Vrad_theta[0][ind_u])**2+(U_Vrad_psi[0][ind_u])**2) for ind_u in range(len(theta))])
    
    # Including shear:
    U_Vrad_sh_theta,U_Vrad_sh_psi,U_Vh_sh,U_Vrad_S_GUM,U_Vrad_sh_range= [],[],[],[],[]       
    for ind_alpha in range(len(alpha)):
        U_Vrad_sh_theta.append([Vh*(((np.sin(np.radians(theta_noisy[ind_u]))*rho_noisy[ind_u])/(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))**alpha[ind_alpha])*np.cos(np.radians(psi[ind_u]))*np.cos(np.radians(theta[ind_u]))*np.radians(stdv_theta*theta[ind_u])*abs((alpha[ind_alpha]/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
        U_Vrad_sh_psi.append([Vh*(((np.sin(np.radians(theta_noisy[ind_u]))*rho_noisy[ind_u])/(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))**alpha[ind_alpha])*np.cos(np.radians(theta[ind_u]))*np.sin(np.radians(psi[ind_u]))*np.radians(stdv_psi*psi[ind_u]) for ind_u in range(len(psi))])            
        U_Vrad_sh_range.append([Vh*(((np.sin(np.radians(theta_noisy[ind_u]))*rho_noisy[ind_u])/(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))**alpha[ind_alpha])*alpha[ind_alpha]*(1/rho[ind_u])*np.cos(np.radians(theta[ind_u]))*np.cos(np.radians(psi[ind_u]))*(stdv_rho*rho[ind_u]) for ind_u in range(len(rho))])
        U_Vrad_S_GUM.append([np.sqrt((np.mean(U_Vrad_sh_theta[ind_alpha][ind_u]))**2+(np.mean(U_Vrad_sh_psi[ind_alpha][ind_u]))**2+(np.mean(U_Vrad_sh_range[ind_alpha][ind_u]))**2) for ind_u in range(len(rho)) ])
  # (((np.sin(np.radians(theta_noisy[ind_u]))*rho_noisy[ind_u])/(np.sin(np.radians(theta[ind_u]))*rho[ind_u]))**alpha[ind_alpha])
            
       
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
    ax2.plot(theta,U_Vrad_homo_GUM[0],'b-',label='U Uniform flow GUM')
    for ind_a in range(len(alpha)):
        c=next(color)
        ax2.plot(theta,U_Vrad_S_GUM[ind_a],'r-.',label='U Shear GUM (\u03B1 = {})'.format(alpha[ind_a]),c=c)
    
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
    
    #Plot Uncertainty in Vrad with theta
    fig,ax2=plt.subplots()
    ax2.plot(theta,U_Vrad_homo_GUM[0],'b-',label='U Uniform flow GUM')
    color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))   
    for ind_a in range(len(alpha)):
        c=next(color)
        ax2.plot(theta,U_Vrad_S_GUM[ind_a],'-',label='U Shear GUM  (\u03B1 = {})'.format(alpha[ind_a]),c=c)    
    ax2.plot(theta,U_Vrad_homo_MC[0],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
    ax2.plot(theta,U_Vrad_S_MC[0],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')
    ax2.legend()
    ax2.set_xlabel('Theta [°]',fontsize=25)
    ax2.set_ylabel('Uncertainty [m/s]',fontsize=25)
    ax2.grid(axis='both')
    plt.title('Vrad Uncertainty',fontsize=30)
    
    
    #Plot Uncertainty in Vrad with psi
    fig,ax3=plt.subplots()
    ax3.plot(psi,U_Vrad_homo_GUM[0],'b-',label='U Uniform flow GUM')
    color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))   
    for ind_a in range(len(alpha)):
        c=next(color)
        ax3.plot(psi,U_Vrad_S_GUM[ind_a],'r-',label='U Shear GUM  (\u03B1 = {})'.format(alpha[ind_a]),c=c)
    ax3.plot(psi,U_Vrad_homo_MC[0],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
    ax3.plot(psi,U_Vrad_S_MC[0],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')
    ax3.legend()
    ax3.set_xlabel('Psi [°]',fontsize=25)
    ax3.set_ylabel('Uncertainty [m/s]',fontsize=25)
    ax3.grid(axis='both')
    plt.title('Vrad Uncertainty',fontsize=30)
    
    #Plot Uncertainty in Vrad with rho
    fig,ax4=plt.subplots()
    ax4.plot(rho,U_Vrad_homo_GUM[0],'b-',label='U Uniform flow GUM')
    color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))   
    for ind_a in range(len(alpha)):
        c=next(color)
        ax4.plot(rho,U_Vrad_S_GUM[ind_a],'r-',label='U Shear GUM  (\u03B1 = {})'.format(alpha[ind_a]),c=c)
    ax4.plot(rho,U_Vrad_homo_MC[0],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
    ax4.plot(rho,U_Vrad_S_MC[0],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')
    ax4.legend()
    ax4.set_xlabel('rho [m]',fontsize=25)
    ax4.set_ylabel('Uncertainty [m/s]',fontsize=25)
    ax4.grid(axis='both')
    plt.title('Vrad Uncertainty',fontsize=30)
   
    # Histogram
    # plt.figure()
    # plt.hist(Vrad_PL[0],21)
    # plt.title('Histogram Radial velocity',fontsize=30)
    # plt.xlabel('Vrad [m/s]',fontsize=25)
    # plt.ylabel('Occurrences [-]',fontsize=25)
    
    
    # fig,axs5 = plt.subplots()  
    # axs5=plt.axes(projection='3d')
    # axs5.plot(theta, psi,U_Vrad_S_MC[0])