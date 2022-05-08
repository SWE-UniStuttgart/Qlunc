# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:41:20 2022

@author: fcosta
"""
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 21 08:45:23 2022

@author: fcosta

"""
import scipy as sc
from scipy.stats import norm
from matplotlib.pyplot import cm
import os
os.chdir('C:/SWE_LOCAL/GIT_Qlunc/UnderDevelopment/')
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm
from DEMO_Mean_WF import weightingFun
from numpy.linalg import norm
# from Utils import Qlunc_Help_standAlone as SA

GUM    = 1
MC     = 1

#%% 1. Define inputs

class inputs ():
    def __init__(self,Href,Vref,Mode,alpha,N_MC,Npoints,rho,theta,psi,wind_direction,stdv_rho,stdv_theta,stdv_psi):
        self.Href          = Href
        self.Vref          = Vref
        self.Mode          = Mode
        self.alpha          = np.array([alpha]) # shear exponent
        self.N_MC           = np.round(N_MC) #number of points for the MC simulation
        self.Npoints        = Npoints #N° of measuring points
        self.rho            = np.linspace(rho[0],rho[1],Npoints)
        self.theta          = np.linspace(theta[0],theta[1],Npoints)
        self.psi            = np.linspace(psi[0],psi[1],Npoints)
        self.wind_direction = wind_direction # direction from x axis
        self.stdv_rho       = stdv_rho
        self.stdv_theta     = stdv_theta
        self.stdv_psi       = stdv_psi

  

# Weighting function
class WF_param():
    def __init__(self,pulsed, truncation,tau_meas,tau,c_l):
        self.pulsed     = pulsed
        self.truncation = truncation # N° of Zr to truncate the WF
        self.tau_meas   = tau_meas
        self.tau        = tau
        self.c_l        = c_l


#%% 2. Instantiate inputs
inputs=inputs(Href        = 1e-100,
              Vref        = 2,  
              Mode        = 'SCAN',
              alpha       = 0.2, # shear exponent
              N_MC        = 50000, #number of points for the MC simulation for each point in Npoints
              Npoints     = 150, #N° of measuring points 
              rho         = [2000,2000],
              theta       = [1,89],
              wind_direction = 0,
              psi         = [45,45],
              stdv_rho    = 0,    
              stdv_theta  = 0.0573,     
              stdv_psi    = 0 ) 

WF_param=WF_param(pulsed     = 1,
                  truncation = 10,
                  tau_meas   = 119.5e-9,
                  tau        = 95e-9,
                  c_l        = 3e8)
#%% This is an implementation for the Universal CS (avoid for now)
# href=100
# x_u=[1,0,0] 
# # n=[1,0,0] # lidar longitudinal axis 
# normal_u =[0,0,1] # normal vector plane xy
# if inputs.Mode == 'SCAN':
#     n=[1,0,0]
# elif inputs.Mode =='VAD':
#     n=[0,0,1]
# phi0=[]
# theta0=[]
# height= []
# # Projection of "n" onto the plane defined by the normal vector "normal_u"
# v_pro= n-np.multiply(n,normal_u)*normal_u

# # Find the angles phi (between vector x_u defining the coordinate system and projection onto plane xy of the lidar longitudinal axis defined by n -->azimuth) 
# # and theta (lidar longitudinal axis and its projection onto plane xy --> elevation angle)
# phi0   = np.degrees(math.atan2(norm(np.cross(v_pro,x_u)),np.dot(v_pro,x_u)))
# theta0 = np.degrees(math.atan2(norm(np.cross(v_pro,n)),np.dot(v_pro,n)))
# for ind in range(len(elevation_angle_lidar)):   
#     if n==normal_u: # vertical mode ("n" pointing 90° from the horizontal plane)   
#         height.append( href+abs(np.cos(np.radians(theta+elevation_angle_lidar[ind]))*rho[ind]))
#     else: 
#         height.append( href+np.sin(np.radians(theta+elevation_angle_lidar[ind]))*rho[ind])
            
# pdb.set_trace()
#%% 3. Calculate coordinates of the noisy points
rho_noisy   = []
theta_noisy = []
psi_noisy   = []
for ind_noise in range(inputs.Npoints):
    rho_noisy.append(np.random.normal(inputs.rho[ind_noise],inputs.stdv_rho,inputs.N_MC))
    theta_noisy.append(np.random.normal(inputs.theta[ind_noise],inputs.stdv_theta,inputs.N_MC))
    psi_noisy.append(np.random.normal(inputs.psi[ind_noise],inputs.stdv_psi,inputs.N_MC))

# Cartesian Point coordinates
# def polar2cart(r, theta, psi):
#     return [
#          r * np.cos(np.radians(psi)) * np.sin(np.radians(90-theta)),
#          r * np.sin(np.radians(psi)) * np.sin(np.radians(90-theta)),
#          r * np.cos(np.radians(90-theta))
#          	] 

# x,y,z=[],[],[]
# for rho_i,theta_i,psi_i in zip (rho_noisy,theta_noisy,psi_noisy):
#     [x,y,z].append([polar2cart(rho_i[ii],theta_i[ii],psi_i[ii]) for ii in range(len(rho_i))])


#%% 4. MONTECARLO METHOD
# Define inputs
if MC==1:

    # Homogeneous flow 
    
    # Calculate radial speed
    Vrad_homo = []
    # Vrad_homo=([inputs.Vref*np.cos(np.radians(theta_noisy[ind_theta]))*np.cos(np.radians(psi_noisy[ind_theta])) for ind_theta in range (len(theta_noisy))])
    Vrad_homo=([100*np.cos(np.radians(theta_noisy[ind_theta]))*np.cos(np.radians(psi_noisy[ind_theta]))/(np.cos(np.radians(inputs.theta[ind_theta]))*np.cos(np.radians(inputs.psi[ind_theta]))) for ind_theta in range (len(theta_noisy))])

    # simulation to get reconstructed Vref from the simulated points
    Vh_rec_homo_MC=[]
    for index_vrad in range(inputs.Npoints):      
        Vh_rec_homo_MC.append(Vrad_homo[index_vrad]/(math.cos(np.deg2rad(inputs.psi[index_vrad]))*math.cos(np.deg2rad(inputs.theta[index_vrad]))))
    
    # Uncertainty
    U_Vh_homo,U_Vrad_homo_MC=[],[]
    # U_Vh_homo.append([np.std(Vh_rec_homo_MC[ind_stdv]) for ind_stdv in range(len(Vh_rec_homo_MC))])
    U_Vrad_homo_MC.append([np.std(Vrad_homo[ind_stdv])  for ind_stdv in range(len(Vrad_homo))])
    
    # Including shear model
    U_Vh_PL,U_Vrad_S_MC=[],[]
    # Calculate the hights
    H0 = [inputs.Href+ np.multiply(inputs.rho[ind_mul],np.sin(np.deg2rad(inputs.theta[ind_mul]))) for ind_mul in range(len(inputs.theta)) ] # Original heights
    H  = [inputs.Href+np.multiply(rho_noisy[ind_mul],np.sin(np.deg2rad(theta_noisy[ind_mul]))) for ind_mul in range(len(theta_noisy))] # Noisy heights
    # Calculate the radial speed for the noisy points 
    Vrad_PL=[]
    for ind_npoints in range(len(inputs.rho)):
        #Vrad_PL.append (inputs.Vref*(np.cos(np.radians(psi_noisy[ind_npoints]))*np.cos(np.radians(theta_noisy[ind_npoints])))*(((inputs.Href+np.sin(np.radians(theta_noisy[ind_npoints]))*rho_noisy[ind_npoints])/inputs.Href)**inputs.alpha[0]))
        
        Vrad_PL.append (100*(np.cos(np.radians(psi_noisy[ind_npoints]))*np.cos(np.radians(theta_noisy[ind_npoints])))*(((inputs.Href+np.sin(np.radians(theta_noisy[ind_npoints]))*rho_noisy[ind_npoints])/inputs.Href)**inputs.alpha[0])\
                           /((np.cos(np.radians(inputs.psi[ind_npoints]))*np.cos(np.radians(inputs.theta[ind_npoints])))*(((inputs.Href+np.sin(np.radians(inputs.theta[ind_npoints]))*inputs.rho[ind_npoints])/inputs.Href)**inputs.alpha[0])))

    # Uncertainty: For this to be compared with Vrad_weighted[1] I need to weight Vrad_PL 
    U_Vrad_S_MC.append([np.nanstd(Vrad_PL[ind_stdv]) for ind_stdv in range(len(Vrad_PL))])
    
    
    # U_Vrad_S_MC=Vrad_weighted[1]    
    # #Create the data to apply the weighting function to different heights
    # H_interp=[]
    # Vrad_int=[]
    # rho_int=np.linspace(0,3000,5000)
    # for ind_int in range(len(inputs.theta)):
    #     H_interp.append(inputs.Href+np.multiply(rho_int,np.sin(np.deg2rad(inputs.theta[ind_int]))) )
    #     pdb.set_trace()
    #     # These are all the points along the probe volume. Do I have to use theta , psi and rho here instead of noisy contributions????
    #     Vrad_int.append ([inputs.Vref*(np.cos(np.radians(inputs.psi[ind_int]))*np.cos(np.radians(inputs.theta[ind_int])))*(((Hinterp_i)/inputs.Href)**inputs.alpha[0]) for Hinterp_i in H_interp[ind_int]])
    # # Weighting function
    # Vrad_weighted=weightingFun(H,H0,Vrad_PL,Vrad_int,rho_noisy,theta_noisy, psi_noisy,WF_param,inputs,rho_int)
   
    
    
#####################################################
    # Power Law model        
    # Calculate radial speed
    # if pulsed =1:


    # Rayleigh_length = (c_l*tau_meas)/(2*math.erf(np.sqrt(np.log(2))*(tau_meas)/(tau)))/2
    # Trun_val        = truncation*Rayleigh_length
    # offset          = 100
    
    # rho_lorentz,theta_lorentz,psi_lorentz = [],[],[]
    # H_lorentz,H_lorentz_sorted      = [],[]
    # nn=0
    # for vec_rho in range(Npoints):
    #     # Sort to apply the weighting function
    #     # index_sort=[list(enumerate(rho_noisy[vec_rho_sor])) for vec_rho_sor in range(np.shape(rho_noisy)[0])] #Get indexes
    #       index_sort=[list(enumerate(rho_noisy[vec_rho])) for vec_rho_sor in range(np.shape(rho_noisy)[0])] #Get indexes
        
    #     # for ind2sort in range(len(rho)):
    #     index_sort[vec_rho].sort(key=lambda x:x[1])  # Sort by value. Ascendent
    #     index_sort=index_sort[nn]
    #     nn+=1
    #     # pdb.set_trace()
    #     rho_lorentz0,theta_lorentz0,psi_lorentz0 = [],[],[]
    #     for inn in range(N):
    #         rho_lorentz0.append(index_sort[inn][1])
    #         theta_lorentz.append([theta_noisy[ind_r][inn] for ind_r in range(Npoints)])
    #         psi_lorentz.append([psi_noisy[ind_r][inn] for ind_r in range(Npoints)])
    #     # theta_noisy_sorted=[theta_noisy[vec_rho_sor] for vec_rho_sor in r)]
    #     rho_lorentz.append(rho_lorentz0)
    #     pdb.set_trace()

    # rho_lorentz=rho_lorentz[0]
    # WeightingFunction=[]
    # offset = 100
    # focus_distance = 0
    # z=np.linspace(-Trun_val,Trun_val,1001)
    
###########################################################
    # fig,ax=plt.subplots(), ax.plot(z,WeightingFunction)
    # fig,axs0=plt.subplots(), axs0.hist(WeightingFunction/z)
    # Vrad_PL,Vh_rec_shear,Vrad_PL_PB = [],[],[]
    
         
           
    # U_Vh_PL.append([np.std(Vh_rec_shear[ind_stdv])*Vref for ind_stdv in range(len(Vh_rec_shear))])
    # g=np.digitize(WeightingFunction,np.linspace(np.min(WeightingFunction),np.max(WeightingFunction),500))

#%% 5. GUM METHOD
            
if GUM==1:
   
    # Homogeneous flow
    U_Vrad_homo_GUM,U_Vrad_theta,U_Vrad_psi,U_Vh,U_Vrad_range=[],[],[],[],[]
    # U_Vrad_theta.append([inputs.Vref*np.cos(np.radians(inputs.psi[ind_u]))*np.sin(np.radians(inputs.theta[ind_u]))*np.radians(inputs.stdv_theta) for ind_u in range(len(inputs.theta))])
    # U_Vrad_theta.append([inputs.Vref*np.cos(np.radians(inputs.psi[ind_u]))*np.sin(np.radians(inputs.theta[ind_u]))*np.radians(inputs.stdv_theta) for ind_u in range(len(inputs.theta))])
    # U_Vrad_psi.append([inputs.Vref*np.cos(np.radians(inputs.theta[ind_u]))*np.sin(np.radians(inputs.psi[ind_u]))*np.radians(inputs.stdv_psi) for ind_u in range(len(inputs.theta))])
    
    # Unceratinty (%)
    U_Vrad_theta.append([100*np.tan(np.radians(inputs.theta[ind_u]))*np.radians(inputs.stdv_theta) for ind_u in range(len(inputs.theta))])    
    U_Vrad_psi.append([100*np.tan(np.radians(inputs.psi[ind_u]))*np.radians(inputs.stdv_psi) for ind_u in range(len(inputs.theta))])       
    U_Vrad_homo_GUM.append([np.sqrt((U_Vrad_theta[0][ind_u])**2+(U_Vrad_psi[0][ind_u])**2) for ind_u in range(len(inputs.theta))])
    
    # Including shear:
    U_Vrad_sh_theta,U_Vrad_sh_psi,U_Vh_sh,U_Vrad_S_GUM,U_Vrad_sh_range= [],[],[],[],[]       
    for ind_alpha in range(len(inputs.alpha)):
        
        
       #U_Vrad_sh_theta.append([inputs.Vref*(((np.sin(np.radians(theta_noisy[ind_u]))*rho_noisy[ind_u])/(np.sin(np.radians(inputs.theta[ind_u]))*inputs.rho[ind_u]))**inputs.alpha[ind_alpha])*np.cos(np.radians(inputs.psi[ind_u]))*np.cos(np.radians(inputs.theta[ind_u]))*np.radians(inputs.stdv_theta*inputs.theta[ind_u])*abs((inputs.alpha[ind_alpha]/math.tan(np.radians(inputs.theta[ind_u])))-np.tan(np.radians(inputs.theta[ind_u])) ) for ind_u in range(len(inputs.theta))])
       #U_Vrad_sh_theta.append([inputs.Vref*(((inputs.Href+(np.sin(np.radians(inputs.theta[ind_u]))*inputs.rho[ind_u]))/inputs.Href)**inputs.alpha[ind_alpha])*np.cos(np.radians(inputs.psi[ind_u]))*np.cos(np.radians(inputs.theta[ind_u]))*np.radians(inputs.stdv_theta)*((inputs.alpha[ind_alpha]*(inputs.rho[ind_u]*np.cos(np.radians(inputs.theta[ind_u]))/(inputs.Href+inputs.rho[ind_u]*np.sin(np.radians(inputs.theta[ind_u])))))-np.tan(np.radians(inputs.theta[ind_u])) ) for ind_u in range(len(inputs.theta))])
       #U_Vrad_sh_psi.append([inputs.Vref*(((inputs.Href+np.sin(np.radians(inputs.theta[ind_u]))*inputs.rho[ind_u])/(inputs.Href))**inputs.alpha[ind_alpha])*np.cos(np.radians(inputs.theta[ind_u]))*np.sin(np.radians(inputs.psi[ind_u]))*np.radians(inputs.stdv_psi) for ind_u in range(len(inputs.psi))])            
       # U_Vrad_sh_range.append([inputs.Vref*(((inputs.Href+np.sin(np.radians(inputs.theta[ind_u]))*inputs.rho[ind_u])/(inputs.Href))**inputs.alpha[ind_alpha])*inputs.alpha[ind_alpha]*np.sin(np.radians(inputs.theta[ind_u]))/(inputs.Href+(np.sin(np.radians(inputs.theta[ind_u]))*inputs.rho[ind_u]))*np.cos(np.radians(inputs.theta[ind_u]))*np.cos(np.radians(inputs.psi[ind_u]))*(inputs.stdv_rho) for ind_u in range(len(inputs.rho))])
       # U_Vrad_S_GUM.append([np.sqrt((np.mean(U_Vrad_sh_theta[ind_alpha][ind_u]))**2+(np.mean(U_Vrad_sh_psi[ind_alpha][ind_u]))**2+(np.mean(U_Vrad_sh_range[ind_alpha][ind_u]))**2) for ind_u in range(len(inputs.rho)) ])    
        
       # Uncertainty in %:
        U_Vrad_sh_theta.append([np.sqrt((100*np.radians(inputs.stdv_theta)*((inputs.alpha[ind_alpha]*(inputs.rho[ind_u]*np.cos(np.radians(inputs.theta[ind_u]))/(inputs.Href+inputs.rho[ind_u]*np.sin(np.radians(inputs.theta[ind_u])))))-np.tan(np.radians(inputs.theta[ind_u])) ))**2) for ind_u in range(len(inputs.theta))])
        U_Vrad_sh_psi.append([np.sqrt((100*np.tan(np.radians(inputs.psi[ind_u]))*np.radians(inputs.stdv_psi))**2) for ind_u in range(len(inputs.psi))])            
        U_Vrad_sh_range.append([np.sqrt((100*np.sin(np.radians(inputs.theta[ind_u]))*inputs.alpha[ind_alpha]/(inputs.rho[ind_u]*np.sin(np.radians(inputs.theta[ind_u]))+inputs.Href)*inputs.stdv_rho)**2) for ind_u in range(len(inputs.rho))])
                    
        
        U_Vrad_S_GUM.append([np.sqrt(((U_Vrad_sh_theta[ind_alpha][ind_u]))**2+((U_Vrad_sh_psi[ind_alpha][ind_u]))**2+((U_Vrad_sh_range[ind_alpha][ind_u]))**2) for ind_u in range(len(inputs.rho)) ])
            
        
#%% 6. Plot errors
# pdb.set_trace()

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
    # plt.title('Vref Uncertainty')
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
    ax2.plot(inputs.theta,U_Vrad_theta[0],'b-',label='U Uniform flow GUM')
    color=iter(cm.rainbow(np.linspace(0,1,len(inputs.alpha))))   
    for ind_a in range(len(inputs.alpha)):
        c=next(color)
        ax2.plot(inputs.theta,U_Vrad_sh_theta[ind_a],'-',label='U Shear GUM')    
    ax2.plot(inputs.theta,U_Vrad_homo_MC[0],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
    ax2.plot(inputs.theta,U_Vrad_S_MC[0],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')
    ax2.legend(loc=2, prop={'size': 15})
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
    r'$\rho=%.2f$' % (inputs.rho[0], ),
    r'$\psi=%.2f$' % (inputs.psi[0], ),
    r'N={}'.format(inputs.N_MC, ),
    r'Href={}'.format(inputs.Href, ),
     r'$\alpha%.2f$={}'.format(inputs.alpha[ind_a] )))
    ax2.tick_params(axis='x', labelsize=17)
    ax2.tick_params(axis='y', labelsize=17)
    # place a tex1t box in upper left in axes coords
    ax2.text(0.5, 0.95, textstr, transform=ax2.transAxes, fontsize=14,horizontalalignment='left',verticalalignment='top', bbox=props)
    ax2.set_xlabel('Theta [°]',fontsize=25)
    ax2.set_ylabel('Uncertainty [m/s]',fontsize=25)
    ax2.grid(axis='both')
    plt.title('Vrad Uncertainty',fontsize=30)
    plt.show()


    #Plot Uncertainty in Vrad with psi
    fig,ax3=plt.subplots()
    ax3.plot(inputs.psi,U_Vrad_psi[0],'b-',label='U Uniform flow GUM')
    color=iter(cm.rainbow(np.linspace(0,1,len(inputs.alpha))))   
    for ind_a in range(len(inputs.alpha)):
        c=next(color)
        ax3.plot(inputs.psi,U_Vrad_sh_psi[ind_a],'r-',label='U Shear GUM  (\u03B1 = {})'.format(inputs.alpha[ind_a]),c=c)
    ax3.plot(inputs.psi,U_Vrad_homo_MC[0],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
    ax3.plot(inputs.psi,U_Vrad_S_MC[0],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')
    ax3.legend()
    ax3.set_xlabel('Psi [°]',fontsize=25)
    ax3.set_ylabel('Uncertainty [%]',fontsize=25)
    ax3.grid(axis='both')
    plt.title('Vrad Uncertainty',fontsize=30)
    ax3.tick_params(axis='x', labelsize=17)
    ax3.tick_params(axis='y', labelsize=17)
    
    
    #Plot Uncertainty in Vrad with rho
    fig,ax4=plt.subplots()
    ax4.plot(inputs.rho,U_Vrad_homo_GUM[0],'b-',label='U Uniform flow GUM')
    color=iter(cm.rainbow(np.linspace(0,1,len(inputs.alpha))))   
    for ind_a in range(len(inputs.alpha)):
        c=next(color)
        ax4.plot(inputs.rho,U_Vrad_sh_range[ind_a],'r-',label='U Shear GUM')
    ax4.plot(inputs.rho,U_Vrad_homo_MC[0],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
    ax4.plot(inputs.rho,U_Vrad_S_MC[0],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')
    ax4.legend(loc=2, prop={'size': 15})
    ax4.set_xlabel('rho [m]',fontsize=25)
    ax4.set_ylabel('Uncertainty [%]',fontsize=25)
    ax4.grid(axis='both')
    plt.title('Vrad Uncertainty',fontsize=30)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
    r'$\theta=%.2f$' % (inputs.theta[0], ),
    r'$\psi=%.2f$' % (inputs.psi[0], ),
    r'N={}'.format(inputs.N_MC, ),
    r'Href={}'.format(inputs.Href, ),
    r'$\alpha%.2f$={}'.format(inputs.alpha[ind_a] )
    ))
    
    ax4.tick_params(axis='x', labelsize=17)
    ax4.tick_params(axis='y', labelsize=17)
    # ax4.text(0.5, 0.95, textstr, transform=ax2.transAxes, fontsize=14,horizontalalignment='left',verticalalignment='top', bbox=props)
    plt.show()
    
    
    #Plot Global Uncertainty
    fig,ax1=plt.subplots()
    ax1.plot(inputs.rho,U_Vrad_homo_GUM[0],'b-',label='U Uniform flow GUM')
    color=iter(cm.rainbow(np.linspace(0,1,len(inputs.alpha))))   
    for ind_a in range(len(inputs.alpha)):
        c=next(color)
        ax1.plot(inputs.rho,U_Vrad_S_GUM[ind_a],'-',label='U Shear GUM')    
    ax1.plot(inputs.rho,U_Vrad_homo_MC[0],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
    ax1.plot(inputs.rho,U_Vrad_S_MC[0],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')
    ax1.legend(loc=2, prop={'size': 15})
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
    r'$\rho=%.2f$' % (inputs.rho[0], ),
    r'$\psi=%.2f$' % (inputs.psi[0], ),
    r'N={}'.format(inputs.N_MC, ),
    r'Href={}'.format(inputs.Href, ),
     r'$\alpha%.2f$={}'.format(inputs.alpha[ind_a] )))
    ax1.tick_params(axis='x', labelsize=17)
    ax1.tick_params(axis='y', labelsize=17)
    # place a tex1t box in upper left in axes coords
    ax1.text(0.5, 0.95, textstr, transform=ax2.transAxes, fontsize=14,horizontalalignment='left',verticalalignment='top', bbox=props)
    ax1.set_xlabel('Theta [°]',fontsize=25)
    ax1.set_ylabel('Uncertainty [%]',fontsize=25)
    ax1.grid(axis='both')
    plt.title('Vrad Global Uncertainty',fontsize=30)
    plt.show()
    
    
print('U_MCarlo(%): ',np.mean(U_Vrad_S_MC[0]))
print('U_GUM(%)   : ',U_Vrad_S_GUM[0][0])
    
    
    
    # Histogram
    # plt.figure()
    # plt.hist(Vrad_PL[0],21)
    # plt.title('Histogram Radial velocity',fontsize=30)
    # plt.xlabel('Vrad [m/s]',fontsize=25)
    # plt.ylabel('Occurrences [-]',fontsize=25)
    
    
fig,axs5 = plt.subplots()  
axs5=plt.axes(projection='3d')
axs5.plot(inputs.theta, inputs.psi,U_Vrad_S_MC[0])
axs5.plot(inputs.theta, inputs.psi,U_Vrad_S_GUM[0])
axs5.set_xlabel('Theta [°]',fontsize=25)
axs5.set_ylabel('Psi [°]',fontsize=25)
axs5.set_zlabel('Uncertainty [%]',fontsize=25)

    

