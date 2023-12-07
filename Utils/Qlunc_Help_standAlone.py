# -*- coding: utf-8 -*-
""".

Created on Mon May 18 00:03:43 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c) 
"""

from Utils.Qlunc_ImportModules import *
import pdb

#%%# used to flatt at some points along the code:
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (list,tuple)) else (a,))) 

#%% Rotation matrix for inclinometers in scanner    
def sum_mat(noisy_yaw,noisy_pitch, noisy_roll):
    """.
    
    Calculates the rotation matrix to apply uncertainty due to inclinometers deployment. Location: Qlunc_Help_standAlone.py
    
    Parameters
    ----------    
    * Noisy_yaw, noisy_pitch, noisy_roll
        Errors in yaw, pitch and roll
        
        
    Returns
    -------    
    Mean of the matrix after MonteCarlo simulation
    """
    R=[]
    for i in range(len(noisy_yaw)):    
        R.append([[np.cos(noisy_yaw[i])*np.cos(noisy_pitch[i])  ,  np.cos(noisy_yaw[i])*np.sin(noisy_pitch[i])*np.sin(noisy_roll[i])-np.sin(noisy_yaw[i])*np.cos(noisy_roll[i])  ,  np.cos(noisy_yaw[i])*np.sin(noisy_pitch[i])*np.cos(noisy_roll[i])+np.sin(noisy_yaw[i])*np.sin(noisy_roll[i])],
                  [np.sin(noisy_yaw[i])*np.cos(noisy_pitch[i])  ,  np.sin(noisy_yaw[i])*np.sin(noisy_pitch[i])*np.sin(noisy_roll[i])+np.cos(noisy_yaw[i])*np.cos(noisy_roll[i])  ,  np.sin(noisy_yaw[i])*np.sin(noisy_pitch[i])*np.cos(noisy_roll[i])-np.cos(noisy_yaw[i])*np.sin(noisy_roll[i])],
                  [       -np.sin(noisy_pitch[i])               ,  np.cos(noisy_pitch[i])*np.sin(noisy_roll[i])                                                                  ,  np.cos(noisy_pitch[i])*np.cos(noisy_roll[i])]])
    R_mean=np.sum(R,axis=0)/len(noisy_yaw)
    return R_mean

#%% sum dB:
def sum_dB(data,uncorrelated):
    """.
    
    Add up dB's. Location: Qlunc_Help_standAlone.py
    
    Parameters
    ----------    
    * SNR_data
        Signal to noise ratio
        
    * Bool
        Uncorrelated noise (default): True
        
    Returns
    -------    
    Sum of dBW
    
    """
    Sum_decibels=[] 
    # Sum_in_watts=[]
    to_watts=[]
    if uncorrelated:
        for ind in range(len(data)):
            to_watts.append(10**(data[ind]/10))
        Sum_in_watts=sum(to_watts)
        Sum_decibels=10*np.log10(Sum_in_watts)         
    else:
        print('correlated noise use case is not included yet')
        # Sumat= []
        # Sum_decibels=[]
        # for ii in data:
        #     watts=10**(ii/10)   
        #     Sumat.append (watts)
        # Sum_in_watts = sum(Sumat)
        # Sum_decibels.append(10*np.log10(Sum_in_watts) )        
    return Sum_decibels

#%% Combine uncertainties:
# The uncertainty combination is made following GUM
def unc_comb(data): 
    """.
    
    Uncertainty expansion - Location: Qlunc_Help_standAlone.py
    
    Parameters
    ----------   
    * data
        data is provided as a list of elements want to add on. Input data is expected to be in dB.
        
    Returns
    -------   
    list
    
    """
    data_watts  = []
    res_dB      = []
    res_watts   = []
    zipped_data = []
    if not isinstance (data,np.ndarray):
        data=np.array(data)    
    
    if len(data)==1:
        res_dB = list(data)
    else:
        for data_row in range(np.shape(data)[0]):# transform into watts        
            try:  
                data_db=data[data_row,:]
            except:
                data_db=data[data_row][0]             
            data_watts.append(10**(data_db/10))
        for i in range(len(data_watts[0])): # combining all uncertainties making sum of squares and the sqrt of the sum
            zipped_data.append(list(zip(*data_watts))[i])
            res_watts.append(np.sqrt(sum(map (lambda x: x**2,zipped_data[i])))) #  Combined stdv
            # res_watts.append(sum(map (lambda x: x**2,zipped_data[i]))) #   Combined Variance
            
            res_dB=10*np.log10(res_watts) #Convert into dB 
        del data_db
    return np.array(res_dB)

#%% System coordinates transformation

    
def sph2cart(rho,theta,phi): 
    x=[]
    y=[]
    z=[]    
    for i in range(len(rho)):
        x.append(rho[i] * np.cos(theta[i]) * np.cos(phi[i]))
        y.append(rho[i] * np.cos(theta[i]) * np.sin(phi[i]) )
        z.append(rho[i] * np.sin(theta[i]) )
    return(np.around(x,5) , np.around(y,5) , np.around(z,5))

def cart2sph(x,y,z): 
    rho=[]
    theta=[]
    phi=[]
    for ind in range(len(z)):
        rho.append(np.sqrt(x[ind]**2 + y[ind]**2 + z[ind]**2))
        if z[ind] < 0:
            theta.append(-math.acos( np.sqrt(x[ind]**2 + y[ind]**2) / np.sqrt(x[ind]**2 + y[ind]**2 + z[ind]**2) ))
        elif z[ind] >= 0:
            theta.append(math.acos( np.sqrt(x[ind]**2 + y[ind]**2) / np.sqrt(x[ind]**2 + y[ind]**2 + z[ind]**2) ))
        phi.append(math.atan2(y[ind] , x[ind]))
 
    return(np.array(rho) , np.array(theta) , np.array(phi)) # foc_dist, aperture angle, azimuth


#%% pulsed lidar probe volume calculations

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]




#%% RMSE

def rmse(f,ff):
    rm=[]
    rms=[]
    # ind_rm=0
    sum_rm=[]
    # for ffi,fi in zip(ff,f):
    rm=([(np.array(ff)-np.array(f))**2])
    rms=(np.sqrt(np.sum(rm)/len(ff)))
    # ind_rm=ind_rm+1
    return np.array(rms)



   # LoveU LU!


def VLOS_param (Lidar,rho,theta,psi,u_theta1,u_psi1,u_rho1,N_MC,Hl,Vref,Href,alpha,wind_direction_TEST,ind_wind_dir):
    #####################################
    #####################################
    # HARD CODED  Important HARD CODED  Need solution!!!
    wind_direction_TEST = np.radians([0])
    # wind_tilt_TEST      = np.radians([0])
    ######################################
    #######################################

    
    #If want to vary range    
    if len (rho) !=1:
        rho_TEST   = rho
        theta_TEST = theta*np.ones(len(rho_TEST))
        psi_TEST   = psi*np.ones(len(rho_TEST))
        u_theta1   = 0
        u_psi1   = 0
        ind_i = theta_TEST
    #If want to vary elevation angle 
    elif len(theta)!=1:
        theta_TEST = theta
        rho_TEST   = rho[0]*np.ones(len(theta_TEST))
        psi_TEST   = psi*np.ones(len(theta_TEST))
        u_rho1   = 0
        u_psi1   = 0
        ind_i = rho_TEST
    #If want to vary azimuth angle  
    elif len(psi)!=1:
        psi_TEST   = psi
        rho_TEST   = rho[0]*np.ones(len(psi_TEST))
        theta_TEST = theta*np.ones(len(psi_TEST))
        u_rho1   = 0
        u_theta1  = 0
        ind_i = rho_TEST

    # Calculate radial speed uncertainty for an heterogeneous flow
    U_VLOS_T_MC,U_VLOS_T_GUM,U_VLOS_T=[],[],[]   
    Corr=Lidar.optics.scanner.correlations
  
    # MC method
    # Covariance matrix
    cov_MAT    =  [[u_theta1**2 ,        u_theta1*u_psi1*0   ,u_theta1*u_rho1*0],
                   [u_theta1*u_psi1*0 ,     u_psi1**2           ,u_psi1*u_rho1*0   ],
                   [u_theta1*u_rho1*0 ,     u_psi1*u_rho1*0     ,u_rho1**2         ]]
   
    U_VLOS1=[]
    U_VLOS_T=U_VLOS_MC(Lidar,cov_MAT,theta_TEST,psi_TEST,rho_TEST,Hl,Href,alpha,wind_direction_TEST,Vref,0,U_VLOS1)
    #Store results
    U_VLOS_T_MC.append(np.array(U_VLOS_T))
    
    
    # GUM method
    U_VLOS_T_GUM=(U_VLOS_GUM (Lidar,theta_TEST,psi_TEST,rho_TEST,u_theta1,u_psi1,u_rho1,Hl,Vref,Href,alpha,wind_direction_TEST,0)) # For an heterogeneous flow (shear))  
    return (U_VLOS_T,U_VLOS_T_GUM,rho_TEST,theta_TEST,psi_TEST)        



def U_VLOS_MC(Lidar,cov_MAT,theta,psi,rho,Hl,Href,alpha,wind_direction,Vref,ind_wind_dir,U_VLOS1):
     """.
    
     Performs a Montecarlo simulation to estimate the uncertainty in the line of sight velocity ( $V_{LOS}$ ). Location: Qlunc_Help_standAlone.py
    
     Parameters
     ----------    
     * correlated distributions theta, psi and rho
    
     * wind direction [degrees]
     
     * ind_wind_dir: index for looping
    
     * $H_{ref}$: Reference height  at which $V_{ref}$ is taken [m]
    
     * $V_{ref}$: reference velocity taken from an external sensor (e.g. cup anemometer) [m/s]
    
     * alpha: power law exponent [-] 
    
     * Hl: Lidar height [m]
               
     Returns
     -------    
     * Estimated line of sight wind speed [np.array]
     * Estimated average of the $V_{LOS}$
     * Estimated uncertainty in the $V_{LOS}$ [int]
     """
     
     for ind_0 in range(len(theta)):

     
         # Multivariate
         # pdb.set_trace()
         Theta1_cr,Psi1_cr,Rho1_cr= multivariate_normal.rvs([theta[ind_0] , psi[ind_0] , rho[ind_0]] , cov_MAT , Lidar.optics.scanner.N_MC).T   
         
         A=(Hl + (np.sin(Theta1_cr) * Rho1_cr)) / Href   
         VLOS1 = Vref * (A**alpha) * (np.cos(Theta1_cr) * (np.cos(Psi1_cr - wind_direction[ind_wind_dir]) + (np.tan(0)*np.tan(Theta1_cr))))
         
         U_VLOS1.append(np.std(VLOS1))
     # pdb.set_trace()
     return(U_VLOS1)


def U_VLOS_GUM (Lidar,theta1,psi1,rho1,u_theta1,u_psi1,u_rho1,Hl,Vref,Href,alpha,wind_direction,ind_wind_dir):
    Cont_Theta,Cont_Psi,Cont_Rho=[],[],[]
    """.
    
    Analytical model based on the Guide to the expression of Uncertainty in Measurements (GUM) to estimate the uncertainty in the line of sight velocity ( $V_{LOS}$ ). Location: Qlunc_Help_standAlone.py
    
    Parameters
    ----------    
    * correlated distributions theta, psi and rho
    
    * wind direction [degrees]
     
    * ind_wind_dir: index for looping
    
    * $H_{ref}$: Reference height  at which $V_{ref}$ is taken [m]
    
    * $V_{ref}$: reference velocity taken from an external sensor (e.g. cup anemometer) [m/s]
    
    * alpha: power law exponent [-] 
    
    * Hl: Lidar height [m]
     
    * CROS_CORR: Correlation terms --> correlation betwee elevation and azimuth angles
        
        
    Returns
    -------    
    * Estimated uncertainty in the line of sight wind speed [np.array]
     
    """
    U_Vlos1_GUM=[]
    for i in range(len(rho1)):
        H_t1 = ((rho1[i] * np.sin(theta1[i])+Hl) / Href)
        
        # for ind_alpha in range(len(alpha)):
        # VLOS uncertainty
        # Calculate and store Vlosi
        # Vlos_GUM['V1'] = Vref * (H_t1**alpha) * np.cos(theta1) * np.cos(psi1 - wind_direction[ind_wind_dir])
        
        # Partial derivatives Vlosi with respect theta, psi and rho
    
        # pdb.set_trace()
        dVlos1dtheta1   =     Vref * (H_t1**alpha) * (alpha * ((rho1[i] * (np.cos(theta1[i]))**2) / (rho1[i] * np.sin(theta1[i])+Hl)) - np.sin(theta1[i])) * np.cos(psi1[i] - wind_direction[ind_wind_dir])
        dVlos1dpsi1     =   - Vref * (H_t1**alpha) * (np.cos(theta1[i]) * np.sin(psi1[i] - wind_direction[ind_wind_dir]))
        dVlos1drho1     =     Vref * (H_t1**alpha) * alpha * (np.sin(theta1[i]) / (rho1[i] * np.sin(theta1[i])+Hl)) * np.cos(theta1[i]) * np.cos(psi1[i] - wind_direction[ind_wind_dir])
    
        Ux= [[u_theta1*u_theta1 ,u_theta1*u_psi1*0   ,u_theta1*u_psi1*0],
            [u_psi1*u_theta1*0   ,u_psi1*u_psi1      ,u_psi1*u_rho1*0],
            [u_theta1*u_rho1*0   ,u_rho1*u_psi1*0     ,u_rho1*u_rho1]]
    
    
        # pdb.set_trace()
        # Store data
        Cont_Theta.append(dVlos1dtheta1)
        Cont_Psi.append(dVlos1dpsi1)
        Cont_Rho.append(dVlos1drho1)
        # Influence coefficients matrix for Vlosi uncertainty estimation
        Cx = np.array([dVlos1dtheta1  , dVlos1dpsi1  ,  dVlos1drho1]).T     
        # Ouputs covariance matrix
        
        Uy=Cx.dot(Ux).dot(np.transpose(Cx))
        
        # Uncertainty of Vlosi. Here we account for rho, theta and psi uncertainties and their correlations.
        U_Vlos1_GUM.append((np.sqrt(Uy[0][0])))

    return([U_Vlos1_GUM,Cont_Theta,Cont_Psi,Cont_Rho])



#%%
#############################
#####CovarainceMatrix
#############################

'''
This function calculated the covariance matrix
'''
def MultiVar (Lidar,Vlos_corrcoeff, U_Vlos,autocorr_theta,autocorr_psi,autocorr_rho,autocorr_V,mode ):
    # Covariance Matrix:

        if mode=='GUM1':
            # pdb.set_trace()

            psi1_psi2_corr        = Lidar.optics.scanner.correlations[0]
            psi1_psi3_corr        = Lidar.optics.scanner.correlations[1]
            psi2_psi3_corr        = Lidar.optics.scanner.correlations[2]           
            
            theta1_theta2_corr    = Lidar.optics.scanner.correlations[3]
            theta1_theta3_corr    = Lidar.optics.scanner.correlations[4]
            theta2_theta3_corr    = Lidar.optics.scanner.correlations[5]
            
            rho1_rho2_corr        = Lidar.optics.scanner.correlations[6]
            rho1_rho3_corr        = Lidar.optics.scanner.correlations[7]
            rho2_rho3_corr        = Lidar.optics.scanner.correlations[8]
            
            psi1_theta1_corr      = Lidar.optics.scanner.correlations[9]
            psi2_theta2_corr      = Lidar.optics.scanner.correlations[10]
            psi3_theta3_corr      = Lidar.optics.scanner.correlations[11]
            
            psi1_theta2_corr      = Lidar.optics.scanner.correlations[12]
            psi1_theta3_corr      = Lidar.optics.scanner.correlations[13]            
            
            psi2_theta1_corr      = Lidar.optics.scanner.correlations[14]            
            psi2_theta3_corr      = Lidar.optics.scanner.correlations[15]
            
            psi3_theta1_corr      = Lidar.optics.scanner.correlations[16]
            psi3_theta2_corr      = Lidar.optics.scanner.correlations[17]
            
            
            #Lidar1
            u_theta1 = np.radians(Lidar.optics.scanner.stdv_cone_angle[0][0])
            u_psi1   = np.radians(Lidar.optics.scanner.stdv_azimuth[0][0])
            u_rho1   = Lidar.optics.scanner.stdv_focus_dist [0][0]
            # Lidar 2 
            u_theta2 = np.radians(Lidar.optics.scanner.stdv_cone_angle[1][0])
            u_psi2   = np.radians(Lidar.optics.scanner.stdv_azimuth[1][0])
            u_rho2   = Lidar.optics.scanner.stdv_focus_dist [1][0]
            # Lidar 3 
            u_theta3 = np.radians(Lidar.optics.scanner.stdv_cone_angle[2][0])
            u_psi3   = np.radians(Lidar.optics.scanner.stdv_azimuth[2][0])
            u_rho3   =Lidar.optics.scanner.stdv_focus_dist [2][0]


        elif mode=='GUM2':
            # pdb.set_trace()
            psi1_psi2_corr        = Lidar.optics.scanner.correlations[0]
            psi1_psi3_corr        = Lidar.optics.scanner.correlations[1]
            psi2_psi3_corr        = Lidar.optics.scanner.correlations[2]           
            theta1_theta2_corr    = Lidar.optics.scanner.correlations[3]
            theta1_theta3_corr    = Lidar.optics.scanner.correlations[4]
            theta2_theta3_corr    = Lidar.optics.scanner.correlations[5]
            rho1_rho2_corr        = Lidar.optics.scanner.correlations[6]
            rho1_rho3_corr        = Lidar.optics.scanner.correlations[7]
            rho2_rho3_corr        = Lidar.optics.scanner.correlations[8]
            psi1_theta1_corr      = Lidar.optics.scanner.correlations[9]
            psi2_theta2_corr      = Lidar.optics.scanner.correlations[10]
            psi3_theta3_corr      = Lidar.optics.scanner.correlations[11]
            psi1_theta2_corr      = Lidar.optics.scanner.correlations[12]
            psi1_theta3_corr      = Lidar.optics.scanner.correlations[13]            
            psi2_theta1_corr      = Lidar.optics.scanner.correlations[14]            
            psi2_theta3_corr      = Lidar.optics.scanner.correlations[15]
            psi3_theta1_corr      = Lidar.optics.scanner.correlations[16]
            psi3_theta2_corr      = Lidar.optics.scanner.correlations[17]

            #Lidar1
            u_theta1 = 0
            u_psi1   = 0
            u_rho1   = 0
            # Lidar 2 
            u_theta2 = 0
            u_psi2   = 0
            u_rho2   = 0
            # Lidar 3 
            u_theta3 = 0
            u_psi3   = 0
            u_rho3   = 0
            # pdb.set_trace()
        elif mode=='MC1':
            
            psi1_psi2_corr        = Lidar.optics.scanner.correlations[0]
            psi1_psi3_corr        = Lidar.optics.scanner.correlations[1]
            psi2_psi3_corr        = Lidar.optics.scanner.correlations[2]           
            theta1_theta2_corr    = Lidar.optics.scanner.correlations[3]
            theta1_theta3_corr    = Lidar.optics.scanner.correlations[4]
            theta2_theta3_corr    = Lidar.optics.scanner.correlations[5]
            rho1_rho2_corr        = Lidar.optics.scanner.correlations[6]
            rho1_rho3_corr        = Lidar.optics.scanner.correlations[7]
            rho2_rho3_corr        = Lidar.optics.scanner.correlations[8]
            psi1_theta1_corr      = Lidar.optics.scanner.correlations[9]
            psi2_theta2_corr      = Lidar.optics.scanner.correlations[10]
            psi3_theta3_corr      = Lidar.optics.scanner.correlations[11]
            psi1_theta2_corr      = Lidar.optics.scanner.correlations[12]
            psi1_theta3_corr      = Lidar.optics.scanner.correlations[13]            
            psi2_theta1_corr      = Lidar.optics.scanner.correlations[14]            
            psi2_theta3_corr      = Lidar.optics.scanner.correlations[15]
            psi3_theta1_corr      = Lidar.optics.scanner.correlations[16]
            psi3_theta2_corr      = Lidar.optics.scanner.correlations[17]

            
            #Lidar1
            u_theta1 = np.radians(Lidar.optics.scanner.stdv_cone_angle[0][0])
            u_psi1   = np.radians(Lidar.optics.scanner.stdv_azimuth[0][0])
            u_rho1   = Lidar.optics.scanner.stdv_focus_dist [0][0]
            # Lidar 2 
            u_theta2 = np.radians(Lidar.optics.scanner.stdv_cone_angle[1][0])
            u_psi2   = np.radians(Lidar.optics.scanner.stdv_azimuth[1][0])
            u_rho2   = Lidar.optics.scanner.stdv_focus_dist [1][0]
            # Lidar 3 
            u_theta3 = np.radians(Lidar.optics.scanner.stdv_cone_angle[2][0])
            u_psi3   = np.radians(Lidar.optics.scanner.stdv_azimuth[2][0])
            u_rho3   =Lidar.optics.scanner.stdv_focus_dist [2][0]

        elif mode=='MC2':
            # pdb.set_trace()            
            psi1_psi2_corr        = Lidar.optics.scanner.correlations[0]
            psi1_psi3_corr        = Lidar.optics.scanner.correlations[1]
            psi2_psi3_corr        = Lidar.optics.scanner.correlations[2]           
            theta1_theta2_corr    = Lidar.optics.scanner.correlations[3]
            theta1_theta3_corr    = Lidar.optics.scanner.correlations[4]
            theta2_theta3_corr    = Lidar.optics.scanner.correlations[5]
            rho1_rho2_corr        = Lidar.optics.scanner.correlations[6]
            rho1_rho3_corr        = Lidar.optics.scanner.correlations[7]
            rho2_rho3_corr        = Lidar.optics.scanner.correlations[8]
            psi1_theta1_corr      = Lidar.optics.scanner.correlations[9]
            psi2_theta2_corr      = Lidar.optics.scanner.correlations[10]
            psi3_theta3_corr      = Lidar.optics.scanner.correlations[11]
            psi1_theta2_corr      = Lidar.optics.scanner.correlations[12]
            psi1_theta3_corr      = Lidar.optics.scanner.correlations[13]            
            psi2_theta1_corr      = Lidar.optics.scanner.correlations[14]            
            psi2_theta3_corr      = Lidar.optics.scanner.correlations[15]
            psi3_theta1_corr      = Lidar.optics.scanner.correlations[16]
            psi3_theta2_corr      = Lidar.optics.scanner.correlations[17]

            #Lidar1
            u_theta1 = 0
            u_psi1   = 0
            u_rho1   = 0
            # Lidar 2 
            u_theta2 = 0
            u_psi2   = 0
            u_rho2   = 0
            # Lidar 3 
            u_theta3 = 0
            u_psi3   = 0
            u_rho3   = 0
            # pdb.set_trace()
            
        psi1_rho1_corr        = 0
        psi1_rho2_corr        = 0
        psi1_rho3_corr        = 0
        psi2_rho1_corr        = 0
        psi2_rho2_corr        = 0
        psi2_rho3_corr        = 0
        psi3_rho3_corr        = 0
        psi3_rho2_corr        = 0
        psi3_rho1_corr        = 0
        theta1_rho1_corr      = 0
        theta1_rho2_corr      = 0
        theta1_rho3_corr      = 0
        theta2_rho1_corr      = 0  
        theta2_rho2_corr      = 0
        theta2_rho3_corr      = 0
        theta3_rho1_corr      = 0
        theta3_rho2_corr      = 0
        theta3_rho3_corr      = 0
        
        if len(Lidar.optics.scanner.origin)==3:
            cov_MAT0=[[np.array(u_theta1**2*autocorr_theta),                    np.array(u_theta2*u_theta1*theta1_theta2_corr),           np.array(u_theta3*u_theta1*theta1_theta3_corr),       np.array(u_psi1*u_theta1*psi1_theta1_corr) ,      np.array(u_psi2*u_theta1*psi2_theta1_corr),      np.array(u_psi3*u_theta1*psi3_theta1_corr),     np.array(u_rho1*u_theta1*theta1_rho1_corr),    np.array(u_rho2*u_theta1*theta1_rho2_corr),         np.array(u_rho3*u_theta1*theta1_rho3_corr)   ,     u_theta1*U_Vlos[0]*0,                    u_theta1*U_Vlos[1]*0 ,                          u_theta1*U_Vlos[2]  *0  ],
                     [np.array(u_theta1*u_theta2*theta1_theta2_corr),          np.array(u_theta2**2*autocorr_theta),                     np.array(u_theta3*u_theta2*theta2_theta3_corr),       np.array(u_psi1*u_theta2*psi1_theta2_corr),       np.array(u_psi2*u_theta2*psi2_theta2_corr) ,     np.array(u_psi3*u_theta2*psi3_theta2_corr),     np.array(u_rho1*u_theta2*theta2_rho1_corr),     np.array(u_rho2*u_theta2*theta2_rho2_corr),         np.array(u_rho3*u_theta2*theta2_rho3_corr  )    , u_theta2*U_Vlos[0]*0                   , u_theta2*U_Vlos[1]*0 ,                          u_theta2*U_Vlos[2]  *0  ],
                     [np.array(u_theta1*u_theta3*theta1_theta3_corr)  ,        np.array(u_theta3*u_theta2*theta2_theta3_corr),           np.array(u_theta3**2*autocorr_psi),                   np.array(u_theta3*u_psi1*psi1_theta3_corr),       np.array(u_theta3*u_psi2*psi2_theta3_corr),      np.array(u_theta3*u_psi3*psi3_theta3_corr),     np.array(u_rho1*u_theta3*theta3_rho1_corr),     np.array(u_rho2*u_theta3*theta3_rho2_corr),         np.array( u_rho3*u_theta3*theta3_rho3_corr   )  , u_theta3*U_Vlos[0]*0                   , u_theta3*U_Vlos[1] *0   ,                       u_theta3*U_Vlos[2]  *0  ],                 
                     
                     [np.array(u_theta1*u_psi1*psi1_theta1_corr)  ,            np.array(u_theta2*u_psi1*psi1_theta2_corr),              np.array(u_theta3*u_psi1*psi1_theta3_corr),            np.array(u_psi1**2*autocorr_psi),                 np.array(u_psi2*u_psi1*psi1_psi2_corr),          np.array(u_psi1*u_psi3*psi1_psi3_corr),        np.array(u_rho1*u_psi1*psi1_rho1_corr),        np.array(  u_rho2*u_psi1*psi1_rho2_corr)        ,   np.array(u_rho3*u_psi1*psi1_rho3_corr   )     ,    u_psi1*U_Vlos[0]*0                        , u_psi1*U_Vlos[1]*0     ,                      u_psi1*U_Vlos[2]  *0   ],
                     [np.array(u_theta1*u_psi2*psi2_theta1_corr),              np.array(u_theta2*u_psi2*psi2_theta2_corr) ,             np.array(u_theta3*u_psi2*psi2_theta3_corr) ,           np.array(u_psi1*u_psi2*psi1_psi2_corr),           np.array(      u_psi2**2*autocorr_psi),          np.array(u_psi2*u_psi3*psi2_psi3_corr),        np.array(u_rho1*u_psi2*psi2_rho1_corr),        np.array(  u_rho2*u_psi2*psi2_rho2_corr)      ,     np.array(u_rho3*u_psi2*psi2_rho3_corr ),           u_psi2*U_Vlos[0]*0                         , u_psi2*U_Vlos[1]*0                          ,u_psi2*U_Vlos[2]  *0   ],
                     [np.array(u_theta1*u_psi3*psi3_theta1_corr),              np.array(u_theta2*u_psi3*psi3_theta2_corr) ,             np.array(u_theta3*u_psi3*psi3_theta3_corr) ,           np.array(u_psi1*u_psi3*psi1_psi3_corr),           np.array(u_psi2*u_psi3*psi2_psi3_corr)   ,       np.array(u_psi3**2*autocorr_psi),              np.array(u_rho1*u_psi3*psi3_rho1_corr),        np.array(   u_rho2*u_psi3*psi3_rho2_corr)      ,    np.array(  u_rho3*u_psi3*psi3_rho3_corr ),            u_psi3*U_Vlos[0]*0                      , u_psi3*U_Vlos[1]*0                          ,u_psi3*U_Vlos[2]  *0   ],
                                          
                     
                     [np.array(u_theta1*u_rho1*theta1_rho1_corr),              np.array(u_theta2*u_rho1*theta2_rho1_corr),            np.array(u_theta3*u_rho1*theta3_rho1_corr )    ,            np.array(u_psi1*u_rho1*psi1_rho1_corr),        np.array(   u_psi2*u_rho1*psi2_rho1_corr),       np.array(   u_psi3*u_rho1*psi2_rho3_corr),     np.array(   u_rho1**2*autocorr_rho),           np.array(   u_rho2*u_rho1*rho1_rho2_corr )     ,    np.array(u_rho3*u_rho1*rho1_rho3_corr )     ,       u_rho1*U_Vlos[0]*0                         ,u_rho1*U_Vlos[1]*0                          ,u_rho1*U_Vlos[2]     *0],
                     [np.array(u_theta1*u_rho2*theta1_rho2_corr),              np.array(u_theta2*u_rho2*theta2_rho2_corr),            np.array(u_theta3*u_rho2*theta3_rho2_corr),                 np.array(u_psi1*u_rho2*psi1_rho2_corr),        np.array(  u_psi2*u_rho2*psi2_rho2_corr),        np.array(  u_psi3*u_rho2*psi3_rho2_corr),      np.array( u_rho1*u_rho2*rho1_rho2_corr),       np.array(    u_rho2**2*autocorr_rho    )       ,    np.array( u_rho3*u_rho2*rho2_rho3_corr),             u_rho2*U_Vlos[0] *0  ,                    u_rho2*U_Vlos[1] *0 ,                         u_rho2*U_Vlos[2]    *0 ],
                     [np.array(u_theta1*u_rho3*theta1_rho3_corr),              np.array(u_theta2*u_rho3*theta2_rho3_corr),            np.array(u_theta3*u_rho3*theta3_rho3_corr),                 np.array(u_psi1*u_rho3*psi1_rho3_corr),        np.array(  u_psi2*u_rho3*psi2_rho3_corr),        np.array(  u_psi3*u_rho3*psi3_rho3_corr),      np.array( u_rho1*u_rho3*rho1_rho3_corr),       np.array(   u_rho3*u_rho2*rho2_rho3_corr )     ,    np.array(  u_rho3**2*autocorr_rho ),                 u_rho3*U_Vlos[0] *0  ,                     u_rho3*U_Vlos[1] *0 ,                        u_rho3*U_Vlos[2]    *0 ],
                                        
                     
                     [u_theta1*U_Vlos[0]*0    ,                          u_theta2*U_Vlos[0]*0               ,          u_theta3*U_Vlos[0]*0               ,        u_psi1*U_Vlos[0]*0    ,                      u_psi2*U_Vlos[0]*0,                 u_psi3*U_Vlos[0]*0,                   u_rho1*U_Vlos[0]*0   ,                 u_rho2*U_Vlos[0]*0   ,                    u_rho3*U_Vlos[0]*0   ,            U_Vlos[0]**2*autocorr_V,                     U_Vlos[0]*U_Vlos[1]*Vlos_corrcoeff[0]   ,     U_Vlos[0]*U_Vlos[2]*Vlos_corrcoeff[1]],
                     [u_theta1*U_Vlos[1]*0    ,                          u_theta2*U_Vlos[1]*0               ,          u_theta3*U_Vlos[1]*0               ,        u_psi1*U_Vlos[1]*0    ,                      u_psi2*U_Vlos[1]*0,                 u_psi3*U_Vlos[1]*0,                  u_rho1*U_Vlos[1]*0   ,                 u_rho2*U_Vlos[1]*0   ,                    u_rho3*U_Vlos[1]*0   ,            U_Vlos[0]*U_Vlos[1]*Vlos_corrcoeff[0]   ,    U_Vlos[1]**2*autocorr_V,                      U_Vlos[1]*U_Vlos[2]*Vlos_corrcoeff[2]],
                     [u_theta1*U_Vlos[2]*0    ,                          u_theta2*U_Vlos[2]*0               ,          u_theta3*U_Vlos[2]*0               ,         u_psi1*U_Vlos[2]*0    ,                     u_psi2*U_Vlos[2]*0,                 u_psi3*U_Vlos[2]*0,                   u_rho1*U_Vlos[2]*0   ,                 u_rho2*U_Vlos[2]*0   ,                   u_rho3*U_Vlos[2]*0   ,             U_Vlos[0]*U_Vlos[2]*Vlos_corrcoeff[1]  ,    U_Vlos[1]*U_Vlos[2]*Vlos_corrcoeff[2] ,      U_Vlos[2]**2*autocorr_V             ]
                     ]
            cov_MAT=[]
            for i in cov_MAT0:
                cov_MAT.append(list([n.item() for n in i]))
        
        else:      
            cov_MAT0=[[np.array([u_theta1**2*autocorr_theta]),                  np.array([u_theta2*u_theta1*theta1_theta2_corr]),    np.array([u_psi1*u_theta1*psi1_theta1_corr]) ,     np.array([u_psi2*u_theta1*psi2_theta1_corr]),   np.array([u_rho1*u_theta1*theta1_rho1_corr]),  np.array([u_rho2*u_theta1*theta1_rho2_corr])   ,    u_theta1*U_Vlos[0]*0,                        u_theta1*U_Vlos[1]*0                     ],
                     [np.array([u_theta1*u_theta2*theta1_theta2_corr]),        np.array([u_theta2**2*autocorr_theta]),            np.array([u_psi1*u_theta2*psi1_theta2_corr]),      np.array([u_psi2*u_theta2*psi2_theta2_corr]) ,  np.array([u_rho1*u_theta2*theta2_rho1_corr]),  np.array([u_rho2*u_theta2*theta2_rho2_corr])  ,     u_theta2*U_Vlos[0]*0                       ,u_theta2*U_Vlos[1]*0               ],
                     [np.array([u_theta1*u_psi1*psi1_theta1_corr])  ,     np.array([u_theta2*u_psi1*psi1_theta2_corr]),             np.array([u_psi1**2*autocorr_psi]),                 np.array([u_psi2*u_psi1*psi1_psi2_corr]),       np.array([u_rho1*u_psi1*psi1_rho1_corr]),      np.array([u_rho2*u_psi1*psi1_rho2_corr])       ,    u_psi1*U_Vlos[0]*0                         ,u_psi1*U_Vlos[1]*0                    ],
                     [np.array([u_theta1*u_psi2*psi2_theta1_corr]),       np.array([u_theta2*u_psi2*psi2_theta2_corr]) ,          np.array([u_psi1*u_psi2*psi1_psi2_corr]),               np.array([u_psi2**2*autocorr_psi]),            np.array([u_rho1*u_psi2*psi2_rho1_corr]),      np.array([u_rho2*u_psi2*psi2_rho2_corr] )     ,     u_psi2*U_Vlos[0]*0                         ,u_psi2*U_Vlos[1]*0                  ],
                     [np.array([u_theta1*u_rho1*theta1_rho1_corr]),       np.array([u_theta2*u_rho1*theta2_rho1_corr]),        np.array([u_psi1*u_rho1*psi1_rho1_corr]),               np.array([u_psi2*u_rho1*psi2_rho1_corr]),      np.array([u_rho1**2*autocorr_rho]),            np.array([u_rho2*u_rho1*rho1_rho2_corr])      ,      u_rho1*U_Vlos[0]*0                         ,u_rho1*U_Vlos[1]*0                   ],
                     [np.array([u_theta1*u_rho2*theta1_rho2_corr]),       np.array([u_theta2*u_rho2*theta2_rho2_corr]),        np.array([u_psi1*u_rho2*psi1_rho2_corr]),              np.array([u_psi2*u_rho2*psi2_rho2_corr]),      np.array(  [u_rho1*u_rho2*rho1_rho2_corr]),    np.array([u_rho2**2*autocorr_rho])     ,               u_rho2*U_Vlos[0]*0  ,                        u_rho2*U_Vlos[1]*0                 ],
                     [u_theta1*U_Vlos[0]*0    ,                                             u_theta2*U_Vlos[0]*0     ,                     u_psi1*U_Vlos[0]*0   ,                                u_psi2*U_Vlos[0]*0,                            u_rho1*U_Vlos[0]*0   ,                        u_rho2*U_Vlos[0]*0 ,                       U_Vlos[0]**2*autocorr_V,                   U_Vlos[0]*U_Vlos[1]*Vlos_corrcoeff[0]          ],
                     [u_theta1*U_Vlos[1]*0  ,                                               u_theta2*U_Vlos[1]*0  ,                        u_psi1*U_Vlos[1]*0 ,                                  u_psi2*U_Vlos[1]*0,                              u_rho1*U_Vlos[1]*0  ,                       u_rho2*U_Vlos[1]*0 ,                      U_Vlos[0]*U_Vlos[1]*Vlos_corrcoeff[0],      U_Vlos[1]**2*autocorr_V                     ]]
            cov_MAT=[]
            for i in cov_MAT0:
                cov_MAT.append(list([n.item() for n in i]))
        return  cov_MAT

#%% Calculate correlations between Vlos1, ans Vlos2
def Vlos_correlations(Lidar,Vlos_corr,Atmospheric_Scenario,wind_direction, ind_wind_dir,ind_alpha,lidars):
    U_Vlos_MCM =np.zeros(len(Lidar.optics.scanner.origin))
    # Find the first covariance  matrix
    cov_MAT=MultiVar(Lidar,Vlos_corr, U_Vlos_MCM  ,       1     ,      1     ,       1    ,    0 ,     'MC1'  )
    # Find the multivariate distributions
    if len(Lidar.optics.scanner.origin)==3: # For the triple solution
        # pdb.set_trace()
        Theta1_cr,Theta2_cr,Theta3_cr,Psi1_cr,Psi2_cr,Psi3_cr,Rho1_cr,Rho2_cr,Rho3_cr,Vlos1_cr,Vlos2_cr,Vlos3_cr=multivariate_normal.rvs(np.concatenate([lidars['Lidar0_Spherical']['theta'] , lidars['Lidar1_Spherical']['theta'] , lidars['Lidar2_Spherical']['theta'], lidars['Lidar0_Spherical']['psi'] , lidars['Lidar1_Spherical']['psi'] , lidars['Lidar2_Spherical']['psi'], lidars['Lidar0_Spherical']['rho'] , lidars['Lidar1_Spherical']['rho'] , lidars['Lidar2_Spherical']['rho'], np.array([0]) , np.array([0]), np.array([0])],axis=0), cov_MAT , Lidar.optics.scanner.N_MC).T
        # Store data
        Theta_cr = [Theta1_cr , Theta2_cr ,Theta3_cr] 
        Psi_cr   = [Psi1_cr , Psi2_cr , Psi3_cr ]
        Rho_cr   = [Rho1_cr , Rho2_cr, Rho3_cr]

    else: # For the dual solution
        Theta1_cr,Theta2_cr,Psi1_cr,Psi2_cr,Rho1_cr,Rho2_cr,Vlos1_cr,Vlos2_cr=multivariate_normal.rvs(np.concatenate([lidars['Lidar0_Spherical']['theta'] , lidars['Lidar1_Spherical']['theta'] , lidars['Lidar0_Spherical']['psi'] , lidars['Lidar1_Spherical']['psi'] , lidars['Lidar0_Spherical']['rho'] , lidars['Lidar1_Spherical']['rho'] , np.array([0]), np.array([0])],axis=0), cov_MAT , Lidar.optics.scanner.N_MC).T
        # Store data
        Theta_cr = [Theta1_cr , Theta2_cr ] 
        Psi_cr   = [Psi1_cr , Psi2_cr  ]
        Rho_cr   = [Rho1_cr , Rho2_cr]
    
    Vlos_cr,U_Vlos_MCM=[],[]    
    beta=np.radians(Atmospheric_Scenario.wind_tilt)
    for i in range(len(Lidar.optics.scanner.origin)):
        H_cr= (Rho_cr[i] * np.sin(Theta_cr[i]) + Lidar.optics.scanner.origin[i][2]) / Lidar.optics.scanner.Href
        
        ### VLOS calculations ############################      
        Vlos_cr.append (Atmospheric_Scenario.Vref * (H_cr**Atmospheric_Scenario.PL_exp[ind_alpha]) * np.cos(Theta_cr[i]) * (np.cos(Psi_cr[i] - wind_direction[ind_wind_dir]) + (np.tan(beta)*np.tan(Theta_cr[i]))))
        
        ### Uncertainty VLOS calculations ############################  
        U_Vlos_MCM.append(np.sqrt(np.std(Vlos_cr[i])**2 + Lidar.lidar_inputs.dataframe['Intrinsic Uncertainty [m/s]']**2 + Lidar.optics.scanner.stdv_Estimation[0][0]**2))
       
    # CORRELATIONS Vlos
    Corr_combi        = list(itertools.combinations(Vlos_cr, 2)) # amount of Vlos combinations
    Correlations_Vlos = {'V1':[],'V2':[],'V3':[]}
    for i_combi in range(len(Corr_combi)):
        Correlations_Vlos['V{}'.format(i_combi+1)].append(np.corrcoef(Corr_combi[i_combi])[0][1])


    return (Correlations_Vlos, Vlos_cr, U_Vlos_MCM,Theta_cr ,Psi_cr,Rho_cr )
    
#%% ##########################################
##########################################
#Uncertainty Vlos and Vh following MCM
##########################################
##############################################
#%%
'''
This function performs an estimation of the uncertainty in Vlos and u and v wind velocity components based on the Montecarlo method. 

First it calculates the uncertainty of Vlos assigning multivariated normally distributed probability density functions
    to the iput quantities rho, theta (aperture angle) and psi(azimuth). The covariance matrix includes the correlations between 
    the aperture and azimuth angles of the same lidar. The cross-correlations betwen different lidars are not taken 
    into account at this stage because each lidar is mesuring a Vlos value independtly.  

Second, we use the statics of Vlos, mean and stdv, jointly with the correlation between lidars to create a second set of multivariated
    distributions which are used to estimate the reconstructed u and v wind veloctiy components and their uncertainties. u anv v components
    inherit the uncertainty in pointing and range from the Vlos uncertainty.

'''

def MCM_Vh_lidar_uncertainty (Lidar,Atmospheric_Scenario,wind_direction,ind_alpha,lidars):
    Vh,U_Vh_MCM                                           = [],[]
    Vlos1_MC_cr2_s,Vlos2_MC_cr2_s,Vlos3_MC_cr2_s          = [],[],[]
    Theta1_cr2_s,Theta2_cr2_s,Theta3_cr2_s,Psi1_cr2_s,Psi2_cr2_s,Psi3_cr2_s,Rho1_cr2_s,Rho2_cr2_s,Rho3_cr2_s=[],[],[],[],[],[],[],[],[]    
    u0,v0,w0 = [],[],[]
    U_Vlos_MCM = {"V1":[],"V2":[],"V3":[]}
    Vlos_corr_MCM={"V12":[],"V13":[],"V23":[]}
    for ind_wind_dir in range(len(wind_direction)):  
        
        ######## Vlos multivariate distribution #####################
        # Multivariate distributions and correlation between the Vlos's: 
        Vlos_corr0 = np.zeros(len(Lidar.optics.scanner.origin))
        Vlos_corr,Vlos_MCM,U_Vlos,Theta_cr,Psi_cr,Rho_cr  =  Vlos_correlations(Lidar,Vlos_corr0,Atmospheric_Scenario,wind_direction, ind_wind_dir,ind_alpha,lidars)
        
        
        #Store data
        for i in range(len(Lidar.optics.scanner.origin)):
            U_Vlos_MCM['V{}'.format(i+1)].append(U_Vlos[i]) 
        Vlos_corr_MCM['V12'].append(Vlos_corr['V1'])
        Vlos_corr_MCM['V13'].append(Vlos_corr['V2'])
        Vlos_corr_MCM['V23'].append(Vlos_corr['V3'])


        ######### Vh multivariate  ####################################                                                                                                                                                                                                                                                                                                                             ## MCM - uv     
        # Covariance matrix    
        Vlos_corrCoef_MCM=[Vlos_corr['V1'],Vlos_corr['V2'],Vlos_corr['V3']]
        cov_MAT_Vh = MultiVar(Lidar, Vlos_corrCoef_MCM     ,  U_Vlos ,          1   ,         1     ,         1 ,            1 ,        'MC2' )
           
        
        if len(Lidar.optics.scanner.origin)==3:    
            
            Theta1_cr2,Theta2_cr2,Theta3_cr2,Psi1_cr2,Psi2_cr2,Psi3_cr2,Rho1_cr2,Rho2_cr2,Rho3_cr2,Vlos1_MC_cr2,Vlos2_MC_cr2,Vlos3_MC_cr2= multivariate_normal.rvs(np.ndarray.tolist(np.concatenate([lidars['Lidar0_Spherical']['theta'] , lidars['Lidar1_Spherical']['theta'] , lidars['Lidar2_Spherical']['theta'], lidars['Lidar0_Spherical']['psi'] , lidars['Lidar1_Spherical']['psi'] , lidars['Lidar2_Spherical']['psi'], lidars['Lidar0_Spherical']['rho'] , lidars['Lidar1_Spherical']['rho'] , lidars['Lidar2_Spherical']['rho'],np.array([np.mean(Vlos_MCM[0])]) , np.array([np.mean(Vlos_MCM[1])]),np.array([np.mean(Vlos_MCM[2])])],axis=0)) , cov_MAT_Vh , Lidar.optics.scanner.N_MC).T
            # # 3D
            u,v,w=Wind_vector(Theta1_cr2,Theta2_cr2,Theta3_cr2,Psi1_cr2,Psi2_cr2,Psi3_cr2,Vlos1_MC_cr2,Vlos2_MC_cr2,Vlos3_MC_cr2)             
            u0.append(np.mean(u))
            v0.append(np.mean(v))           
            w0.append(np.mean(w))
                
            Vh.append(np.sqrt(u**2 + v**2 + w**2))
            U_Vh_MCM.append(np.std(Vh[ind_wind_dir]))
            
             #Storing data
            Vlos_cr_s = [Vlos1_MC_cr2_s.append(Vlos1_MC_cr2),
                         Vlos2_MC_cr2_s.append(Vlos2_MC_cr2),
                         Vlos3_MC_cr2_s.append(Vlos3_MC_cr2)]
            
            Theta_cr_s = [Theta1_cr2_s.append(Theta1_cr2) ,     
                          Theta2_cr2_s.append(Theta2_cr2),
                          Theta3_cr2_s.append(Theta3_cr2)]
            
            Psi_cr_s = [Psi1_cr2_s.append(Psi1_cr2),       
                        Psi2_cr2_s.append(Psi2_cr2),
                        Psi3_cr2_s.append(Psi3_cr2)]
            
            Rho_cr_s = [Rho1_cr2_s.append(Rho1_cr2),
                        Rho2_cr2_s.append(Rho2_cr2),
                        Rho3_cr2_s.append(Rho3_cr2)]
        else:
            # # Multivariate distributions:       
            Theta1_cr2,Theta2_cr2,Psi1_cr2,Psi2_cr2,Rho1_cr2,Rho2_cr2,Vlos1_MC_cr2,Vlos2_MC_cr2= multivariate_normal.rvs((np.concatenate([lidars['Lidar0_Spherical']['theta'] , lidars['Lidar1_Spherical']['theta'] , lidars['Lidar0_Spherical']['psi'] , lidars['Lidar1_Spherical']['psi'] ,  lidars['Lidar0_Spherical']['rho'] , lidars['Lidar1_Spherical']['rho'] , np.array([np.mean(Vlos_MCM[0])]) , np.array([np.mean(Vlos_MCM[1])])],axis=0)), cov_MAT_Vh , Lidar.optics.scanner.N_MC).T
            
            #Storing data
            Vlos_cr_s = [Vlos1_MC_cr2_s.append(Vlos1_MC_cr2),
                         Vlos2_MC_cr2_s.append(Vlos2_MC_cr2)]
           
            Theta_cr_s = [Theta1_cr2_s.append(Theta1_cr2),
                          Theta2_cr2_s.append(Theta2_cr2)]
            
            Psi_cr_s = [Psi1_cr2_s.append(Psi1_cr2),
                        Psi2_cr2_s.append(Psi2_cr2)]
            
            Rho_cr_s = [Rho1_cr2_s.append(Rho1_cr2),
                        Rho2_cr2_s.append(Rho2_cr2)]
            
            # Uncertainty in horizontal velocity
            Vh_num    = np.sqrt( ((Vlos1_MC_cr2 * np.cos(Theta2_cr2))**2 + (Vlos2_MC_cr2 * np.cos(Theta1_cr2))**2) - 2*(Vlos1_MC_cr2 * Vlos2_MC_cr2 * np.cos(Theta1_cr2) * np.cos(Theta2_cr2) * np.cos(Psi1_cr2 - Psi2_cr2)) )
            Vh_denom  = np.cos(Theta1_cr2) * np.cos(Theta2_cr2) * np.sin(Psi1_cr2 - Psi2_cr2)       
            Vh.append(Vh_num / Vh_denom)       
            U_Vh_MCM.append(np.std(Vh[ind_wind_dir]))


    # Store the multivariate distributions
    Mult_param          =  [Vlos1_MC_cr2_s,Vlos2_MC_cr2_s,Vlos3_MC_cr2_s,Theta1_cr2_s,Theta2_cr2_s,Theta3_cr2_s,Psi1_cr2_s,Psi2_cr2_s,Psi3_cr2_s,Rho1_cr2_s,Rho2_cr2_s,Rho3_cr2_s]
    # pdb.set_trace()
    return Vlos_corr_MCM,U_Vlos_MCM , Mult_param  , U_Vh_MCM


#%% ##########################################
##########################################
#Uncertainty of Vlos following GUM model
##########################################
##############################################
#%%

def GUM_Vlos_lidar_uncertainty(Lidar,Atmospheric_Scenario,wind_direction,ind_alpha,lidars):    

    u_V_LOS1Theta1,u_V_LOS1Psi1,u_V_LOS1Rho1,u_V_LOS2Theta2,u_V_LOS2Psi2,u_V_LOS2Rho2,u_V_LOS3Theta3,u_V_LOS3Psi3,u_V_LOS3Rho3=[], [],[],[],[],[],[],[],[]
    U_Vlos_GUM = {"V1":[],"V2":[],"V3":[]}
    Correlation_coeff={"V1":[],"V2":[],"V3":[]}
    Sens_coeff={'V1_theta':[],'V2_theta':[],'V3_theta':[],'V1_psi':[],'V2_psi':[],'V3_psi':[],'V1_rho':[],'V2_rho':[],'V3_rho':[]}
    VL=[]
    Vlos_GUM ={'V1':[],'V2':[],'V3':[]}

    Corrcoef_Vlos=[]
    
    # Tilt angle:
    beta=np.radians(Atmospheric_Scenario.wind_tilt)
    
    for ind_wind_dir in range(len(wind_direction)):  
        
        # VLOS
        Vlos_corr = np.zeros(len(Lidar.optics.scanner.origin))
        U_Vlos_GUM0 =np.zeros(len(Lidar.optics.scanner.origin))
        Ux=MultiVar(Lidar, Vlos_corr,U_Vlos_GUM0    ,1   ,            1          ,1            ,     1,   'GUM1'  )  
        

        # VLOS uncertainty
        # Calculate and store Vlosi
        
        dVlosdtheta,dVlosdpsi,dVlosdrho=[],[],[]
        for i in range(len(Lidar.optics.scanner.origin)):
            H_cr = ( (lidars['Lidar{}_Spherical'.format(i)]['rho'] * np.sin(lidars['Lidar{}_Spherical'.format(i)]['theta']) + Lidar.optics.scanner.origin[i][2]) / Lidar.optics.scanner.Href)
            Vlos_GUM['V{}'.format(i+1)].append(Atmospheric_Scenario.Vref*(H_cr**Atmospheric_Scenario.PL_exp[ind_alpha])*np.cos(lidars['Lidar{}_Spherical'.format(i)]['theta'])*(np.cos(lidars['Lidar{}_Spherical'.format(i)]['psi']-wind_direction[ind_wind_dir])+np.tan(beta)*np.tan(lidars['Lidar{}_Spherical'.format(i)]['theta'])))
      
            # Partial derivatives Vlosi with respect theta, psi and rho
            dVlosdtheta.append(Atmospheric_Scenario.Vref*((H_cr)**Atmospheric_Scenario.PL_exp[ind_alpha]) * (( Atmospheric_Scenario.PL_exp[ind_alpha]*((lidars['Lidar{}_Spherical'.format(i)]['rho']*(np.cos(lidars['Lidar{}_Spherical'.format(i)]['theta']))**2) / (lidars['Lidar{}_Spherical'.format(i)]['rho']*np.sin(lidars['Lidar{}_Spherical'.format(i)]['theta']) + Lidar.optics.scanner.origin[i][2]))-np.sin(lidars['Lidar{}_Spherical'.format(i)]['theta']) ) * ( np.cos(lidars['Lidar{}_Spherical'.format(i)]['psi']-wind_direction[ind_wind_dir]) + np.tan(beta)*np.tan(lidars['Lidar{}_Spherical'.format(i)]['theta']) ) + (np.tan(beta)/np.cos(lidars['Lidar{}_Spherical'.format(i)]['theta'])))    )       
            dVlosdpsi.append(- Atmospheric_Scenario.Vref*((H_cr)**Atmospheric_Scenario.PL_exp[ind_alpha]) * (np.cos(lidars['Lidar{}_Spherical'.format(i)]['theta'])*np.sin(lidars['Lidar{}_Spherical'.format(i)]['psi'] - wind_direction[ind_wind_dir])))          
            dVlosdrho.append(Atmospheric_Scenario.Vref*((H_cr)**Atmospheric_Scenario.PL_exp[ind_alpha]) * Atmospheric_Scenario.PL_exp[ind_alpha]*(np.sin(lidars['Lidar{}_Spherical'.format(i)]['theta']) / (lidars['Lidar{}_Spherical'.format(i)]['rho']*np.sin(lidars['Lidar{}_Spherical'.format(i)]['theta']) + Lidar.optics.scanner.origin[i][2]))*np.cos(lidars['Lidar{}_Spherical'.format(i)]['theta'])*(np.cos(lidars['Lidar{}_Spherical'.format(i)]['psi']-wind_direction[ind_wind_dir])+(np.tan(beta)/np.cos(lidars['Lidar{}_Spherical'.format(i)]['theta']))))
            
            # Store contributions:
            Sens_coeff['V{}_theta'.format(i+1)].append(dVlosdtheta[i])
            Sens_coeff['V{}_psi'.format(i+1)].append(dVlosdpsi[i])
            Sens_coeff['V{}_rho'.format(i+1)].append(dVlosdrho[i])
        

        if len(Lidar.optics.scanner.origin)==3:
            # Influence coefficients matrix for Vlosi uncertainty estimation
            
            Cx = np.array([[dVlosdtheta[0],        np.array([0]),       np.array([0]),        dVlosdpsi[0],    np.array([0]),      np.array([0]),       dVlosdrho[0]  , np.array([0]),   np.array([0])   ,np.array([0]),np.array([0]),np.array([0])],
                           [np.array([0]),        dVlosdtheta[1],       np.array([0]),       np.array([0]),     dVlosdpsi[1],      np.array([0]),       np.array([0]) ,  dVlosdrho[1],   np.array([0])  ,np.array([0]),np.array([0]),np.array([0])],
                           [np.array([0]),        np.array([0]),        dVlosdtheta[2],      np.array([0]),     np.array([0]),      dVlosdpsi[2],       np.array([0])  ,  np.array([0]) , dVlosdrho[2]  ,np.array([0]),np.array([0]),np.array([0])]])
        else:
            Cx = np.array([[dVlosdtheta[0],      np.array([0]),        dVlosdpsi[0],      np.array([0]),          dVlosdrho[0]  ,  np.array([0]), np.array([0])   ,np.array([0])],
                            [ np.array([0]),        dVlosdtheta[1],    np.array([0]),        dVlosdpsi[1],        np.array([0]) ,  dVlosdrho[1], np.array([0])   ,np.array([0])]])
        Cx=Cx[:,:,0]
        # Ouput covariance matrix
        Uy=Cx.dot(Ux).dot(np.transpose(Cx))
        
        # CORRELATIONS Vlos
        Corr_combi = list(itertools.combinations(range(len(Uy)), 2)) # amount of Vlos combinations
        
        

        for i_combi in range(len(Corr_combi)):            
            Correlation_coeff['V{}'.format(i_combi+1)].append(Uy[Corr_combi[i_combi][0]][Corr_combi[i_combi][1]]/np.sqrt(Uy[Corr_combi[i_combi][1]][Corr_combi[i_combi][1]]*Uy[Corr_combi[i_combi][0]][Corr_combi[i_combi][0]]))
        for i in range(len(Lidar.optics.scanner.origin)):
            # U_est ##############
            U_Vlos_GUM["V{}".format(i+1)].append(np.sqrt(Uy[i][i]+ Lidar.lidar_inputs.dataframe['Intrinsic Uncertainty [m/s]']**2+Lidar.optics.scanner.stdv_Estimation[0][0]**2))       
        
            # Store data
            # VL.append(Vlos_GUM["V{}".format(i+1)])
        
        
        Corrcoef_Vlos.append(Uy[0][1]/np.sqrt(Uy[1][1]*Uy[0][0]))

    return(Correlation_coeff, U_Vlos_GUM, Vlos_GUM, Sens_coeff)

#%% ##########################################
##########################################
#Uncertainty of Vh following GUM
##########################################
##############################################
#%%
def GUM_Vh_lidar_uncertainty (Lidar,Atmospheric_Scenario,Correlation_coeff,wind_direction,lidars ,Vlos_GUM,U_Vlos_GUM):
        # Vh Uncertainty
        UUy,U_Vh_GUM=[],[]
        u0,v0,w0=[],[],[]
        Sensitivity_Coefficients={'dV1':[],'dV2':[],'dV3':[],'dV1V2':[],'dV1V3':[],'dV2V3':[]}
        
        for ind_wind_dir in range(len(wind_direction)):  
            if len(Lidar.optics.scanner.origin)==3:
                u,v,w=Wind_vector(lidars['Lidar0_Spherical']['theta'] , lidars['Lidar1_Spherical']['theta'] , lidars['Lidar2_Spherical']['theta'], lidars['Lidar0_Spherical']['psi'] , lidars['Lidar1_Spherical']['psi'] , lidars['Lidar2_Spherical']['psi'], Vlos_GUM['V1'][ind_wind_dir],Vlos_GUM['V2'][ind_wind_dir],Vlos_GUM['V3'][ind_wind_dir])
                
                u0.append(u)
                v0.append(v)
                w0.append(w)
                
                # Definingn coefficients:
                a=1/(2*np.sqrt(u**2 + v**2 + w**2))
                b=(np.cos(lidars['Lidar0_Spherical']['theta']) *np.cos(lidars['Lidar2_Spherical']['theta'])*np.sin(lidars['Lidar1_Spherical']['theta'])*np.sin(lidars['Lidar2_Spherical']['psi']-lidars['Lidar0_Spherical']['psi']) + np.cos(lidars['Lidar1_Spherical']['theta'])*np.cos(lidars['Lidar2_Spherical']['theta'])*np.sin(lidars['Lidar0_Spherical']['theta'])*np.sin(lidars['Lidar1_Spherical']['psi']-lidars['Lidar2_Spherical']['psi']) + np.cos(lidars['Lidar0_Spherical']['theta'])*np.cos(lidars['Lidar1_Spherical']['theta'])*np.sin(lidars['Lidar2_Spherical']['theta'])*np.sin(lidars['Lidar0_Spherical']['psi']-lidars['Lidar1_Spherical']['psi']))
    
                dVwind_Vlos1= (a)*((-2*u*(np.cos(lidars['Lidar1_Spherical']['theta']) * np.sin(lidars['Lidar1_Spherical']['psi'])  * np.sin(lidars['Lidar2_Spherical']['theta']) - np.cos(lidars['Lidar2_Spherical']['theta'])*np.sin(lidars['Lidar2_Spherical']['psi'])*np.sin(lidars['Lidar1_Spherical']['theta']))/b) - 
                                    (2*v*(np.cos(lidars['Lidar2_Spherical']['psi'])  * np.cos(lidars['Lidar2_Spherical']['theta']) * np.sin(lidars['Lidar1_Spherical']['theta']) - np.cos(lidars['Lidar1_Spherical']['psi'])*np.cos(lidars['Lidar1_Spherical']['theta'])*np.sin(lidars['Lidar2_Spherical']['theta']))/b) -
                                    (2*w*np.cos(lidars['Lidar1_Spherical']['theta'])  * np.cos(lidars['Lidar2_Spherical']['theta']) * np.sin(lidars['Lidar2_Spherical']['psi']-lidars['Lidar1_Spherical']['psi']) )/b)
                
                dVwind_Vlos2= (a)*((-2*u*((np.cos(lidars['Lidar2_Spherical']['theta'])*np.sin(lidars['Lidar2_Spherical']['psi'])*np.sin(lidars['Lidar0_Spherical']['theta']) - np.cos(lidars['Lidar0_Spherical']['theta'])*np.sin(lidars['Lidar0_Spherical']['psi'])*np.sin(lidars['Lidar2_Spherical']['theta'])))/b) - 
                                    (2*v*((np.cos(lidars['Lidar0_Spherical']['psi'])*np.cos(lidars['Lidar0_Spherical']['theta'])*np.sin(lidars['Lidar2_Spherical']['theta']) - np.cos(lidars['Lidar2_Spherical']['psi'])*np.cos(lidars['Lidar2_Spherical']['theta'])*np.sin(lidars['Lidar0_Spherical']['theta']))/b)) -
                                    (2*w*(np.cos(lidars['Lidar0_Spherical']['theta']) * np.cos(lidars['Lidar2_Spherical']['theta']) * np.sin(lidars['Lidar0_Spherical']['psi']-lidars['Lidar2_Spherical']['psi'])))/b)
                
                dVwind_Vlos3= (a)*((-2*u*(np.cos(lidars['Lidar0_Spherical']['theta'])*np.sin(lidars['Lidar0_Spherical']['psi'])*np.sin(lidars['Lidar1_Spherical']['theta']) - np.cos(lidars['Lidar1_Spherical']['theta'])* np.sin(lidars['Lidar1_Spherical']['psi'])*np.sin(lidars['Lidar0_Spherical']['theta']))/b )- 
                                    (2*v*((np.cos(lidars['Lidar1_Spherical']['psi'])*np.cos(lidars['Lidar1_Spherical']['theta'])*np.sin(lidars['Lidar0_Spherical']['theta']) - np.cos(lidars['Lidar0_Spherical']['psi'])*np.cos(lidars['Lidar0_Spherical']['theta'])*np.sin(lidars['Lidar1_Spherical']['theta'])))/b) -
                                    (2*w*(np.cos(lidars['Lidar0_Spherical']['theta']) * np.cos(lidars['Lidar1_Spherical']['theta']) * np.sin(lidars['Lidar1_Spherical']['psi']-lidars['Lidar0_Spherical']['psi'])))/b)
                
                #Sensitivity matrix
                Cx = np.array([[np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),dVwind_Vlos1,dVwind_Vlos2,dVwind_Vlos3]])
                
                # Store data:
                # Contributions different sensitivity coefficients
                Sensitivity_Coefficients['dV1'].append((dVwind_Vlos1*U_Vlos_GUM['V1'][ind_wind_dir])**2)
                Sensitivity_Coefficients['dV2'].append((dVwind_Vlos2*U_Vlos_GUM['V2'][ind_wind_dir])**2)
                Sensitivity_Coefficients['dV3'].append((dVwind_Vlos3*U_Vlos_GUM['V3'][ind_wind_dir])**2) 
                Sensitivity_Coefficients['dV1V2'].append(2*dVwind_Vlos1*U_Vlos_GUM['V1'][ind_wind_dir]*dVwind_Vlos2*U_Vlos_GUM['V2'][ind_wind_dir]*Correlation_coeff['V1'][ind_wind_dir])
                Sensitivity_Coefficients['dV1V3'].append(2*dVwind_Vlos1*U_Vlos_GUM['V1'][ind_wind_dir]*dVwind_Vlos3*U_Vlos_GUM['V3'][ind_wind_dir]*Correlation_coeff['V2'][ind_wind_dir])
                Sensitivity_Coefficients['dV2V3'].append(2*dVwind_Vlos2*U_Vlos_GUM['V2'][ind_wind_dir]*dVwind_Vlos3*U_Vlos_GUM['V3'][ind_wind_dir]*Correlation_coeff['V3'][ind_wind_dir])
                
                # Storing UVlos and correlations
                U_VLOS_GUM_list   = [U_Vlos_GUM['V1'][ind_wind_dir],U_Vlos_GUM['V2'][ind_wind_dir],U_Vlos_GUM['V3'][ind_wind_dir]]
                Corrcoef_GUM_list = [Correlation_coeff['V1'][ind_wind_dir],Correlation_coeff['V2'][ind_wind_dir],Correlation_coeff['V3'][ind_wind_dir]]
            else:
                num1 = np.sqrt(((Vlos_GUM['V1'][ind_wind_dir]*np.cos(lidars['Lidar1_Spherical']['theta']))**2)+((Vlos_GUM['V2'][ind_wind_dir]*np.cos(lidars['Lidar0_Spherical']['theta']))**2)-(2*Vlos_GUM['V1'][ind_wind_dir]*Vlos_GUM['V2'][ind_wind_dir]*np.cos(lidars['Lidar0_Spherical']['psi']-lidars['Lidar1_Spherical']['psi'])*np.cos(lidars['Lidar0_Spherical']['theta'])*np.cos(lidars['Lidar1_Spherical']['theta'])))
                den=np.cos(lidars['Lidar0_Spherical']['theta'])*np.cos(lidars['Lidar1_Spherical']['theta'])*np.sin(lidars['Lidar0_Spherical']['psi']-lidars['Lidar1_Spherical']['psi'])
                den=np.cos(lidars['Lidar0_Spherical']['theta'])*np.cos(lidars['Lidar1_Spherical']['theta'])*np.sin(lidars['Lidar0_Spherical']['psi']-lidars['Lidar1_Spherical']['psi'])
                dVh_Vlos1= (1/den)*(1/(num1))*(Vlos_GUM['V1'][ind_wind_dir]*((np.cos(lidars['Lidar1_Spherical']['theta']))**2)-Vlos_GUM['V2'][ind_wind_dir]*np.cos(lidars['Lidar0_Spherical']['theta'])*np.cos(lidars['Lidar1_Spherical']['theta'])*np.cos(lidars['Lidar0_Spherical']['psi']-lidars['Lidar1_Spherical']['psi']))
                dVh_Vlos2= (1/den)*(1/(num1))*(Vlos_GUM['V2'][ind_wind_dir]*((np.cos(lidars['Lidar0_Spherical']['theta']))**2)-Vlos_GUM['V1'][ind_wind_dir]*np.cos(lidars['Lidar0_Spherical']['theta'])*np.cos(lidars['Lidar1_Spherical']['theta'])*np.cos(lidars['Lidar0_Spherical']['psi']-lidars['Lidar1_Spherical']['psi']))
               
                #Sensitivity matrix
                Cx = np.array([[np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),dVh_Vlos1,dVh_Vlos2]])
                                
                # Store Sensitivity coefficients:
                Sensitivity_Coefficients['dV1'].append((dVh_Vlos1*U_Vlos_GUM['V1'][ind_wind_dir])**2)
                Sensitivity_Coefficients['dV2'].append((dVh_Vlos2*U_Vlos_GUM['V2'][ind_wind_dir])**2)
                Sensitivity_Coefficients['dV1V2'].append(2*dVh_Vlos1*U_Vlos_GUM['V1'][ind_wind_dir]*dVh_Vlos2*U_Vlos_GUM['V2'][ind_wind_dir]*Correlation_coeff['V1'][ind_wind_dir])
                
                # Storing UVlos and correlations
                U_VLOS_GUM_list   = [U_Vlos_GUM['V1'][ind_wind_dir],U_Vlos_GUM['V2'][ind_wind_dir]]
                Corrcoef_GUM_list = [Correlation_coeff['V1'][ind_wind_dir]]

            Ux=MultiVar(Lidar, Corrcoef_GUM_list,U_VLOS_GUM_list   ,1   ,            1          ,1            ,     1,   'GUM2'  )
            
            Cx=Cx[:,:,0] # Cutting dimensions
            # Output covariance matrix
            UyVh = np.array(Cx).dot(Ux).dot(np.transpose(Cx))
            UUy.append(UyVh)
            U_Vh_GUM.append(np.sqrt(UyVh[0]))        
        # pdb.set_trace()
        return(U_Vh_GUM,Sensitivity_Coefficients)
    
    
    
#%% Wind direction uncertainties
def U_WindDir_MC(wind_direction,Mult_param):
    """.
    
    Calculates wind direction. Location: Qlunc_Help_standAlone.py
    
    Parameters
    ----------    
    * correlated distributions theta, psi and rho
    
    * wind direction [degrees]
     
    * ind_wind_dir: index for looping
    
    * $H_{ref}$: Reference height  at which $V_{ref}$ is taken [m]
    
    * $V_{ref}$: reference velocity taken from an external sensor (e.g. cup anemometer) [m/s]
    
    * alpha: power law exponent [-] 
    
    * Hl: Lidar height [m]
  
    Returns
    -------    
    u and v wind speed components 
    """
    
    
    Vlos1,Vlos2,Theta1,Theta2,Psi1,Psi2,Rho1,Rho2=Mult_param
    #Wind direction
    U_Wind_direction=[]
    Wind_dir=[]
    for ind_wind_dir in range(len(wind_direction)):
        W_D = (np.arctan((Vlos1[ind_wind_dir]*np.cos(Theta2[ind_wind_dir])*np.cos(Psi2[ind_wind_dir])-Vlos2[ind_wind_dir]*np.cos(Theta1[ind_wind_dir])*np.cos(Psi1[ind_wind_dir]))/(-Vlos1[ind_wind_dir]*np.cos(Theta2[ind_wind_dir])*np.sin(Psi2[ind_wind_dir])+Vlos2[ind_wind_dir]*np.cos(Theta1[ind_wind_dir])*np.sin(Psi1[ind_wind_dir]))))
        U_Wind_direction.append(np.degrees(np.std(W_D)))
    return (U_Wind_direction)
    
#%% U wind direction GUM
def U_WindDir_GUM(Lidar,Atmospheric_Scenario,Corrcoef_Vlos_GUM,wind_direction,theta1,psi1,rho1,theta2,psi2,rho2,u_theta1 ,u_psi1,u_rho1,u_theta2 ,u_psi2,u_rho2,Vlos1,Vlos2,U_Vlos1,U_Vlos2):
    """.
    
    Calculates wind direction. Location: Qlunc_Help_standAlone.py
    
    Parameters
    ----------    
    * correlated distributions theta, psi and rho
    
    * wind direction [degrees]
     
    * ind_wind_dir: index for looping
    
    * $H_{ref}$: Reference height  at which $V_{ref}$ is taken [m]
    
    * $V_{ref}$: reference velocity taken from an external sensor (e.g. cup anemometer) [m/s]
    
    * alpha: power law exponent [-] 
    
    * Hl: Lidar height [m]
  
    Returns
    -------    
     wind direction uncertainty against wind direction
    """
          
    U_u_GUM,U_v_GUM,U_Vh_MCM,U_uv_GUM,U_wind_dir,r_uv=[],[],[],[],[],[]
    dWinDir_Vlos1T,dWinDir_Vlos2T,dWinDir_theta1T,dWinDir_theta2T,dWinDir_psi1T,dWinDir_psi2T=[],[],[],[],[],[]
    for ind_wind_dir in range(len(wind_direction)):
       
        A =  Vlos1[ind_wind_dir]*np.cos(theta2)*np.cos(psi2)-Vlos2[ind_wind_dir]*np.cos(theta1)*np.cos(psi1)        
        B = -Vlos1[ind_wind_dir]*np.cos(theta2)*np.sin(psi2)+Vlos2[ind_wind_dir]*np.cos(theta1)*np.sin(psi1)
        X =  1/(1+(A/B)**2)
        
        
        # Matrix of sensitivity coefficients
        
        dWinDir_Vlos1   =  X*np.cos(theta2)*(B*np.cos(psi2)+A*np.sin(psi2))/B**2
        dWinDir_Vlos2   = -X*np.cos(theta1)*(B*np.cos(psi1)+A*np.sin(psi1))/B**2
        
        dWinDir_theta1  =  X*((Vlos2[ind_wind_dir]*np.sin(theta1)*np.cos(psi1)*B+A*Vlos2[ind_wind_dir]*np.sin(theta1)*np.sin(psi1))/(B**2))
        dWinDir_theta2  =  X*((-Vlos1[ind_wind_dir]*np.sin(theta2)*np.cos(psi2)*B-A*Vlos1[ind_wind_dir]*np.sin(theta2)*np.sin(psi2))/(B**2))
        
        dWinDir_psi1    =  X*(Vlos2[ind_wind_dir]*np.cos(theta1)*np.sin(psi1)*B-A*Vlos2[ind_wind_dir]*np.cos(theta1)*np.cos(psi1))/(B**2)
        dWinDir_psi2    =  X*(-Vlos1[ind_wind_dir]*np.cos(theta2)*np.sin(psi2)*B+A*Vlos1[ind_wind_dir]*np.cos(theta2)*np.cos(psi2))/(B**2)

        UxWinDir=MultiVar(Lidar,Corrcoef_Vlos_GUM[ind_wind_dir], U_Vlos1[ind_wind_dir],U_Vlos2[ind_wind_dir],  [0,0], [0,0],  [0,0]  ,  0,0,0,1, 'GUM2' )
        CxWinDir=[0,0,0,0,0,0,dWinDir_Vlos1,dWinDir_Vlos2]           
        UyWinDir=np.array(CxWinDir).dot(UxWinDir).dot(np.transpose(CxWinDir))
        
        # Data storage:
        dWinDir_Vlos1T.append(dWinDir_Vlos1)
        dWinDir_Vlos2T.append(dWinDir_Vlos2)
        dWinDir_theta1T.append(dWinDir_theta1)        
        dWinDir_theta2T.append(dWinDir_theta2)
        dWinDir_psi1T.append(dWinDir_psi1)
        dWinDir_psi2T.append(dWinDir_psi2)        
                
        U_wind_dir.append(np.degrees(np.sqrt(UyWinDir)))
    
    return (U_wind_dir)

    
  
########### Intrinsic lidar uncertainty

def U_intrinsic(Lidar,List_Unc_lidar,Qlunc_yaml_inputs):   
    V_ref            = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']      # Reference voltaje ADC
    lidar_wavelength = Qlunc_yaml_inputs['Components']['Laser']['Wavelength'] # wavelength of the laser source.
    fd               = 2*V_ref/lidar_wavelength  # Doppler frequency corresponding to Vref
    corr_wavelength_fd=1
    
    # Analytical solution:    
    u_intrinsic = np.sqrt((fd*List_Unc_lidar['Stdv wavelength [m]']/2)**2+(Qlunc_yaml_inputs['Components']['Laser']['Wavelength']*List_Unc_lidar['Stdv Doppler f_peak [Hz]']/2)**2+(fd*Qlunc_yaml_inputs['Components']['Laser']['Wavelength']*List_Unc_lidar['Stdv Doppler f_peak [Hz]']*List_Unc_lidar['Stdv wavelength [m]'])*corr_wavelength_fd/2) 
    return u_intrinsic


def Wind_vector(theta1,theta2,theta3,psi1,psi2,psi3, Vlos1,Vlos2,Vlos3):
    # # 3D
    # pdb.set_trace()
    
    
    u = - (Vlos3* (np.cos(theta1)*np.sin(psi1)*np.sin(theta2) - np.cos(theta2)* np.sin(psi2)*np.sin(theta1))  +                           
           Vlos2 *(np.cos(theta3)*np.sin(psi3)*np.sin(theta1) - np.cos(theta1)*np.sin(psi1)*np.sin(theta3)) +                 
           Vlos1 *(np.cos(theta2)*np.sin(psi2)*np.sin(theta3) - np.cos(theta3)*np.sin(psi3)*np.sin(theta2)) ) /(np.cos(theta1) *np.cos(theta3)*np.sin(theta2)*np.sin(psi3-psi1)+ 
           np.cos(theta2)*np.cos(theta3)*np.sin(theta1)*np.sin(psi2-psi3)          + 
           np.cos(theta1)*np.cos(theta2)*np.sin(theta3)*np.sin(psi1-psi2))
    
    
    v = - (Vlos1* (np.cos(psi3)*np.cos(theta3)*np.sin(theta2) - np.cos(psi2)*np.cos(theta2)*np.sin(theta3)) +                 
           Vlos2* (np.cos(psi1)*np.cos(theta1)*np.sin(theta3) - np.cos(psi3)*np.cos(theta3)*np.sin(theta1)) +                 
           Vlos3* (np.cos(psi2)*np.cos(theta2)*np.sin(theta1) - np.cos(psi1)*np.cos(theta1)*np.sin(theta2)))/(np.cos(theta1) *np.cos(theta3)*np.sin(theta2)*np.sin(psi3-psi1)+ 
           np.cos(theta2)*np.cos(theta3)*np.sin(theta1)*np.sin(psi2-psi3)          + 
           np.cos(theta1)*np.cos(theta2)*np.sin(theta3)*np.sin(psi1-psi2))
    
    
    w =- (Vlos1 * np.cos(theta2) * np.cos(theta3) * np.sin(psi3-psi2)   +                
          Vlos2 * np.cos(theta1) * np.cos(theta3) * np.sin(psi1-psi3)   +                 
          Vlos3 * np.cos(theta1) * np.cos(theta2) * np.sin(psi2-psi1))  /(np.cos(theta1) *np.cos(theta3)*np.sin(theta2)*np.sin(psi3-psi1)+ 
           np.cos(theta2)*np.cos(theta3)*np.sin(theta1)*np.sin(psi2-psi3)          + 
           np.cos(theta1)*np.cos(theta2)*np.sin(theta3)*np.sin(psi1-psi2))                                                               
    # pdb.set_trace()                        
    return u,v,w



#%% 3D velocity vector
'''
u1 = -((-Vlos3 *np.cos(theta2)* np.sin(psi2)* np.sin(theta1) + 
        Vlos2* np.cos(theta3) *np.sin(psi3) *np.sin(theta1) + 
        Vlos3 *np.cos(theta1) *np.sin(psi1) *np.sin(theta2) - 
        Vlos1* np.cos(theta3)* np.sin(psi3) *np.sin(theta2) - 
        Vlos2* np.cos(theta1) *np.sin(psi1) *np.sin(theta3) + 
        Vlos1 *np.cos(theta2)* np.sin(psi2) *np.sin(theta3))/
       (np.cos(psi3) *np.cos(theta2) *np.cos(theta3) *np.sin(psi2) *np.sin(theta1) - 
        np.cos(psi2) *np.cos(theta2) *np.cos(theta3) *np.sin(psi3)* np.sin(theta1) - 
        np.cos(psi3) *np.cos(theta1) *np.cos(theta3) *np.sin(psi1)* np.sin(theta2) + 
        np.cos(psi1)* np.cos(theta1) *np.cos(theta3) *np.sin(psi3)* np.sin(theta2) + 
        np.cos(psi2)* np.cos(theta1) *np.cos(theta2)* np.sin(psi1)* np.sin(theta3) - 
        np.cos(psi1) *np.cos(theta1) *np.cos(theta2)* np.sin(psi2)* np.sin(theta3)))
  
v1 = -((Vlos3 *np.cos(psi2) *np.cos(theta2)* np.sin(theta1) - 
        Vlos2 *np.cos(psi3)* np.cos(theta3)* np.sin(theta1) - 
        Vlos3 *np.cos(psi1)* np.cos(theta1) *np.sin(theta2) + 
        Vlos1 *np.cos(psi3)* np.cos(theta3) *np.sin(theta2) + 
        Vlos2 *np.cos(psi1)* np.cos(theta1) *np.sin(theta3) - 
        Vlos1 *np.cos(psi2)* np.cos(theta2) *np.sin(theta3))/(np.cos(psi3) *np.cos(theta2) *np.cos(theta3) *np.sin(psi2) *np.sin(theta1) - 
        np.cos(psi2)* np.cos(theta2) *np.cos(theta3)*np.sin(psi3)* np.sin(theta1) - 
        np.cos(psi3) *np.cos(theta1) *np.cos(theta3)* np.sin(psi1)* np.sin(theta2) + 
        np.cos(psi1) *np.cos(theta1) *np.cos(theta3)* np.sin(psi3) *np.sin(theta2) + 
        np.cos(psi2) *np.cos(theta1) *np.cos(theta2)* np.sin(psi1) *np.sin(theta3) - 
        np.cos(psi1) *np.cos(theta1) *np.cos(theta2)* np.sin(psi2) *np.sin(theta3)))
  
w1 = -((-Vlos3* np.cos(psi2) *np.cos(theta1)* np.cos(theta2) *np.sin(psi1) + 
        Vlos2* np.cos(psi3) *np.cos(theta1)* np.cos(theta3)* np.sin(psi1) + 
        Vlos3 *np.cos(psi1) *np.cos(theta1) *np.cos(theta2) *np.sin(psi2) - 
        Vlos1 *np.cos(psi3) *np.cos(theta2) *np.cos(theta3) *np.sin(psi2) - 
        Vlos2 *np.cos(psi1) *np.cos(theta1) *np.cos(theta3) *np.sin(psi3) + 
        Vlos1 *np.cos(psi2) *np.cos(theta2) *np.cos(theta3) *np.sin(psi3))/(np.cos(psi3) *np.cos(theta2)* np.cos(theta3)* np.sin(psi2) *np.sin(theta1) - 
        np.cos(psi2)* np.cos(theta2) *np.cos(theta3) *np.sin(psi3) *np.sin(theta1) - 
        np.cos(psi3) *np.cos(theta1) *np.cos(theta3) *np.sin(psi1) *np.sin(theta2) + 
        np.cos(psi1) *np.cos(theta1) *np.cos(theta3) *np.sin(psi3) *np.sin(theta2) + 
        np.cos(psi2) *np.cos(theta1) *np.cos(theta2) *np.sin(psi1) *np.sin(theta3) - 
        np.cos(psi1) *np.cos(theta1) *np.cos(theta2) *np.sin(psi2) *np.sin(theta3)))
'''
'''
u = -(-Vlos3* np.cos(theta1) *np.sin(psi1)* np.sin(psi2) + 
            Vlos3 *np.cos(theta2) *np.sin(psi1) *np.sin(psi2) + 
            Vlos2 *np.cos(theta1) *np.sin(psi1)* np.sin(psi3) - 
            Vlos2 *np.cos(theta3)* np.sin(psi1)* np.sin(psi3) - 
            Vlos1 *np.cos(theta2) *np.sin(psi2) *np.sin(psi3) + 
            Vlos1 *np.cos(theta3) *np.sin(psi2) *np.sin(psi3))/(np.cos(psi3) *np.cos(
              theta1) *np.cos(theta3) *np.sin(psi1)* np.sin(psi2) - 
            np.cos(psi3) *np.cos(theta2)* np.cos(theta3) *np.sin(psi1) *np.sin(psi2) - 
            np.cos(psi2) *np.cos(theta1)* np.cos(theta2) *np.sin(psi1) *np.sin(psi3) + 
            np.cos(psi2) *np.cos(theta2)* np.cos(theta3) *np.sin(psi1) *np.sin(psi3) + 
            np.cos(psi1) *np.cos(theta1)* np.cos(theta2) *np.sin(psi2) *np.sin(psi3) - 
            np.cos(psi1) *np.cos(theta1)* np.cos(theta3) *np.sin(psi2)* np.sin(psi3))

v =-(Vlos3 *np.cos(psi2) *np.cos(theta2) *np.sin(psi1) - 
            Vlos2 *np.cos(psi3) *np.cos(theta3) *np.sin(psi1) - 
            Vlos3 *np.cos(psi1) *np.cos(theta1) *np.sin(psi2) + 
            Vlos1 *np.cos(psi3) *np.cos(theta3) *np.sin(psi2) + 
            Vlos2 *np.cos(psi1) *np.cos(theta1) *np.sin(psi3) - 
            Vlos1 *np.cos(psi2) *np.cos(theta2) *np.sin(psi3))/(-np.cos(psi3) *np.cos(
              theta1)* np.cos(theta3) *np.sin(psi1) *np.sin(psi2) + 
            np.cos(psi3) *np.cos(theta2) *np.cos(theta3) *np.sin(psi1) *np.sin(psi2) + 
            np.cos(psi2) *np.cos(theta1) *np.cos(theta2) *np.sin(psi1) *np.sin(psi3) - 
            np.cos(psi2) *np.cos(theta2) *np.cos(theta3) *np.sin(psi1) *np.sin(psi3) - 
            np.cos(psi1) *np.cos(theta1) *np.cos(theta2) *np.sin(psi2) *np.sin(psi3) + 
            np.cos(psi1) *np.cos(theta1) *np.cos(theta3) *np.sin(psi2) *np.sin(psi3))

w=-(-Vlos3 *np.cos(psi2) *np.cos(theta1)* np.cos(theta2) *np.sin(psi1) + 
            Vlos2* np.cos(psi3) *np.cos(theta1)* np.cos(theta3) *np.sin(psi1) + 
            Vlos3 *np.cos(psi1) *np.cos(theta1)* np.cos(theta2) *np.sin(psi2) - 
            Vlos1 *np.cos(psi3) *np.cos(theta2)* np.cos(theta3) *np.sin(psi2) - 
            Vlos2 *np.cos(psi1) *np.cos(theta1)* np.cos(theta3) *np.sin(psi3) + 
            Vlos1 *np.cos(psi2) *np.cos(theta2)* np.cos(theta3) *np.sin(
              psi3))/(-np.cos(psi3)* np.cos(theta1)* np.cos(theta3) *np.sin(psi1) *np.sin(
              psi2) + np.cos(psi3) *np.cos(theta2) *np.cos(theta3)* np.sin(psi1) *np.sin(
              psi2) + np.cos(psi2) *np.cos(theta1) *np.cos(theta2) *np.sin(psi1) *np.sin(
              psi3) - np.cos(psi2) *np.cos(theta2) *np.cos(theta3) *np.sin(psi1) *np.sin(
              psi3) - np.cos(psi1) *np.cos(theta1) *np.cos(theta2) *np.sin(psi2) *np.sin(
              psi3) + np.cos(psi1) *np.cos(theta1)* np.cos(theta3) *np.sin(psi2) *np.sin(
              psi3))

'''
#%% MINE

   
    # u =  (Vlos3* (np.cos(theta1) - np.cos(theta2)) *np.sin(psi1)* np.sin(psi2)  +                 
    #       Vlos2 *(np.cos(theta3) - np.cos(theta1)) *np.sin(psi1)* np.sin(psi3)  +                 
    #       Vlos1 *(np.cos(theta2) - np.cos(theta3)) *np.sin(psi2) *np.sin(psi3)) /(np.cos(theta1) *np.cos(theta3)*np.sin(psi2)*np.sin(psi1-psi3)+ 
    #       np.cos(theta2)*np.cos(theta3)*np.sin(psi1)*np.sin(psi3-psi2)          + 
    #       np.cos(theta1)*np.cos(theta2)*np.sin(psi3)*np.sin(psi2-psi1))
    
    
    # v =  (Vlos1* (np.cos(psi3)*np.cos(theta3)*np.sin(psi2) - np.cos(psi2)*np.cos(theta2)*np.sin(psi3)) +                 
    #       Vlos2* (np.cos(psi1)*np.cos(theta1)*np.sin(psi3) - np.cos(psi3)*np.cos(theta3)*np.sin(psi1)) +                 
    #       Vlos3* (np.cos(psi2)*np.cos(theta2)*np.sin(psi1) - np.cos(psi1)*np.cos(theta1)*np.sin(psi2)))/ (np.cos(theta1) *np.cos(theta3)*np.sin(psi2)*np.sin(psi1-psi3) + 
    #       np.cos(theta2)*np.cos(theta3)*np.sin(psi1)*np.sin(psi3-psi2)   + 
    #       np.cos(theta1)*np.cos(theta2)*np.sin(psi3)*np.sin(psi2-psi1))
    
    
    # w = (Vlos1 * np.cos(theta2) * np.cos(theta3) * np.sin(psi3-psi2)   +                
    #      Vlos2 * np.cos(theta1) * np.cos(theta3) * np.sin(psi1-psi3)   +                 
    #      Vlos3 * np.cos(theta1) * np.cos(theta2) * np.sin(psi2-psi1))  / (np.cos(theta1) *np.cos(theta3)*np.sin(psi2)*np.sin(psi1-psi3) + 
    #      np.cos(theta2)*np.cos(theta3)*np.sin(psi1)*np.sin(psi3-psi2)  + 
    #      np.cos(theta1)*np.cos(theta2)*np.sin(psi3)*np.sin(psi2-psi1)) 