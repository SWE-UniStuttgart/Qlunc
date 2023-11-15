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
        x.append(rho[i]*np.cos(theta[i])*np.cos(phi[i]))
        y.append(rho[i]*np.cos(theta[i])*np.sin(phi[i]) )
        z.append(rho[i]*np.sin(theta[i]) )
    return(np.around(x,5),np.around(y,5),np.around(z,5))

def cart2sph(x,y,z): 
    rho=[]
    theta=[]
    phi=[]
    for ind in range(len(z)):
        rho.append(np.sqrt(x[ind]**2+y[ind]**2+z[ind]**2))
        if z[ind]<0:
            theta.append(-math.acos(np.sqrt(x[ind]**2+y[ind]**2)/np.sqrt(x[ind]**2+y[ind]**2+z[ind]**2)))
        elif z[ind]>=0:
            theta.append(math.acos(np.sqrt(x[ind]**2+y[ind]**2)/np.sqrt(x[ind]**2+y[ind]**2+z[ind]**2)))
        phi.append(math.atan2(y[ind],x[ind]))
        
        # if z[ind]>0:
        #         phi.append(np.arctan(np.sqrt(x[ind]**2+y[ind]**2)/z[ind]))
        # elif z[ind]==0:
        #         phi.append(np.array(np.pi/2))
        # elif z[ind]<0:
        #         phi.append((np.pi)+(np.arctan(np.sqrt(x[ind]**2+y[ind]**2)/z[ind])))
        # LOVE U Lucia!!
        # if x[ind]>0:
        #     if  y[ind]>=0:
        #         theta.append(np.arctan(y[ind]/x[ind]))            
        #     elif  y[ind]<0:
        #         theta.append((2.0*np.pi)+(np.arctan(y[ind]/x[ind])))           
        # elif x[ind]<0:
        #     theta.append((np.pi)+(np.arctan(y[ind]/x[ind])))            
        # elif x[ind]==0:
        #         theta.append(np.pi/2.0*(np.sign(y[ind])))
    # print(np.degrees(theta))
    return(np.array(rho),np.array(theta),np.array(phi)) # foc_dist, aperture angle, azimuth
#%% NDF function

def to_netcdf(DataXarray,Qlunc_yaml_inputs,Lidar,Atmospheric_Scenario):
    #DataXarray=Lidar.lidar_inputs.dataframe
    """.
    
    Save the project to an netcdf file - Location: Qlunc_Help_standAlone.py
    
    Parameters
    ----------    
    * DataXarray
        Data frame containing uncertainties of the lidar in xarray format. The name of the project is specify in the input yaml file.
        Change the name to create a different project. Otherwise the data is appended to the existing file. To read the 
        netndf file:
            xr.open_dataarray('C:/Users/fcosta/SWE_LOCAL/GIT_Qlunc/Projects/' + '<name_of_the_project>.nc')
        
    Returns
    -------    
    .netcdf file
    
    """
    if os.path.isfile('./Projects/' + Qlunc_yaml_inputs['Project']+ '.nc'):
        # Read the new lidar data
        Lidar.lidar_inputs.dataframe['Lidar']
        # time      = Atmospheric_Scenario.time 
        names     = [Lidar.LidarID]
        component = [i for i in DataXarray.keys()]
        data      = [ii for ii in DataXarray.values()]
        df_read   = xr.open_dataarray('./Projects/' + Qlunc_yaml_inputs['Project']+ '.nc')
        
        # Creating the new Xarray:
        dr = xr.DataArray(data,
                          coords = [component,names],
                          dims   = ('Components','Names'))
        
        # Concatenate data from different lidars
        df = xr.concat([df_read,dr],dim='Names')
        df_read.close()
        os.remove('./Projects/' +  Qlunc_yaml_inputs['Project']+ '.nc')
        df.to_netcdf('./Projects/'+ Qlunc_yaml_inputs['Project']+ '.nc','w')
    else:
                
        names     = [Lidar.LidarID]
        component = [i for i in DataXarray.keys()]
        # time      = [Atmospheric_Scenario.time ] 
        data      =[ii for ii in DataXarray.values()]
        df = xr.DataArray(data,
                          coords = [component,names],
                          dims   = ('Components','Names'))
        if not os.path.exists('./Projects'):
            os.makedirs('./Projects')       
        df.to_netcdf('./Projects/'+ Qlunc_yaml_inputs['Project']+ '.nc','w')
        return df
        
        # READ netcdf FILE.
        # da=xr.open_dataarray('C:/Users/fcosta/SWE_LOCAL/GIT_Qlunc/Projects/' + 'Gandia.nc')

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


#%% Creates a sphere of radius equal to the estimates error distance around the lidar theoretical measured point

def sample_sphere(r,npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    xi,yi,zi=r*vec
    return xi,yi,zi



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

#%% Define meshgrid for the errors in pointing accuracy and focus range
def mesh (theta,psi,rho):
    box=np.meshgrid(theta,psi,rho)
    # Get coordinates of the points on the grid
    box_positions = np.vstack(map(np.ravel, box))
    theta=box_positions[0]
    psi=box_positions[1]
    rho=box_positions[2]
    # fig = plt.figure()

    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(theta, psi, rho)

    # ax.set_xlabel('theta')
    # ax.set_ylabel('psi')
    # ax.set_zlabel('rho')
    # plt.show()
    return theta,psi,rho,box

   # LoveU LU!


def VLOS_param (Lidar,rho,theta,psi,u_theta1,u_psi1,u_rho1,N_MC,Hl,Vref,Href,alpha,wind_direction_TEST,ind_wind_dir):
    #####################################
    #####################################
     # HARD CODED  Important HARD CODED  Need solution!!!
    ######################################
    #######################################
    wind_direction_TEST = np.radians([0])
    # wind_tilt_TEST      = np.radians([0])
    
    #If want to vary range    
    if len (rho) !=1:
        rho_TEST   = rho
        theta_TEST = theta*np.ones(len(rho_TEST))
        psi_TEST   = psi*np.ones(len(rho_TEST))
        u_theta1   = 0
        u_psi1   = 0
        ind_i = theta_TEST
    elif len(theta)!=1:
        theta_TEST = theta
        rho_TEST   = rho[0]*np.ones(len(theta_TEST))
        psi_TEST   = psi*np.ones(len(theta_TEST))
        u_rho1   = 0
        u_psi1   = 0
        ind_i = rho_TEST
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
    cov_MAT    =  [[u_theta1**2 ,           u_theta1*u_psi1*0   ,u_theta1*u_rho1*0],
                   [u_theta1*u_psi1*0 ,     u_psi1**2           ,u_psi1*u_rho1*0   ],
                   [u_theta1*u_rho1*0 ,     u_psi1*u_rho1*0     ,u_rho1**2         ]]
   

    U_VLOS1=[]
    U_VLOS_T=U_VLOS_MC(Lidar,cov_MAT,theta_TEST,psi_TEST,rho_TEST,Hl,Href,alpha,wind_direction_TEST,Vref,0,U_VLOS1)
    # pdb.set_trace()    
    #Store results
    U_VLOS_T_MC.append(np.array(U_VLOS_T))
    
    
    # GUM method
    U_VLOS_T_GUM.append(U_VLOS_GUM (Lidar,theta_TEST,psi_TEST,rho_TEST,u_theta1,u_psi1,u_rho1,Hl,Vref,Href,alpha,wind_direction_TEST,0)) # For an heterogeneous flow (shear))  
    # pdb.set_trace()
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
         Theta1_cr,Psi1_cr,Rho1_cr= multivariate_normal.rvs([theta[ind_0],psi[ind_0],rho[ind_0]], cov_MAT,Lidar.optics.scanner.N_MC).T   
         
         A=(Hl+(np.sin(Theta1_cr)*Rho1_cr))/Href   
         VLOS1 = Vref*(A**alpha)*(np.cos(Theta1_cr)*np.cos(Psi1_cr-wind_direction[ind_wind_dir])) #-np.sin(theta_corr[0][ind_npoints])*np.tan(wind_tilt[ind_npoints])
         
         U_VLOS1.append(np.nanstd(VLOS1))
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
    
    H_t1 = ((rho1*np.sin(theta1)+Hl)/Href)
    U_Vlos1_GUM=[]
    # for ind_alpha in range(len(alpha)):
    # VLOS uncertainty
    # Calculate and store Vlosi
    Vlos1_GUM = Vref*(H_t1**alpha)*np.cos(theta1)*np.cos(psi1-wind_direction[ind_wind_dir])
    # Vlos2_GUM = Vref*(np.sign(H_t2)*((abs(H_t2))**alpha))*np.cos(theta2)*np.cos(psi2-wind_direction[ind_wind_dir])    

    
    # Partial derivatives Vlosi with respect theta, psi and rho
    dVlos1dtheta1   =     Vref*(H_t1**alpha)*(alpha*((rho1*(np.cos(theta1))**2)/(rho1*np.sin(theta1)+Hl))-np.sin(theta1))*np.cos(psi1-wind_direction[ind_wind_dir])
    dVlos1dpsi1     =   - Vref*(H_t1**alpha)*(np.cos(theta1)*np.sin(psi1-wind_direction[ind_wind_dir]))
    dVlos1drho1     =     Vref*(H_t1**alpha)*alpha*(np.sin(theta1)/(rho1*np.sin(theta1)+Hl))*np.cos(theta1)*np.cos(psi1-wind_direction[ind_wind_dir])
    # pdb.set_trace()
    Ux=MultiVar(Lidar, 0,      0,     0,   [u_theta1,0],   [u_psi1,0],  [u_rho1,0]        ,1   ,            1          ,1            ,     1,   'GUM2'  )    
    
    # Inputs' correlation matrix for Vlosi uncertainty estimation
    # theta1_rho1_corr=0
    # psi1_rho1_corr=0
    # Ux = np.array([[      u_theta1**2,                       0 ,   u_psi1*u_theta1*psi1_theta1_corr ,    0 ,   u_rho1*u_theta1*theta1_rho1_corr ,    0 ],
    #                [            0 ,                          0,                    0 ,                   0 ,                  0,                     0 ],
    #                [u_theta1*u_psi1*psi1_theta1_corr  ,      0 ,                u_psi1**2,               0 ,       u_rho1*u_psi1*psi1_rho1_corr ,    0 ],
    #                [            0 ,                          0,                     0 ,                  0,                    0 ,                   0 ],
    #                [u_theta1*u_rho1*theta1_rho1_corr ,       0 ,       u_psi1*u_rho1*psi1_rho1_corr ,    0,                   u_rho1**2,             0 ],
    #                [              0,                         0 ,                     0 ,                 0 ,                   0 ,                   0 ]])
    
    
    # Store data
    Cont_Theta.append(dVlos1dtheta1)
    Cont_Psi.append(dVlos1dpsi1)
    Cont_Rho.append(dVlos1drho1)
    # Influence coefficients matrix for Vlosi uncertainty estimation
    Cx = np.array([[dVlos1dtheta1  ,          0      ,  dVlos1dpsi1  ,      0        ,  dVlos1drho1  ,  0,   0,  0   ],
                   [       0       ,          0  ,             0        ,   0  ,                0    ,  0 ,  0  ,0   ]])     
    # Ouputs covariance matrix
    Uy=Cx.dot(Ux).dot(np.transpose(Cx))
    
    # Uncertainty of Vlosi. Here we account for rho, theta and psi uncertainties and their correlations.
    U_Vlos1_GUM.append((np.sqrt(Uy[0][0])))
    # U_Vlos2_GUM.append(np.sqrt(Uy[1][1]))

    return([U_Vlos1_GUM,Cont_Theta,Cont_Psi,Cont_Rho])



#%%
#############################
#####CovarainceMatrix
#############################

'''
This function calculated the covariance matrix
'''
def MultiVar (Lidar,Vlos_corrcoeff, U_Vlos1,U_Vlos2,  theta_stds, psi_stds,  rho_stds,autocorr_theta,autocorr_psi,autocorr_rho,autocorr_V,mode ):
    # Covariance Matrix:
        # psi1_theta1_corr      = Lidar.optics.scanner.correlations[3]

        # pdb.set_trace()
        if mode=='GUM1':
            # pdb.set_trace()
            psi1_theta1_corr      = Lidar.optics.scanner.correlations[3]
            psi2_theta2_corr      = Lidar.optics.scanner.correlations[6]
            rho1_rho2_corr        = Lidar.optics.scanner.correlations[2]
            psi1_psi2_corr        = Lidar.optics.scanner.correlations[0]
            theta1_theta2_corr    = Lidar.optics.scanner.correlations[1]
            psi1_theta2_corr      = Lidar.optics.scanner.correlations[4]
            psi2_theta1_corr      = Lidar.optics.scanner.correlations[5]
            u_Vlos1_Vlos2_corr    = 0#Vlos_corrcoeff#
        
        elif mode=='GUM2':
            psi1_theta1_corr      = Lidar.optics.scanner.correlations[3]
            psi2_theta2_corr      = Lidar.optics.scanner.correlations[6] 
            rho1_rho2_corr        = Lidar.optics.scanner.correlations[2]
            psi1_psi2_corr        = Lidar.optics.scanner.correlations[0]
            theta1_theta2_corr    = Lidar.optics.scanner.correlations[1]
            psi1_theta2_corr      = Lidar.optics.scanner.correlations[4]
            psi2_theta1_corr      = Lidar.optics.scanner.correlations[5]
            u_Vlos1_Vlos2_corr    = Vlos_corrcoeff#Lidar.optics.scanner.correlations[7]
        elif mode=='MC1':
            
            psi1_theta1_corr      = Lidar.optics.scanner.correlations[3]
            psi2_theta2_corr      = Lidar.optics.scanner.correlations[6]
            rho1_rho2_corr        = Lidar.optics.scanner.correlations[2]
            psi1_psi2_corr        = Lidar.optics.scanner.correlations[0]
            theta1_theta2_corr    = Lidar.optics.scanner.correlations[1]
            psi1_theta2_corr      = Lidar.optics.scanner.correlations[4]
            psi2_theta1_corr      = Lidar.optics.scanner.correlations[5]
            u_Vlos1_Vlos2_corr    = 0#Vlos_corrcoeff#

        elif mode=='MC2':            
            psi1_theta1_corr      = Lidar.optics.scanner.correlations[3]
            psi2_theta2_corr      = Lidar.optics.scanner.correlations[6]
            rho1_rho2_corr        = Lidar.optics.scanner.correlations[2]
            psi1_psi2_corr        = Lidar.optics.scanner.correlations[0]
            theta1_theta2_corr    = Lidar.optics.scanner.correlations[1]
            psi1_theta2_corr      = Lidar.optics.scanner.correlations[4]
            psi2_theta1_corr      = Lidar.optics.scanner.correlations[5]
            u_Vlos1_Vlos2_corr    = Vlos_corrcoeff#Lidar.optics.scanner.correlations[7]
            
        psi1_rho1_corr        = 0
        psi1_rho2_corr        = 0
        psi2_rho1_corr        = 0
        psi2_rho2_corr        = 0
        theta1_rho1_corr      = 0
        theta1_rho2_corr      = 0
        theta2_rho1_corr      = 0
        theta2_rho2_corr      = 0

        # pdb.set_trace()
        cov_MAT=[[theta_stds[0]**2*autocorr_theta,                  theta_stds[1]*theta_stds[0]*theta1_theta2_corr,       psi_stds[0]*theta_stds[0]*psi1_theta1_corr ,      psi_stds[1]*theta_stds[0]*psi2_theta1_corr,   rho_stds[0]*theta_stds[0]*theta1_rho1_corr,  rho_stds[1]*theta_stds[0]*theta1_rho2_corr   ,  theta_stds[0]*U_Vlos1*0,                         theta_stds[0]*U_Vlos2  *0                      ],
                 [theta_stds[0]*theta_stds[1]*theta1_theta2_corr,          theta_stds[1]**2*autocorr_theta,               psi_stds[0]*theta_stds[1]*psi1_theta2_corr,       psi_stds[1]*theta_stds[1]*psi2_theta2_corr ,  rho_stds[0]*theta_stds[1]*theta2_rho1_corr,  rho_stds[1]*theta_stds[1]*theta2_rho2_corr   ,  theta_stds[1]*U_Vlos1*0                        , theta_stds[1]*U_Vlos2   *0                    ],
                 [theta_stds[0]*psi_stds[0]*psi1_theta1_corr  ,     theta_stds[1]*psi_stds[0]*psi1_theta2_corr,               psi_stds[0]**2*autocorr_psi,                  psi_stds[1]*psi_stds[0]*psi1_psi2_corr,       rho_stds[0]*psi_stds[0]*psi1_rho1_corr,      rho_stds[1]*psi_stds[0]*psi1_rho2_corr        , psi_stds[0]*U_Vlos1*0                          , psi_stds[0]*U_Vlos2     *0                      ],
                 [theta_stds[0]*psi_stds[1]*psi2_theta1_corr,       theta_stds[1]*psi_stds[1]*psi2_theta2_corr ,          psi_stds[0]*psi_stds[1]*psi1_psi2_corr,                 psi_stds[1]**2*autocorr_psi,            rho_stds[0]*psi_stds[1]*psi2_rho1_corr,      rho_stds[1]*psi_stds[1]*psi2_rho2_corr       ,  psi_stds[1]*U_Vlos1*0                          , psi_stds[1]*U_Vlos2     *0                     ],
                 [theta_stds[0]*rho_stds[0]*theta1_rho1_corr,          theta_stds[1]*rho_stds[0]*theta2_rho1_corr,        psi_stds[0]*rho_stds[0]*psi1_rho1_corr,           psi_stds[1]*rho_stds[0]*psi2_rho1_corr,                rho_stds[0]**2*autocorr_rho,          rho_stds[1]*rho_stds[0]*rho1_rho2_corr      , rho_stds[0]*U_Vlos1*0                           ,rho_stds[0]*U_Vlos2     *0                    ],
                 [theta_stds[0]*rho_stds[1]*theta1_rho2_corr,          theta_stds[1]*rho_stds[1]*theta2_rho2_corr,         psi_stds[0]*rho_stds[1]*psi1_rho2_corr,          psi_stds[1]*rho_stds[1]*psi2_rho2_corr,       rho_stds[0]*rho_stds[1]*rho1_rho2_corr,           rho_stds[1]**2*autocorr_rho           ,    rho_stds[1]*U_Vlos1 *0  ,                        rho_stds[1]*U_Vlos2    *0                     ],
                 [theta_stds[0]*U_Vlos1*0    ,                                 theta_stds[1]*U_Vlos1*0               ,             psi_stds[0]*U_Vlos1*0    ,                          psi_stds[1]*U_Vlos1*0,                      rho_stds[0]*U_Vlos1*0   ,                rho_stds[1]*U_Vlos1*0   ,                   U_Vlos1**2*autocorr_V,                     U_Vlos1*U_Vlos2*u_Vlos1_Vlos2_corr          ],
                 [theta_stds[0]*U_Vlos2*0   ,                                  theta_stds[1]*U_Vlos2*0              ,             psi_stds[0]*U_Vlos2*0  ,                             psi_stds[1]*U_Vlos2*0,                      rho_stds[0]*U_Vlos2*0    ,               rho_stds[1]*U_Vlos2*0  ,                U_Vlos1*U_Vlos2*u_Vlos1_Vlos2_corr,                 U_Vlos2**2*autocorr_V                       ]]
        
        return  cov_MAT

#%% Calculate correlations between Vlos1, ans Vlos2
def Vlos_correlations(Lidar,Atmospheric_Scenario,wind_direction, ind_wind_dir,ind_alpha,Vlos1,Vlos2,theta1, theta2, psi1,psi2,rho1,rho2,u_theta1,u_theta2,u_psi1,u_psi2,u_rho1,u_rho2):
    
    cov_MAT=MultiVar(Lidar, 0,   0,          0       , [u_theta1,u_theta2], [u_psi1,u_psi2],  [u_rho1,u_rho2]  ,       1     ,      1     ,       1    ,    0 ,     'MC1'  )
    
    Theta1_cr,Theta2_cr,Psi1_cr,Psi2_cr,Rho1_cr,Rho2_cr,Vlos1_cr,Vlos2_cr=multivariate_normal.rvs([theta1,theta2,psi1,psi2,rho1,rho2,0,0], cov_MAT,Lidar.optics.scanner.N_MC).T
    
    H_t1_cr = (Rho1_cr*np.sin(Theta1_cr)+Lidar.optics.scanner.origin[0][2])/Lidar.optics.scanner.Href
    H_t2_cr = (Rho2_cr*np.sin(Theta2_cr)+Lidar.optics.scanner.origin[1][2])/Lidar.optics.scanner.Href   

     
    ### VLOS calculations ############################      
    Vlos1_cr=(Atmospheric_Scenario.Vref*(H_t1_cr**Atmospheric_Scenario.PL_exp[ind_alpha])*np.cos(Theta1_cr)*np.cos(Psi1_cr-wind_direction[ind_wind_dir]))
    Vlos2_cr=(Atmospheric_Scenario.Vref*(H_t2_cr**Atmospheric_Scenario.PL_exp[ind_alpha])*np.cos(Theta2_cr)*np.cos(Psi2_cr-wind_direction[ind_wind_dir]))
    Correlation_Vlos = np.corrcoef(Vlos1_cr,Vlos2_cr)
    # pdb.set_trace()

    return (Correlation_Vlos[0][1],Vlos1_cr,Vlos2_cr,Theta1_cr,Theta2_cr,Psi1_cr,Psi2_cr,Rho1_cr,Rho2_cr)
    
#%% ##########################################
##########################################
#Uncertainty Vh following MCM
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

def MCM_Vh_lidar_uncertainty (Lidar,Atmospheric_Scenario,wind_direction,ind_alpha,theta1,u_theta1,psi1 ,u_psi1,rho1,u_rho1,theta2,u_theta2,psi2,u_psi2,rho2,u_rho2):
    # pdb.set_trace()
    Vh,u,v,U_u_MC,U_v_MC,U_Vh_MCM,CorrCoefTheta2Psi2=[],[],[],[], [],[],[]
    Vlos1,Vlos2,U_Vlos1_MCM,U_Vlos2_MCM=[],[],[],[]
    Vlos1_MC_cr2_s,Vlos2_MC_cr2_s=[],[]
    CorrCoef_U_Vlos,CorrCoef_U_VLOS,CorrCoefVlos2,CorrCoefTheta2,CorrCoefPsi2,CorrCoefTheta1Psi2,CorrCoefTheta2Psi1,CorrCoef_U_uv,CorrCoefTheta1Psi1_2,CorrCoefTheta2Psi2_2=[],[],[],[],[],[],[],[],[],[]
    CorrCoefVlos1,  CorrCoefTheta1,CorrCoefPsi1,CorrCoefThetaPsi1,CorrCoefuv,CorrCoef_Theta1Psi2,CorrCoefTheta1Psi2_2,CorrCoefTheta2Psi1_2,CorrCoefTheta1Psi1=[],[],[],[],[],[],[],[]    ,[]
    Theta1_cr2_s,Theta2_cr2_s,Psi1_cr2_s,Psi2_cr2_s,Rho1_cr2_s,Rho2_cr2_s=[],[],[],[],[],[]
    

    for ind_wind_dir in range(len(wind_direction)):  
        ######## Vlos multivariate distribution #####################

        # # Multivariate distributions: 

        Vlos_corrcoeff,Vlos1_MCM,Vlos2_MCM,Theta1_cr,Theta2_cr,Psi1_cr,Psi2_cr,Rho1_cr,Rho2_cr=Vlos_correlations(Lidar,Atmospheric_Scenario,wind_direction, ind_wind_dir,ind_alpha,Vlos1,Vlos2,theta1, theta2, psi1,psi2,rho1,rho2,u_theta1,u_theta2,u_psi1,u_psi2,u_rho1,u_rho2)
        CorrCoef_U_Vlos.append(Vlos_corrcoeff)
           
        # Store data
        Vlos1.append(Vlos1_MCM)
        Vlos2.append(Vlos2_MCM)
        
        #  Uncertainty Vlosi and Uest uncertainty ##############################
        s_w=0
        # U_Vlos1_MCM0=np.sqrt(np.std(Vlos1_MCM)**2+ Lidar.optics.scanner.stdv_Estimation[0][0]**2+(np.sin(theta1)*s_w)**2)
        # U_Vlos2_MCM0=np.sqrt(np.std(Vlos2_MCM)**2+ Lidar.optics.scanner.stdv_Estimation[0][0]**2+(np.sin(theta1)*s_w)**2)       
        U_Vlos1_MCM0=np.sqrt(np.std(Vlos1_MCM)**2+ Lidar.lidar_inputs.dataframe['Intrinsic Uncertainty [m/s]']**2)
        U_Vlos2_MCM0=np.sqrt(np.std(Vlos2_MCM)**2+ Lidar.lidar_inputs.dataframe['Intrinsic Uncertainty [m/s]']**2)        

        # Store data
        U_Vlos1_MCM.append(U_Vlos1_MCM0 ) 
        U_Vlos2_MCM.append(U_Vlos2_MCM0)
     
        # pdb.set_trace()
        ###CORRELATION COEFFICIENTS 1st multivariate  
        CorrCoef_U_VLOS=(np.corrcoef(U_Vlos1_MCM[ind_wind_dir],U_Vlos2_MCM[ind_wind_dir])[0][1])              
        CorrCoefTheta1Psi1.append( np.corrcoef(Theta1_cr,Psi1_cr)[0][1]) 
        CorrCoefTheta2Psi2.append( np.corrcoef(Theta2_cr,Psi2_cr)[0][1])  
        CorrCoefTheta1.append( np.corrcoef(Theta1_cr,Theta2_cr)[0][1])
        CorrCoefVlos1.append( np.corrcoef(Vlos1[ind_wind_dir],Vlos2[ind_wind_dir])[0][1])        
        CorrCoefPsi1.append( np.corrcoef(Psi1_cr,Psi2_cr)[0][1])
        CorrCoefTheta1Psi2.append( np.corrcoef(Theta1_cr,Psi2_cr)[0][1])  
        CorrCoefTheta2Psi1.append(np.corrcoef(Theta2_cr,Psi1_cr)[0][1])
    

        ######### Vh multivariate  ####################################  
        # for ind_wind_dir in range(len(wind_direction)):             
                                                                                                                                                                                                                                                                                                                           ## MCM - uv     
        # Covariance matrix       
        #                   (Lidar,  Vlos_corrcoeff,    U_Vlos1,           U_Vlos2       ,[u_theta1,u_theta2], [u_psi1,u_psi2],  [u_rho1,u_rho2],    autocorr_theta,   autocorr_psi,   autocorr_rho,    autocorr_V ,    mode)
        cov_MAT_Vh = MultiVar(Lidar, Vlos_corrcoeff,   U_Vlos1_MCM0,    U_Vlos2_MCM0  ,   [0,0], [0,0],  [0,0],           1   ,         1     ,         1 ,            1 ,        'MC2' )
        # pdb.set_trace() 
        
        # # Multivariate distributions:       
        # Theta1_cr2,Theta2_cr2,Psi1_cr2,Psi2_cr2,Rho1_cr2,Rho2_cr2,Vlos1_MC_cr2,Vlos2_MC_cr2= multivariate_normal.rvs([theta1,theta2,psi1,psi2,rho1,rho2,np.mean(Vlos1[ind_wind_dir]),np.mean(Vlos2[ind_wind_dir])], cov_MAT_Vh,Lidar.optics.scanner.N_MC).T
        Theta1_cr2,Theta2_cr2,Psi1_cr2,Psi2_cr2,Rho1_cr2,Rho2_cr2,Vlos1_MC_cr2,Vlos2_MC_cr2= multivariate_normal.rvs([theta1,theta2,psi1,psi2,rho1,rho2,np.mean(Vlos1_MCM),np.mean(Vlos2_MCM)], cov_MAT_Vh,Lidar.optics.scanner.N_MC).T
        
        #Storing data
        Vlos1_MC_cr2_s.append(Vlos1_MC_cr2)
        Vlos2_MC_cr2_s.append(Vlos2_MC_cr2)
        Theta1_cr2_s.append(Theta1_cr2)       
        Theta2_cr2_s.append(Theta2_cr2)
        Psi1_cr2_s.append(Psi1_cr2)       
        Psi2_cr2_s.append(Psi2_cr2)
        Rho1_cr2_s.append(Rho1_cr2)       
        Rho2_cr2_s.append(Rho2_cr2)        
        # pdb.set_trace()
        
        #%% Uncertainty in horizontal velocity
        Vh_num    = np.sqrt(((Vlos1_MC_cr2*np.cos(Theta2_cr2))**2+(Vlos2_MC_cr2*np.cos(Theta1_cr2))**2)-2*(Vlos1_MC_cr2*Vlos2_MC_cr2*np.cos(Theta1_cr2)*np.cos(Theta2_cr2)*np.cos(Psi1_cr2-Psi2_cr2)))
        Vh_denom  = np.cos(Theta1_cr2)*np.cos(Theta2_cr2)*np.sin(Psi1_cr2-Psi2_cr2)       
        Vh.append(Vh_num/Vh_denom)       
        U_Vh_MCM.append(np.std(Vh[ind_wind_dir]))
        
        
        ###CORRELATION COEFFICIENTS 2nd multivariate       
        CorrCoefVlos2.append(np.corrcoef(Vlos1_MC_cr2_s[ind_wind_dir],Vlos2_MC_cr2_s[ind_wind_dir])[0][1])       
        CorrCoefTheta2.append( np.corrcoef(Theta1_cr2,Theta2_cr2)[0][1])
        CorrCoefPsi2.append( np.corrcoef(Psi1_cr2,Psi2_cr2)[0][1])
        CorrCoefTheta2_Psi2_2    = ( np.corrcoef(Theta2_cr2,Psi2_cr2)[0][1])         
        CorrCoefTheta1Psi1_2.append( np.corrcoef(Theta1_cr2,Psi1_cr2)[0][1])
        CorrCoefTheta1Psi2_2.append( np.corrcoef(Theta1_cr2,Psi2_cr2)[0][1])
        CorrCoefTheta2Psi2_2.append( np.corrcoef(Theta2_cr2,Psi2_cr2)[0][1])
        CorrCoefTheta2Psi1_2.append(np.corrcoef(Theta2_cr2,Psi1_cr2)[0][1])
        # pdb.set_trace()
                
        ########### Analysis u and v ##############################       
        u.append((-Vlos1_MC_cr2*np.cos(Theta2_cr2)*np.sin(Psi2_cr2)+Vlos2_MC_cr2*np.cos(Theta1_cr2)*np.sin(Psi1_cr2))/(np.cos(Theta1_cr2)*np.cos(Theta2_cr2)*np.sin(Psi1_cr2-Psi2_cr2)))
        v.append(( Vlos1_MC_cr2*np.cos(Theta2_cr2)*np.cos(Psi2_cr2)-Vlos2_MC_cr2*np.cos(Theta1_cr2)*np.cos(Psi1_cr2))/(np.cos(Theta1_cr2)*np.cos(Theta2_cr2)*np.sin(Psi1_cr2-Psi2_cr2)))
        ### Uncertainty u and v components
        U_u_MC.append(np.std(u[ind_wind_dir]))
        U_v_MC.append(np.std(v[ind_wind_dir]))        
        # Correlation coefficients  u v
        CorrCoef_U_uv.append(np.corrcoef(U_u_MC,U_v_MC)[0][1])
        CorrCoefuv.append(np.corrcoef(u[ind_wind_dir],v[ind_wind_dir])[0][1])
    
    # Store the multivariate distributions
    Mult_param          =  [Vlos1_MC_cr2_s,Vlos2_MC_cr2_s,Theta1_cr2_s,Theta2_cr2_s,Psi1_cr2_s,Psi2_cr2_s,Rho1_cr2_s,Rho2_cr2_s]


    # Store correlation coefficients
    Correlation_coeffs  =  [CorrCoef_U_Vlos,CorrCoefuv,CorrCoefTheta1Psi1,CorrCoefTheta2Psi2,CorrCoefTheta1,CorrCoefVlos1,CorrCoefPsi1,CorrCoefTheta1Psi2,CorrCoef_U_VLOS,
                            CorrCoefVlos2,CorrCoefTheta2,CorrCoefPsi2,CorrCoefTheta2_Psi2_2,CorrCoefTheta1Psi1_2,CorrCoefTheta1Psi2_2,CorrCoefTheta2Psi2_2,CorrCoefTheta2Psi1_2,CorrCoefTheta2Psi1]
    # pdb.set_trace()
    return U_Vlos1_MCM,U_Vlos2_MCM,u,v,U_u_MC,U_v_MC ,Mult_param,Correlation_coeffs,U_Vh_MCM


#%% ##########################################
##########################################
#Uncertainty of u and v wind components following GUM model
##########################################
##############################################
#%%

def GUM_Vlos_lidar_uncertainty(Lidar,Atmospheric_Scenario,wind_direction,ind_alpha,theta1,u_theta1,psi1,u_psi1,rho1  ,u_rho1,theta2,u_theta2,psi2  ,u_psi2,rho2  ,u_rho2):    
    # pdb.set_trace()
    # surr=np.zeros(len(wind_direction))

    
   
    U_Vlos1_GUM,U_Vlos2_GUM,U_u_GUM,U_v_GUM,VL1,VL2,u,v=[],[],[],[],[],[],[],[]
    u_V_LOS1Theta1,u_V_LOS1Psi1,u_V_LOS1Rho1,u_V_LOS2Theta2,u_V_LOS2Psi2,u_V_LOS2Rho2,Corrcoef_Vlos =[], [],[],[],[],[],[]
    H_t1 = ((rho1*np.sin(theta1)+Lidar.optics.scanner.origin[0][2])/Lidar.optics.scanner.Href)
    H_t2 = ((rho2*np.sin(theta2)+Lidar.optics.scanner.origin[1][2])/Lidar.optics.scanner.Href)
    CorrCoef_U_Vlos_GUM=[]
    ### Covariance matrix
    #           (Lidar,Vlos_corrcoeff, U_Vlos1,U_Vlos2,  [u_theta1,u_theta2],   [u_psi1,u_psi2]  ,[u_rho1,u_rho2],    autocorr_theta,  autocorr_psi ,autocorr_rho, autocorr_V ,mode)       
    psi_bba = (np.radians(360)-abs(psi1-psi2))
    # pdb.set_trace()
    for ind_wind_dir in range(len(wind_direction)):  
        
        # VLOS
        Ux=MultiVar(Lidar  ,0,     0,     0,   [u_theta1,u_theta2],   [u_psi1,u_psi2],  [u_rho1,u_rho2]        ,1   ,            1          ,1            ,     1,   'GUM1'  )  
        # VLOS uncertainty
        # Calculate and store Vlosi
        Vlos1_GUM = Atmospheric_Scenario.Vref*(H_t1**Atmospheric_Scenario.PL_exp[ind_alpha])*np.cos(theta1)*np.cos(psi1-wind_direction[ind_wind_dir])
        Vlos2_GUM = Atmospheric_Scenario.Vref*(H_t2**Atmospheric_Scenario.PL_exp[ind_alpha])*np.cos(theta2)*np.cos(psi2-wind_direction[ind_wind_dir])    
        VL1.append(Vlos1_GUM)
        VL2.append(Vlos2_GUM)

        # Calculate and store u and v components
        u_comp=((-Vlos1_GUM*np.sin(psi2)*np.cos(theta2)+Vlos2_GUM*np.sin(psi1)*np.cos(theta1))/(np.cos(theta1)*np.cos(theta2)*np.sin(psi1-psi2)))
        v_comp=(( Vlos1_GUM*np.cos(psi2)*np.cos(theta2)-Vlos2_GUM*np.cos(psi1)*np.cos(theta1))/(np.cos(theta1)*np.cos(theta2)*np.sin(psi1-psi2)))
        u.append(u_comp)
        v.append(v_comp)
        
        # Partial derivatives Vlosi with respect theta, psi and rho
        dVlos1dtheta1   =     Atmospheric_Scenario.Vref*((H_t1)**Atmospheric_Scenario.PL_exp[ind_alpha])*(Atmospheric_Scenario.PL_exp[ind_alpha]*((rho1*(np.cos(theta1))**2)/(rho1*np.sin(theta1)+Lidar.optics.scanner.origin[0][2]))-np.sin(theta1))*np.cos(psi1-wind_direction[ind_wind_dir])
        dVlos2dtheta2   =     Atmospheric_Scenario.Vref*((H_t2)**Atmospheric_Scenario.PL_exp[ind_alpha])*(Atmospheric_Scenario.PL_exp[ind_alpha]*((rho2*(np.cos(theta2))**2)/(rho2*np.sin(theta2)+Lidar.optics.scanner.origin[1][2]))-np.sin(theta2))*np.cos(psi2-wind_direction[ind_wind_dir])   
        dVlos1dpsi1     =   - Atmospheric_Scenario.Vref*((H_t1)**Atmospheric_Scenario.PL_exp[ind_alpha])*(np.cos(theta1)*np.sin(psi1-wind_direction[ind_wind_dir]))
        dVlos2dpsi2     =   - Atmospheric_Scenario.Vref*((H_t2)**Atmospheric_Scenario.PL_exp[ind_alpha])*(np.cos(theta2)*np.sin(psi2-wind_direction[ind_wind_dir]))    
        dVlos1drho1     =     Atmospheric_Scenario.Vref*((H_t1)**Atmospheric_Scenario.PL_exp[ind_alpha])*Atmospheric_Scenario.PL_exp[ind_alpha]*(np.sin(theta1)/(rho1*np.sin(theta1)+Lidar.optics.scanner.origin[0][2]))*np.cos(theta1)*np.cos(psi1-wind_direction[ind_wind_dir])
        dVlos2drho2     =     Atmospheric_Scenario.Vref*((H_t2)**Atmospheric_Scenario.PL_exp[ind_alpha])*Atmospheric_Scenario.PL_exp[ind_alpha]*(np.sin(theta2)/(rho2*np.sin(theta2)+Lidar.optics.scanner.origin[1][2]))*np.cos(theta2)*np.cos(psi2-wind_direction[ind_wind_dir])
        
        # pdb.set_trace()
        # dVlos1dtheta2 = Atmospheric_Scenario.Vref*((H_t2)**Atmospheric_Scenario.PL_exp[ind_alpha])*np.cos(theta2)*np.cos(psi1-wind_direction[ind_wind_dir])*np.sqrt(1-(np.sin(theta2)*rho2/rho1)**2)*(((Atmospheric_Scenario.PL_exp[ind_alpha]*rho2)/(rho2*np.sin(theta2)+Lidar.optics.scanner.origin[1][2]))-((np.sin(theta2)*(rho2/rho1)**2)/(1-(np.sin(theta2)*rho2/rho1)**2)))
        # dVlos2dtheta1 = Atmospheric_Scenario.Vref*((H_t1)**Atmospheric_Scenario.PL_exp[ind_alpha])*np.cos(theta1)*np.cos(psi2-wind_direction[ind_wind_dir])*np.sqrt(1-(np.sin(theta1)*rho1/rho2)**2)*(((Atmospheric_Scenario.PL_exp[ind_alpha]*rho1)/(rho1*np.sin(theta1)+Lidar.optics.scanner.origin[1][2]))-((np.sin(theta1)*(rho1/rho2)**2)/(1-(np.sin(theta1)*rho1/rho2)**2)))
        
        
        # dVlos1dpsi2   = -Atmospheric_Scenario.Vref*((H_t1)**Atmospheric_Scenario.PL_exp[ind_alpha])*np.cos(theta1)*np.sin(psi_bba+psi2-wind_direction[ind_wind_dir])
        # dVlos2dpsi1   = -Atmospheric_Scenario.Vref*((H_t2)**Atmospheric_Scenario.PL_exp[ind_alpha])*np.cos(theta2)*np.sin(psi1-psi_bba-wind_direction[ind_wind_dir])
        
        
        
        
        # pdb.set_trace()
        # Store each contribution
        u_V_LOS1Theta1.append(dVlos1dtheta1)
        u_V_LOS1Psi1.append(dVlos1dpsi1)
        u_V_LOS1Rho1.append(dVlos1drho1)
        u_V_LOS2Theta2.append(dVlos2dtheta2)
        u_V_LOS2Psi2.append(dVlos2dpsi2)
        u_V_LOS2Rho2.append(dVlos2drho2)

        
        # Influence coefficients matrix for Vlosi uncertainty estimation
        # Cx = np.array([[dVlos1dtheta1,dVlos1dtheta2,   dVlos1dpsi1,  dVlos1dpsi2,       dVlos1drho1  ,       0       ,  0  , 0],
        #                 [dVlos2dtheta1,dVlos2dtheta2,     dVlos2dpsi1 , dVlos2dpsi2,           0       ,  dVlos2drho2  ,  0  ,0]])
        Cx = np.array([[dVlos1dtheta1,        0,       dVlos1dpsi1,      0,          dVlos1drho1  ,       0       ,  0  , 0],
                        [     0,        dVlos2dtheta2,         0 ,     dVlos2dpsi2,        0       ,  dVlos2drho2  ,  0  ,0]])
        
        # Ouput covariance matrix
        Uy=Cx.dot(Ux).dot(np.transpose(Cx))
        Corrcoef_Vlos.append(Uy[0][1]/np.sqrt(Uy[1][1]*Uy[0][0]))
        # pdb.set_trace()
        # 
        # U_est ##############
        s_w= 0
        U_Vlos1_GUM.append(np.sqrt(Uy[0][0]+ Lidar.lidar_inputs.dataframe['Intrinsic Uncertainty [m/s]']**2+(np.sin(theta1)*s_w)**2))
        U_Vlos2_GUM.append(np.sqrt(Uy[1][1]+ Lidar.lidar_inputs.dataframe['Intrinsic Uncertainty [m/s]']**2+(np.sin(theta2)*s_w)**2))

        # pdb.set_trace()
       
        
       #%%##################################################################################################################################################################
       #% u and v wind components' uncertainty analysis (this needs its own function since the correlation coefficient Vlos1_Vlos2_corr = r_{Vlos1,Vlos2} needs to be calculated firsts
       ##################################################################################################################################################################
        # Partial derivative u and v respect to Vlos
        dudVlos1 = -(np.sin(psi2)/(np.cos(theta1)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2)))) 
        dudVlos2 =  (np.sin(psi1)/(np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))))
        dvdVlos1 =  np.cos(psi2)/(np.cos(theta1)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2)))
        dvdVlos2 = -(np.cos(psi1)/(np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))))   
        
     
        # Partial derivative u and v respect to theta
        
        dudtheta1 = ((Vlos2_GUM*np.sin(psi1)*(-np.sin(theta1)))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1)-np.cos(psi1)*np.sin(psi2))))+(((-Vlos2_GUM*np.cos(theta1) *np.sin(psi1)+Vlos1_GUM *np.cos(theta2) *np.sin(psi2))*(-np.sin(theta1)))/((np.cos(theta1)**2) *np.cos(theta2) *(np.cos(psi2) *np.sin(psi1) - np.cos(psi1) *np.sin(psi2))))           
        dvdtheta1 = ((-Vlos2_GUM*np.cos(psi1)*(-np.sin(theta1)))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1)-np.cos(psi1)*np.sin(psi2)))) + ((Vlos2_GUM*np.cos(theta1) *np.cos(psi1) -Vlos1_GUM *np.cos(theta2) *np.cos(psi2))*(-np.sin(theta1)))/((np.cos(theta1)**2) *np.cos(theta2) *(np.cos(psi2) *np.sin(psi1) - np.cos(psi1) *np.sin(psi2)))   
        dudtheta2 = (((-Vlos1_GUM*np.sin(psi2)*(-np.sin(theta2)))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))))) +((((-Vlos2_GUM)*np.cos(theta1)*np.sin(psi1) + Vlos1_GUM*np.cos(theta2)*np.sin(psi2))*(-np.sin(theta2)))/(np.cos(theta1)*(np.cos(theta2)**2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))))        
        dvdtheta2 = ((Vlos1_GUM*np.cos(psi2)*(-np.sin(theta2)))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2)))) +(((Vlos2_GUM*np.cos(psi1)*np.cos(theta1) - Vlos1_GUM*np.cos(psi2)*np.cos(theta2))*(-np.sin(theta2)))/(np.cos(theta1)*(np.cos(theta2)**2) * (np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))))
        
       
        # Partial derivative u and v respect to psi
        
        dudpsi1=((np.cos(psi2)*(np.cos(psi1)) - np.sin(psi2)*(-np.sin(psi1)))*(Vlos1_GUM*np.sin(psi2)*np.cos(theta2) - Vlos2_GUM*np.sin(psi1)*np.cos(theta1)))/(np.cos(theta1)*np.cos(theta2)*(np.sin(psi1)*np.cos(psi2) - np.cos(psi1)*np.sin(psi2))**2) +(Vlos2_GUM*np.cos(psi1))/(np.cos(theta2)*(np.sin(psi1)*np.cos(psi2) - np.cos(psi1)*np.sin(psi2)))       
        dvdpsi1= ((np.cos(psi2)*(np.cos(psi1))-np.sin(psi2)*(-np.sin(psi1)))*(Vlos2_GUM*np.cos(psi1)*np.cos(theta1) -  Vlos1_GUM*np.cos(psi2)*np.cos(theta2)))/(np.cos(theta1)*np.cos(theta2)*(np.sin(psi1)*np.cos(psi2) - np.cos(psi1)*np.sin(psi2))**2) - (Vlos2_GUM*(-np.sin(psi1)))/(np.cos(theta2)*(np.sin(psi1)*np.cos(psi2) -np.cos(psi1)*np.sin(psi2)))        
        dudpsi2= ((-Vlos1_GUM*(np.cos(psi2)))/(np.cos(theta1)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2)))) + (((-Vlos2_GUM)*np.cos(theta1)*np.sin(psi1) + Vlos1_GUM*np.cos(theta2)*np.sin(psi2))*(np.sin(psi1)*(-np.sin(psi2) )- np.cos(psi1)*(np.cos(psi2))))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))**2)   
        dvdpsi2= (Vlos1_GUM*(-np.sin(psi2)))/(np.cos(theta1)*(np.cos(psi2)*np.sin(psi1)-np.cos(psi1)*np.sin(psi2))) + ((Vlos2_GUM*np.cos(psi1)*np.cos(theta1)-Vlos1_GUM*np.cos(psi2)*np.cos(theta2))*(np.sin(psi1)*(-np.sin(psi2))-np.cos(psi1)*(np.cos(psi2))))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))**2)
        # pdb.set_trace()
        
        
        # Inputs' covariance matrix for u and v components' uncertainty estimation
        #Param_multivar1=[U_Vlos1_MCM           ,U_Vlos2_MCM,                                  theta_stds2          ,psi_stds2        ,rho_stds2,      psi1_psi2_corr,  theta1_theta2_corr  , rho1_rho2_corr   , psi1_theta1_corr , psi1_theta2_corr  ,       psi2_theta1_corr    , psi2_theta2_corr  , Vlos1_Vlos2_corr ,   psi1_rho1_corr ,psi1_rho2_corr ,psi2_rho1_corr ,psi2_rho2_corr ,theta1_rho1_corr ,theta1_rho2_corr ,theta2_rho1_corr ,theta2_rho2_corr,0,0,0,1 ]
        # Param_multivar2=[U_Vlos1_GUM[ind_wind_dir]     ,     U_Vlos2_GUM[ind_wind_dir]  ,[u_theta1,u_theta2],   [u_psi1,u_psi2]  ,[u_rho1,u_rho2],    psi1_psi2_corr  ,theta1_theta2_corr  ,  rho1_rho2_corr   ,psi1_theta1_corr , psi1_theta2_corr          ,psi2_theta1_corr  , psi2_theta2_corr  ,             0,        psi1_rho1_corr ,psi1_rho2_corr ,psi2_rho1_corr ,psi2_rho2_corr ,theta1_rho1_corr ,theta1_rho2_corr ,theta2_rho1_corr ,theta2_rho2_corr,1,1,1,1 ]
        
                    # (Lidar,Vlos_corrcoeff[0][1], U_Vlos1,  U_Vlos2,  [u_theta1,u_theta2],   [u_psi1,u_psi2],     [u_rho1,u_rho2],   autocorr_theta  ,autocorr_psi,autocorr_rho,autocorr_V ,mode)
        Uxuv=MultiVar(Lidar,               0,            0,     0,     [u_theta1,u_theta2],  [u_psi1,u_psi2],      [u_rho1,u_rho2],         0  ,            0            ,0        ,1        ,'GUM1')
        
        
        
        # Influence coefficients matrix for u and v components' uncertainty estimation
        Cxuv = np.array([[dudtheta1,dudtheta2,dudpsi1,dudpsi2,0,0,dudVlos1,dudVlos2],[dvdtheta1,dvdtheta2,dvdpsi1,dvdpsi2,0,0,dvdVlos1,dvdVlos2]])
        # Cxuv = dVh_dtheta1[0],dVh_dtheta2[0],dVh_dpsi1[0],dVh_dpsi2[0],0,0,dVh_Vlos1[0],dVh_Vlos2[0]
        # # u and v uncertainties estimation
        Uyuv=Cxuv.dot(Uxuv).dot(np.transpose(Cxuv))
        U_u_GUM.append(np.sqrt(Uyuv[0][0]))
        U_v_GUM.append(np.sqrt(Uyuv[1][1]))
        # pdb.set_trace()
        
    # CorrCoef_U_Vlos_GUM = np.corrcoef(u,v)[0][1]
    # CorrCoef_U_Vlos_GUM = np.corrcoef(U_Vlos1_GUM,U_Vlos2_GUM)[0][1]

    # Storing individual uncertainty contributors
    Awachesneip=[u_V_LOS1Theta1,u_V_LOS1Psi1,u_V_LOS1Rho1]
    Awachesneip2=[u_V_LOS2Theta2,u_V_LOS2Psi2,u_V_LOS2Rho2]
    CorrCoef_U_VLOS=(np.corrcoef(U_Vlos1_GUM,U_Vlos2_GUM)[0][1])
    # pdb.set_trace()
    return(VL1,VL2,U_Vlos1_GUM,U_Vlos2_GUM,Corrcoef_Vlos,Awachesneip,Awachesneip2)

#%% ##########################################
##########################################
#Uncertainty of Vh following GUM
##########################################
##############################################
#%%
def GUM_Vh_lidar_uncertainty (Lidar,Atmospheric_Scenario,Corrcoef_Vlos,wind_direction,theta1,psi1,rho1,theta2,psi2 ,rho2,u_theta1,u_theta2,u_psi1,u_psi2,u_rho1,u_rho2 ,Vlos1_GUM,Vlos2_GUM,U_Vlos1_GUM,U_Vlos2_GUM):
        # Vh Uncertainty
        Correlation_Vlos_GUM,UUy,U_Vh_GUM,dV1,dV2,dVt1,dVt2,dVp1,dVp2,dV1V2=[],[],[],[],[],[],[],[],[],[]
        # App_dVh_Vlos1,App_dVh_Vlos2,App_dVh_Vlos12=[],[],[]
        for ind_wind_dir in range(len(wind_direction)):  
            
            num1 = np.sqrt(((Vlos1_GUM[ind_wind_dir]*np.cos(theta2))**2)+((Vlos2_GUM[ind_wind_dir]*np.cos(theta1))**2)-(2*Vlos1_GUM[ind_wind_dir]*Vlos2_GUM[ind_wind_dir]*np.cos(psi1-psi2)*np.cos(theta1)*np.cos(theta2)))
            den=np.cos(theta1)*np.cos(theta2)*np.sin(psi1-psi2)
            den=np.cos(theta1)*np.cos(theta2)*np.sin(psi1-psi2)
            dVh_Vlos1= (1/den)*(1/(num1))*(Vlos1_GUM[ind_wind_dir]*((np.cos(theta2))**2)-Vlos2_GUM[ind_wind_dir]*np.cos(theta1)*np.cos(theta2)*np.cos(psi1-psi2))
            dVh_Vlos2= (1/den)*(1/(num1))*(Vlos2_GUM[ind_wind_dir]*((np.cos(theta1))**2)-Vlos1_GUM[ind_wind_dir]*np.cos(theta1)*np.cos(theta2)*np.cos(psi1-psi2))
            
            # App_dVh_Vlos1.append(dVh_Vlos1)
            # App_dVh_Vlos2.append(dVh_Vlos2)
            # App_dVh_Vlos12.append(dVh_Vlos1*dVh_Vlos2)
            # pdb.set_trace()
            dV1.append((dVh_Vlos1*U_Vlos1_GUM[ind_wind_dir])**2)
            dV2.append((dVh_Vlos2*U_Vlos2_GUM[ind_wind_dir])**2)
            dV1V2.append(2*dVh_Vlos1*U_Vlos1_GUM[ind_wind_dir]*dVh_Vlos2*U_Vlos2_GUM[ind_wind_dir]*Corrcoef_Vlos[ind_wind_dir])
            
            # dVh_dtheta1
            dnum1= (1/(2*num1))*(-2*(Vlos2_GUM[ind_wind_dir]**2)*np.cos(theta1)*np.sin(theta1)+2*Vlos2_GUM[ind_wind_dir]*Vlos1_GUM[ind_wind_dir]*np.sin(theta1)*np.cos(theta2)*np.cos(psi1-psi2))
            dVh_dtheta1 = (dnum1*den+num1*np.sin(theta1)*np.cos(theta2)*np.sin(psi1-psi2))/den**2
            dVt1.append(dVh_dtheta1)
           
            # dVh_dtheta2
            dnum2= (1/(2*num1))*(-2*(Vlos1_GUM[ind_wind_dir]**2)*np.cos(theta2)*np.sin(theta2)+2*Vlos2_GUM[ind_wind_dir]*Vlos1_GUM[ind_wind_dir]*np.sin(theta2)*np.cos(theta1)*np.cos(psi1-psi2))
            dVh_dtheta2 = (dnum2*den+num1*np.cos(theta1)*np.sin(theta2)*np.sin(psi1-psi2))/den**2
            dVt2.append(dVh_dtheta2)
            
            # dVh_dpsi1
            dnum3= (1/(2*num1))*(2*Vlos2_GUM[ind_wind_dir]*Vlos1_GUM[ind_wind_dir]*np.cos(theta2)*np.cos(theta1)*np.sin(psi1-psi2))
            dVh_dpsi1 = (dnum3*den-num1*np.cos(theta1)*np.cos(theta2)*np.cos(psi1-psi2))/(np.cos(theta2)*np.cos(theta1)*np.sin(psi1-psi2))**2
            dVp1.append(dVh_dpsi1)
            
            # dVh_dpsi2
            dnum4= (1/(2*num1))*(-2*Vlos2_GUM[ind_wind_dir]*Vlos1_GUM[ind_wind_dir]*np.cos(theta2)*np.cos(theta1)*np.sin(psi1-psi2))
            dVh_dpsi2 = (dnum4*den+num1*np.cos(theta1)*np.cos(theta2)*np.cos(psi1-psi2))/(np.cos(theta2)*np.cos(theta1)*np.sin(psi1-psi2))**2
            dVp2.append(dVh_dpsi2)

            # Covariance and sensitivity matrices:
            # pdb.set_trace()                
                      # (Lidar,   Vlos_corrcoeff ,                         U_Vlos1,                   U_Vlos2,          theta_stds,    psi_stds,  rho_stds, autocorr_theta,autocorr_psi,autocorr_rho,autocorr_V ,   mode)
            UxVh=MultiVar(Lidar,  Corrcoef_Vlos[ind_wind_dir],   U_Vlos1_GUM[ind_wind_dir],U_Vlos2_GUM[ind_wind_dir],   [u_theta1,u_theta2],  [u_psi1,u_psi2],      [u_rho1,u_rho2],        1,            1,           1    ,    1,        'GUM2' )
            # CxVh=[dVh_dtheta1,dVh_dtheta2,dVh_dpsi1,dVh_dpsi2,0,0,dVh_Vlos1,dVh_Vlos2]
            CxVh=[0,0,0,0,0,0,dVh_Vlos1,dVh_Vlos2]
            UyVh=np.array(CxVh).dot(UxVh).dot(np.transpose(CxVh))
            UUy.append(UyVh)
            U_Vh_GUM.append(np.sqrt(UyVh))
            Correlation_Vlos_GUM.append(UxVh[-1][-2]/(U_Vlos1_GUM[ind_wind_dir]*U_Vlos2_GUM[ind_wind_dir]))       
        
        
        # Vlos1_mil=[((App_dVh_Vlos1[i])*(U_Vlos1_GUM[i]))**2 for i in range(len(U_Vlos1_GUM))]
        # Vlos2_mil=[((App_dVh_Vlos2[i])*(U_Vlos2_GUM[i]))**2 for i in range(len(U_Vlos2_GUM))]
        # pdb.set_trace()
        return(U_Vh_GUM,dV1,dV2,dV1V2,Correlation_Vlos_GUM)
    
    
    
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
        # pdb.set_trace()
        # W_D = np.degrees(np.arctan(v[ind_wind_dir]/u[ind_wind_dir]))
        W_D = (np.arctan((Vlos1[ind_wind_dir]*np.cos(Theta2[ind_wind_dir])*np.cos(Psi2[ind_wind_dir])-Vlos2[ind_wind_dir]*np.cos(Theta1[ind_wind_dir])*np.cos(Psi1[ind_wind_dir]))/(-Vlos1[ind_wind_dir]*np.cos(Theta2[ind_wind_dir])*np.sin(Psi2[ind_wind_dir])+Vlos2[ind_wind_dir]*np.cos(Theta1[ind_wind_dir])*np.sin(Psi1[ind_wind_dir]))))
        U_Wind_direction.append(np.degrees(np.std(W_D)))
    # pdb.set_trace()
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
        # pdb.set_trace()
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
    
        # #Pedersen
        # u_v1_v1           = U_Vlos1[ind_wind_dir]**2
        # u_v2_v2           = U_Vlos2[ind_wind_dir]**2
        # u_v1_v2           = U_Vlos1[ind_wind_dir]*U_Vlos2[ind_wind_dir] #*Vlos1_Vlos2_corr
        # pdb.set_trace()
        # # Partial derivative u and v respect to Vlos
        # dudV1 = -(np.sin(psi2)/(np.cos(theta1)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2)))) 
        # dudV2 =  (np.sin(psi1)/(np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))))
        # dvdV1 =  np.cos(psi2)/(np.cos(theta1)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2)))
        # dvdV2 = -(np.cos(psi1)/(np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))))   
        
     
        # # Partial derivative u and v respect to theta
        
        # dudtheta1 = ((Vlos2[ind_wind_dir]*np.sin(psi1)*(-np.sin(theta1)))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1)-np.cos(psi1)*np.sin(psi2))))+(((-Vlos2[ind_wind_dir]*np.cos(theta1) *np.sin(psi1)+Vlos1[ind_wind_dir]*np.cos(theta2) *np.sin(psi2))*(-np.sin(theta1)))/((np.cos(theta1)**2) *np.cos(theta2) *(np.cos(psi2) *np.sin(psi1) - np.cos(psi1) *np.sin(psi2))))           
        # dvdtheta1 = ((-Vlos2[ind_wind_dir]*np.cos(psi1)*(-np.sin(theta1)))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1)-np.cos(psi1)*np.sin(psi2)))) + ((Vlos2[ind_wind_dir]*np.cos(theta1) *np.cos(psi1) -Vlos1[ind_wind_dir]*np.cos(theta2) *np.cos(psi2))*(-np.sin(theta1)))/((np.cos(theta1)**2) *np.cos(theta2) *(np.cos(psi2) *np.sin(psi1) - np.cos(psi1) *np.sin(psi2)))   
        # dudtheta2 = (((-Vlos1[ind_wind_dir]*np.sin(psi2)*(-np.sin(theta2)))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))))) +((((-Vlos2[ind_wind_dir])*np.cos(theta1)*np.sin(psi1) + Vlos1[ind_wind_dir]*np.cos(theta2)*np.sin(psi2))*(-np.sin(theta2)))/(np.cos(theta1)*(np.cos(theta2)**2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))))        
        # dvdtheta2 = ((Vlos1[ind_wind_dir]*np.cos(psi2)*(-np.sin(theta2)))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2)))) +(((Vlos2[ind_wind_dir]*np.cos(psi1)*np.cos(theta1) - Vlos1[ind_wind_dir]*np.cos(psi2)*np.cos(theta2))*(-np.sin(theta2)))/(np.cos(theta1)*(np.cos(theta2)**2) * (np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))))
        
       
        # # Partial derivative u and v respect to psi
        
        # dudpsi1=((np.cos(psi2)*(np.cos(psi1)) - np.sin(psi2)*(-np.sin(psi1)))*(Vlos1[ind_wind_dir]*np.sin(psi2)*np.cos(theta2) - Vlos2[ind_wind_dir]*np.sin(psi1)*np.cos(theta1)))/(np.cos(theta1)*np.cos(theta2)*(np.sin(psi1)*np.cos(psi2) - np.cos(psi1)*np.sin(psi2))**2) +(Vlos2[ind_wind_dir]*np.cos(psi1))/(np.cos(theta2)*(np.sin(psi1)*np.cos(psi2) - np.cos(psi1)*np.sin(psi2)))       
        # dvdpsi1= ((np.cos(psi2)*(np.cos(psi1))-np.sin(psi2)*(-np.sin(psi1)))*(Vlos2[ind_wind_dir]*np.cos(psi1)*np.cos(theta1) -  Vlos1[ind_wind_dir]*np.cos(psi2)*np.cos(theta2)))/(np.cos(theta1)*np.cos(theta2)*(np.sin(psi1)*np.cos(psi2) - np.cos(psi1)*np.sin(psi2))**2) - (Vlos2[ind_wind_dir]*(-np.sin(psi1)))/(np.cos(theta2)*(np.sin(psi1)*np.cos(psi2) -np.cos(psi1)*np.sin(psi2)))        
        # dudpsi2= ((-Vlos1[ind_wind_dir]*(np.cos(psi2)))/(np.cos(theta1)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2)))) + (((-Vlos2[ind_wind_dir])*np.cos(theta1)*np.sin(psi1) + Vlos1[ind_wind_dir]*np.cos(theta2)*np.sin(psi2))*(np.sin(psi1)*(-np.sin(psi2) )- np.cos(psi1)*(np.cos(psi2))))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))**2)   
        # dvdpsi2= (Vlos1[ind_wind_dir]*(-np.sin(psi2)))/(np.cos(theta1)*(np.cos(psi2)*np.sin(psi1)-np.cos(psi1)*np.sin(psi2))) + ((Vlos2[ind_wind_dir]*np.cos(psi1)*np.cos(theta1)-Vlos1[ind_wind_dir]*np.cos(psi2)*np.cos(theta2))*(np.sin(psi1)*(-np.sin(psi2))-np.cos(psi1)*(np.cos(psi2))))/(np.cos(theta1)*np.cos(theta2)*(np.cos(psi2)*np.sin(psi1) - np.cos(psi1)*np.sin(psi2))**2)
        
        # # u and v uncertainties estimation
        # Uyuv=S_in.dot(Uxuv).dot(np.transpose(S_in))
        # U_u_GUM.append(np.sqrt(Uyuv[0][0]))
        # U_v_GUM.append(np.sqrt(Uyuv[1][1]))
        # U_uv_GUM.append((Uyuv[1] [0]))
        
        # r_uv.append(U_uv_GUM[ind_wind_dir]/np.sqrt(((U_u_GUM[ind_wind_dir])**2)*((U_v_GUM[ind_wind_dir])**2)))

        # # Wind direction
        # dWind_dirdu = 1/(1+(v[ind_wind_dir]/u[ind_wind_dir])**2)*(-v[ind_wind_dir]/u[ind_wind_dir]**2)
        # dWind_dirdv = 1/(1+(v[ind_wind_dir]/u[ind_wind_dir])**2)*(1/u[ind_wind_dir])
        # pdb.set_trace()
        
        # U_wind_dir.append(np.degrees(np.sqrt((dWind_dirdu*U_u_GUM[ind_wind_dir])**2+(dWind_dirdv*U_v_GUM[ind_wind_dir])**2+2*dWind_dirdv*dWind_dirdu*U_uv_GUM[ind_wind_dir])))
        

    # pdb.set_trace()
    return (U_wind_dir)

    

    
########### Intrinsic lidar uncertainty

def U_intrinsic(Lidar,List_Unc_lidar,Qlunc_yaml_inputs):
    # pdb.set_trace()
    V_ref            = Qlunc_yaml_inputs['Atmospheric_inputs']['Vref']      # Reference voltaje ADC
    lidar_wavelength = Qlunc_yaml_inputs['Components']['Laser']['Wavelength'] # wavelength of the laser source.
    fd               = 2*V_ref/lidar_wavelength  # Doppler frequency corresponding to Vref
    corr_wavelength_fd=0
    u_intrinsic = np.sqrt((fd*List_Unc_lidar['Stdv wavelength [m]']/2)**2+(Qlunc_yaml_inputs['Components']['Laser']['Wavelength']*List_Unc_lidar['Stdv Doppler f_peak [Hz]']/2)**2+(fd*Qlunc_yaml_inputs['Components']['Laser']['Wavelength']*List_Unc_lidar['Stdv Doppler f_peak [Hz]']*List_Unc_lidar['Stdv wavelength [m]'])*corr_wavelength_fd/2) 
    return u_intrinsic



# # 3D
# u = -(-Vlos3* np.cos(theta1) *np.sin(psi1)* np.sin(psi2) + 
#         Vlos3 *np.cos(theta2) *np.sin(psi1) *np.sin(psi2) + 
#         Vlos2 *np.cos(theta1) *np.sin(psi1)* np.sin(psi3) - 
#         Vlos2 *np.cos(theta3)* np.sin(psi1)* np.sin(psi3) - 
#         Vlos1 *np.cos(theta2) *np.sin(psi2) *np.sin(psi3) + 
#         Vlos1 *np.cos(theta3) *np.sin(psi2) *np.sin(psi3))/(np.cos(psi3) *np.cos(
#           theta1) *np.cos(theta3) *np.sin(psi1)* np.sin(psi2) - 
#         np.cos(psi3) *np.cos(theta2)* np.cos(theta3) *np.sin(psi1) *np.sin(psi2) - 
#         np.cos(psi2) *np.cos(theta1)* np.cos(theta2) *np.sin(psi1) *np.sin(psi3) + 
#         np.cos(psi2) *np.cos(theta2)* np.cos(theta3) *np.sin(psi1) *np.sin(psi3) + 
#         np.cos(psi1) *np.cos(theta1)* np.cos(theta2) *np.sin(psi2) *np.sin(psi3) - 
#         np.cos(psi1) *np.cos(theta1)* np.cos(theta3) *np.sin(psi2)* np.sin(psi3))
# v =-(Vlos3 *np.cos(psi2) *np.cos(theta2) *np.sin(psi1) - 
#         Vlos2 *np.cos(psi3) *np.cos(theta3) *np.sin(psi1) - 
#         Vlos3 *np.cos(psi1) *np.cos(theta1) *np.sin(psi2) + 
#         Vlos1 *np.cos(psi3) *np.cos(theta3) *np.sin(psi2) + 
#         Vlos2 *np.cos(psi1) *np.cos(theta1) *np.sin(psi3) - 
#         Vlos1 *np.cos(psi2) *np.cos(theta2) *np.sin(psi3))/(-np.cos(psi3) *np.cos(
#           theta1)* np.cos(theta3) *np.sin(psi1) *np.sin(psi2) + 
#         np.cos(psi3) *np.cos(theta2) *np.cos(theta3) *np.sin(psi1) *np.sin(psi2) + 
#         np.cos(psi2) *np.cos(theta1) *np.cos(theta2) *np.sin(psi1) *np.sin(psi3) - 
#         np.cos(psi2) *np.cos(theta2) *np.cos(theta3) *np.sin(psi1) *np.sin(psi3) - 
#         np.cos(psi1) *np.cos(theta1) *np.cos(theta2) *np.sin(psi2) *np.sin(psi3) + 
#         np.cos(psi1) *np.cos(theta1) *np.cos(theta3) *np.sin(psi2) *np.sin(psi3))
# w=-(-Vlos3 *np.cos(psi2) *np.cos(theta1)* np.cos(theta2) *np.sin(psi1) + 
#         Vlos2* np.cos(psi3) *np.cos(theta1)* np.cos(theta3) *np.sin(psi1) + 
#         Vlos3 *np.cos(psi1) *np.cos(theta1)* np.cos(theta2) *np.sin(psi2) - 
#         Vlos1 *np.cos(psi3) *np.cos(theta2)* np.cos(theta3) *np.sin(psi2) - 
#         Vlos2 *np.cos(psi1) *np.cos(theta1)* np.cos(theta3) *np.sin(psi3) + 
#         Vlos1 *np.cos(psi2) *np.cos(theta2)* np.cos(theta3) *np.sin(
#           psi3))/(-np.cos(psi3)* np.cos(theta1)* np.cos(theta3) *np.sin(psi1) *np.sin(
#           psi2) + np.cos(psi3) *np.cos(theta2) *np.cos(theta3)* np.sin(psi1) *np.sin(
#           psi2) + np.cos(psi2) *np.cos(theta1) *np.cos(theta2) *np.sin(psi1) *np.sin(
#           psi3) - np.cos(psi2) *np.cos(theta2) *np.cos(theta3) *np.sin(psi1) *np.sin(
#           psi3) - np.cos(psi1) *np.cos(theta1) *np.cos(theta2) *np.sin(psi2) *np.sin(
#           psi3) + np.cos(psi1) *np.cos(theta1)* np.cos(theta3) *np.sin(psi2) *np.sin(
#           psi3))
