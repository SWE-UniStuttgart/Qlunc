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
    # pdb.set_trace()
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
        # pdb.set_trace()
        for i in range(len(data_watts[0])): # combining all uncertainties making sum of squares and the sqrt of the sum
            zipped_data.append(list(zip(*data_watts))[i])
            res_watts.append(np.sqrt(sum(map (lambda x: x**2,zipped_data[i])))) #  Combined stdv
            # res_watts.append(sum(map (lambda x: x**2,zipped_data[i]))) #   Combined Variance
            
            res_dB=10*np.log10(res_watts) #Convert into dB 
        # pdb.set_trace()
        del data_db
    # pdb.set_trace()
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
    # pdb.set_trace()
    return(np.around(x,5),np.around(y,5),np.around(z,5))

def cart2sph(x,y,z): 
    rho=[]
    theta=[]
    phi=[]
     
    for ind in range(len(z)):
        rho.append(np.sqrt(x[ind]**2+y[ind]**2+z[ind]**2))
        theta.append(math.acos(np.sqrt(x[ind]**2+y[ind]**2)/np.sqrt(x[ind]**2+y[ind]**2+z[ind]**2)))
        phi.append(math.atan2(y[ind],x[ind]))
        # if z[ind]>0:
        #         phi.append(np.arctan(np.sqrt(x[ind]**2+y[ind]**2)/z[ind]))
        # elif z[ind]==0:
        #         phi.append(np.array(np.pi/2))
        # elif z[ind]<0:
        #         phi.append((np.pi)+(np.arctan(np.sqrt(x[ind]**2+y[ind]**2)/z[ind])))

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
    # pdb.set_trace()
    
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
        # pdb.set_trace()
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
    # pdb.set_trace()
    rm=[]
    rms=[]
    # ind_rm=0
    sum_rm=[]
    # for ffi,fi in zip(ff,f):
    # pdb.set_trace()
    rm=([(np.array(ff)-np.array(f))**2])
    rms=(np.sqrt(np.sum(rm)/len(ff)))
    # ind_rm=ind_rm+1
    # pdb.set_trace()
    return np.array(rms)

#%% Define meshgrid for the errors in pointing accuracy and focus range
def mesh (theta,psi,rho):
    # pdb.set_trace()
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


def U_Vh_MC(theta_c, psi_c,rho_c,wind_direction,ind_wind_dir,Href,Vref,alpha,Hl):
    # u
    A0 = Vref*(((Hl+(rho_c[1]*np.sin(theta_c[1])))/Href)**alpha)
    B0 = np.cos(theta_c[1])*np.cos(psi_c[1]-wind_direction[ind_wind_dir])#np.cos(theta_c[1])*np.cos(psi_c[1])*np.cos(wind_direction[ind_wind_dir])+np.cos(theta_c[1])*np.sin(psi_c[1])*np.sin(wind_direction[ind_wind_dir])#+np.sin(theta_c[0])*np.tan(wind_tilt)
    C0 = np.cos(theta_c[0])*np.sin(psi_c[0])
    D0 = Vref*(((Hl+(rho_c[0]*np.sin(theta_c[0])))/Href)**alpha)
    E0 = np.cos(theta_c[0])*np.cos(psi_c[0]-wind_direction[ind_wind_dir])#np.cos(theta_c[0])*np.cos(psi_c[0])*np.cos(wind_direction[ind_wind_dir])+np.cos(theta_c[0])*np.sin(psi_c[0])*np.sin(wind_direction[ind_wind_dir])#+np.sin(theta_c[1])*np.tan(wind_tilt)
    F0 = np.cos(theta_c[1])*np.sin(psi_c[1])
    G  = np.cos(theta_c[0])*np.cos(theta_c[1])*(np.sin(psi_c[0]-psi_c[1])  )  
    H0 = (A0*B0*C0)-(D0*E0*F0)
    
    # # v
    I0 = Vref*(((Hl+(rho_c[0]*np.sin(theta_c[0])))/Href)**alpha)
    J0 = np.cos(theta_c[0])*np.cos(psi_c[0]-wind_direction[ind_wind_dir])#np.cos(theta_c[0])*np.cos(psi_c[0])*np.cos(wind_direction[ind_wind_dir])+np.cos(theta_c[0])*np.sin(psi_c[0])*np.sin(wind_direction[ind_wind_dir])#+np.sin(Theta2_cr)*np.tan(wind_tilt)
    K0 = np.cos(theta_c[1])*np.cos(psi_c[1])
    L0 = Vref*(((Hl+(rho_c[1]*np.sin(theta_c[1])))/Href)**alpha)
    M0 = np.cos(theta_c[1])*np.cos(psi_c[1]-wind_direction[ind_wind_dir])#np.cos(theta_c[1])*np.cos(psi_c[1])*np.cos(wind_direction[ind_wind_dir])+np.cos(theta_c[1])*np.sin(psi_c[1])*np.sin(wind_direction[ind_wind_dir])#+np.sin(Theta1_cr)*np.tan(wind_tilt)
    N0 = np.cos(theta_c[0])*np.cos(psi_c[0])   
    O0 = (I0*J0*K0)-(L0*M0*N0)
    
  
    return ([A0,I0],[B0,J0],[C0,K0],[D0,L0],[E0,M0],[F0,N0],G,[H0,O0])



def U_Vh_GUM(theta_c, psi_c,rho_c,wind_direction,ind_wind_dir,Href,Vref,alpha,Hl):
      #Vh
    
    A = Vref*(((Hl+(rho_c[0]*np.sin(theta_c[0])))/Href)**alpha)
    B = np.cos(theta_c[0])*np.cos(psi_c[0]-wind_direction[ind_wind_dir])#np.cos(theta_c[0])*np.cos(psi_c[0])*np.cos(wind_direction[ind_wind_dir])+np.cos(theta_c[0])*np.sin(psi_c[0])*(np.sin(wind_direction[ind_wind_dir]))
    C = Vref*(((Hl+(rho_c[1]*np.sin(theta_c[1])))/Href)**alpha)
    D = np.cos(theta_c[1])*np.cos(psi_c[1]-wind_direction[ind_wind_dir])#np.cos(theta_c[1])*np.cos(psi_c[1])*np.cos(wind_direction[ind_wind_dir])+np.cos(theta_c[1])*np.sin(psi_c[1])*(np.sin(wind_direction[ind_wind_dir]))
    E = np.cos(theta_c[0])*np.cos(theta_c[1])*np.cos(psi_c[0]-psi_c[1])
    F = np.cos(theta_c[0])*np.cos(theta_c[1])*(np.sin(psi_c[0]-psi_c[1]))
    
    numerator = np.sqrt((A*B*np.cos(theta_c[1]))**2 + (C*D*np.cos(theta_c[0]))**2 - 2*(A*C*B*D*E))    
    dernumerator = 1/(2*numerator) 
    sh_term1 = Vref*alpha[0]*(((rho_c[0]*np.sin(theta_c[0])+Hl)/(Href))**(alpha[0]-1))
    sh_term2 = Vref*alpha[0]*(((rho_c[1]*np.sin(theta_c[1])+Hl)/(Href))**(alpha[0]-1))
    
    # Derivatives
    #Theta
    dert11 =  2*A*B*(np.cos(theta_c[1])**2)*((sh_term1*rho_c[0]*np.cos(theta_c[0])*B/Href)-np.sin(theta_c[0])*A*(np.cos(psi_c[0]-wind_direction[ind_wind_dir])))#2*A*B*(np.cos(theta_c[1])**2)*((sh_term1*rho_c[0]*np.cos(theta_c[0])*B/Href)+(A*(-np.sin(theta_c[0])*np.cos(psi_c[0])*np.cos(wind_direction[ind_wind_dir])-np.sin(theta_c[0])*np.sin(psi_c[0])*np.sin(wind_direction[ind_wind_dir]))))
    dert12 = -2*(C**2)*(D**2)*np.cos(theta_c[0])*(np.sin(theta_c[0]))
    dert13 = -2*C*D*((sh_term1*rho_c[0]*np.cos(theta_c[0])*B*E/Href)-np.sin(theta_c[0])*A*E*np.cos(psi_c[0]-wind_direction[ind_wind_dir])-A*B*(np.sin(theta_c[0])*np.cos(theta_c[1])*np.cos(psi_c[0]-psi_c[1])))#-2*((sh_term1*rho_c[0]*np.cos(theta_c[0])*C*B*D*E/Href)+A*C*(-np.sin(theta_c[0])*np.cos(psi_c[0])*np.cos(wind_direction[ind_wind_dir])-np.sin(theta_c[0])*np.sin(psi_c[0])*np.sin(wind_direction[ind_wind_dir]))*D*E+A*C*B*D*(-np.sin(theta_c[0])*np.cos(theta_c[1])*np.cos(psi_c[0]-psi_c[1])))
    
    dert21 = -2*(A**2)*(B**2)*np.cos(theta_c[1])*(np.sin(theta_c[1]))
    dert22 =  2*C*D*(np.cos(theta_c[0])**2)*(sh_term2*rho_c[1]*np.cos(theta_c[1])*D/Href-C*np.sin(theta_c[1])*np.cos(psi_c[1]-wind_direction[ind_wind_dir]))#2*C*D*(np.cos(theta_c[0])**2)*((sh_term2*rho_c[1]*np.cos(theta_c[1])*D/Href)+(C*(-np.sin(theta_c[1])*np.cos(psi_c[1])*np.cos(wind_direction[ind_wind_dir])-np.sin(theta_c[1])*np.sin(psi_c[1])*np.sin(wind_direction[ind_wind_dir]))))
    dert23 = -2*A*B*((sh_term2*rho_c[1]*np.cos(theta_c[1])*D*E/Href)-E*C*np.sin(theta_c[1])*np.cos(psi_c[1]-wind_direction[ind_wind_dir])-C*D*(np.sin(theta_c[1])*np.cos(theta_c[0])*np.cos(psi_c[0]-psi_c[1])))#-2*((sh_term2*rho_c[1]*np.cos(theta_c[1])*A*B*D*E/Href)+A*C*(-np.sin(theta_c[1])*np.cos(psi_c[1])*np.cos(wind_direction[ind_wind_dir])-np.sin(theta_c[1])*np.sin(psi_c[1])*np.sin(wind_direction[ind_wind_dir]))*B*E+A*C*B*D*(-np.sin(theta_c[1])*np.cos(theta_c[0])*np.cos(psi_c[0]-psi_c[1])))
    # pdb.set_trace()
    
    #Psi
    derp11 =  2*(A**2)*B*(np.cos(theta_c[1])**2)*np.cos(theta_c[0])*(np.sin(wind_direction[ind_wind_dir]-psi_c[0]))#(2*A*B*np.cos(theta_c[1])*A*(-np.cos(theta_c[0])*np.sin(psi_c[0])*np.cos(wind_direction[ind_wind_dir])+np.cos(theta_c[0])*np.cos(psi_c[0])*np.sin(wind_direction[ind_wind_dir])))*np.cos(theta_c[1])
    derp12 = -2*A*C*D*np.cos(theta_c[0])*(E*(np.sin(wind_direction[ind_wind_dir]-psi_c[0]))-B*np.cos(theta_c[1])*(np.sin(psi_c[0]-psi_c[1])))#-2*(A*C*D*E*(-np.cos(theta_c[0])*np.sin(psi_c[0])*np.cos(wind_direction[ind_wind_dir])+np.cos(theta_c[0])*np.cos(psi_c[0])*np.sin(wind_direction[ind_wind_dir]))+A*B*C*D*np.cos(theta_c[0])*np.cos(theta_c[1])*(-np.sin(psi_c[0]-psi_c[1])))
    
    derp21 =  2*(C**2)*D*(np.cos(theta_c[0])**2)*np.cos(theta_c[1])*(np.sin(-psi_c[1]+wind_direction[ind_wind_dir]))#(2*C*D*np.cos(theta_c[0])*C*(-np.cos(theta_c[1])*np.sin(psi_c[1])*np.cos(wind_direction[ind_wind_dir])+np.cos(theta_c[1])*np.cos(psi_c[1])*np.sin(wind_direction[ind_wind_dir])))*np.cos(theta_c[0])
    derp22 = -2*A*C*B*np.cos(theta_c[1])*(E*(np.sin(wind_direction[ind_wind_dir]-theta_c[1]))+D*np.cos(theta_c[0])*(np.sin(psi_c[0]-psi_c[1])))#-2*(A*C*B*E*(-np.cos(theta_c[1])*np.sin(theta_c[1])*np.cos(wind_direction[ind_wind_dir])+np.cos(theta_c[1])*np.cos(psi_c[1])*np.sin(wind_direction[ind_wind_dir]))+A*B*C*D*np.cos(theta_c[0])*np.cos(theta_c[1])*(np.sin(psi_c[0]-psi_c[1])))
    
    #rho
    derr11 = (2*A*(B**2)*(np.cos(theta_c[1]))**2)*sh_term1*np.sin(theta_c[0])/Href
    derr12 = -2*C*B*D*E*sh_term1*np.sin(theta_c[0])/Href
              
    derr21 = (2*C*(D**2)*(np.cos(theta_c[0]))**2)*sh_term2*np.sin(theta_c[1])/Href
    derr22 = -2*A*B*D*E*sh_term2*np.sin(theta_c[1])/Href
              
    return (F,dert11,dert12,dert13,dert21, dert22, dert23,derp11,derp12,derp21,derp22,derr11,derr12,derr21,derr22,numerator,dernumerator)
    # return (A,B,C,D,E,F,Vh)


def U_VLOS_MC(theta_corr,psi_corr,rho_corr,theta1_noisy,Hl,Href,alpha,wind_direction,Vref,ind_wind_dir,VLOS1_list,VLOS2_list):
     VLOS01,U_VLOS1,VLOS02,U_VLOS2=[],[],[],[]
            
     A1    = Vref*((Hl+(np.sin(theta_corr[0])*rho_corr[0]))/Href)**alpha[0]
     A2    = Vref*((Hl+(np.sin(theta_corr[1])*rho_corr[1]))/Href)**alpha[0]
     VLOS1 = A1*(np.cos(theta_corr[0])*np.cos(psi_corr[0]-wind_direction[ind_wind_dir])) #-np.sin(theta_corr[0][ind_npoints])*np.tan(wind_tilt[ind_npoints])
     VLOS2 = A2*(np.cos(theta_corr[1])*np.cos(psi_corr[1]-wind_direction[ind_wind_dir])) #-np.sin(theta_corr[1][ind_npoints])*np.tan(wind_tilt[ind_npoints])
     
     VLOS1_list.append(np.mean(VLOS1))
     VLOS2_list.append(np.mean(VLOS2))
    
     U_VLOS1   = np.nanstd(VLOS1)
     U_VLOS2   = np.nanstd(VLOS2)
     CORR_COEF = np.corrcoef(VLOS1,VLOS2)
    
     return(VLOS1,VLOS2,U_VLOS1,U_VLOS2,CORR_COEF[0][1],VLOS1_list,VLOS2_list)

CCC1=[]
CCC2=[]
CCC3=[]
def U_VLOS_GUM (theta1,theta2,psi1,psi2,rho1,rho2,U_theta1,U_theta2,U_psi1,U_psi2,U_rho1,U_rho2,U_VLOS1,U_VLOS2,Hl,Vref,Href,alpha,wind_direction,ind_wind_dir,CROS_CORR,CORR_COEF):
    U_Vrad_sh_theta1,U_Vrad_sh_psi1,U_Vrad_sh_range1,U_Vrad1_GUM=[],[],[],[]
    U_Vrad_sh_theta2,U_Vrad_sh_psi2,U_Vrad_sh_range2,U_Vrad2_GUM=[],[],[],[]
    
    
    # VLOS1
    # pdb.set_trace()
    U_Vrad_sh_theta1 = Vref*(((Hl+(np.sin(theta1)*rho1))/Href)**alpha[0])*np.cos(psi1-wind_direction[ind_wind_dir])*(alpha[0]*((rho1*(np.cos(theta1)**2)/(Hl+(np.sin(theta1)*rho1))))-np.sin(theta1))*U_theta1    
    U_Vrad_sh_psi1   = Vref*np.cos((theta1))*((((Hl)+(np.sin(theta1)*rho1))/Href)**alpha[0])*(np.sin(-psi1[0]+wind_direction[ind_wind_dir]))*U_psi1
    U_Vrad_sh_range1 = Vref*alpha[0]*(((Hl+(np.sin(theta1)*rho1))/Href)**alpha[0])*np.cos(theta1)*np.cos(psi1-wind_direction[ind_wind_dir])*(np.sin(theta1)/(Hl+(np.sin(theta1)*rho1)))*U_rho1
    # VLOS2
    U_Vrad_sh_theta2 = Vref*(((Hl+(np.sin(theta2)*rho2))/Href)**alpha[0])*np.cos(psi2-wind_direction[ind_wind_dir])*(alpha[0]*((rho2*(np.cos(theta2)**2)/(Hl+(np.sin(theta2)*rho2))))-np.sin(theta2))*U_theta2    
    U_Vrad_sh_psi2   = Vref*np.cos((theta2))*((((Hl)+(np.sin(theta2)*rho2))/Href)**alpha[0])*(np.sin(-psi2[0]+wind_direction[ind_wind_dir]))*(U_psi2) 
    U_Vrad_sh_range2 = Vref*alpha[0]*(((Hl+(np.sin(theta2)*rho2))/Href)**alpha[0])*np.cos(theta2)*np.cos(psi2-wind_direction[ind_wind_dir])*(np.sin(theta2)/(Hl+(np.sin(theta2)*rho2)))*U_rho2
    
    # pdb.set_trace()
    # 2.4 Expanded uncertainty with contributions of theta, psi and rho

    # Correlation terms corresponding to the relation between same angles of different lidars ([theta1,theta2],[psi1,psi2],[rho1,rho2])
    CC_P1_P2 = U_Vrad_sh_psi2*U_Vrad_sh_psi1*CROS_CORR[6]
    CC_T1_T2 = U_Vrad_sh_theta2*U_Vrad_sh_theta1*CROS_CORR[7]
    CC_R1_R2 = U_Vrad_sh_range2*U_Vrad_sh_range1*CROS_CORR[8]
    
    # Correlations terms LOS1 (theta1, rho1 and psi1)
    CC_T1_P1 = U_Vrad_sh_theta1*U_Vrad_sh_psi1*CROS_CORR[0]
    CC_T1_R1 = U_Vrad_sh_theta1*U_Vrad_sh_range1*CROS_CORR[1]
    CC_P1_R1 = U_Vrad_sh_range1*U_Vrad_sh_psi1*CROS_CORR[2]
    CC_VLOS  = 0#U_VLOS1*U_VLOS2*CORR_COEF
    
    CCC1.append(CC_T1_P1)
    CCC2.append(CC_P1_R1)
    CCC3.append(CC_T1_R1)
    # pdb.set_trace()

    U_Vrad1_GUM=np.sqrt(((U_Vrad_sh_theta1)**2+(U_Vrad_sh_psi1)**2+(U_Vrad_sh_range1)**2)+2*(CC_T1_P1+CC_T1_R1+CC_P1_R1+CC_VLOS)) 
    
    # Correlations variables LOS2 (theta2, rho2 and psi2)
    CC_T2_P2 = U_Vrad_sh_theta2*U_Vrad_sh_psi2*CROS_CORR[3]
    CC_T2_R2 =U_Vrad_sh_theta2*U_Vrad_sh_range2*CROS_CORR[4]
    CC_P2_R2 = U_Vrad_sh_range2*U_Vrad_sh_psi2*CROS_CORR[5]
    
    U_Vrad2_GUM=np.sqrt(((U_Vrad_sh_theta2)**2+(U_Vrad_sh_psi2)**2+(U_Vrad_sh_range2)**2)+2*(CC_T2_P2+CC_T2_R2+CC_P2_R2+CC_VLOS))
    return(U_Vrad1_GUM,U_Vrad2_GUM)

