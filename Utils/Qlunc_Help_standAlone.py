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
    # pdb.set_trace()
    del data_db
    return np.array(res_dB)

#%% Spherical into cartesian  coordinate transformation
    #    xcart = rho * cos(phi)*sin(theta) 
    #    ycart = rho * sin(phi)*sin(theta)
    #    zcart = rho * cos(theta)
    
def sph2cart(Lidar): 
    x=[]
    y=[]
    z=[]    
    for i in range(len(Lidar.optics.scanner.focus_dist)):
        x=Lidar.optics.scanner.focus_dist[i]*np.cos(np.deg2rad(Lidar.optics.scanner.phi))*np.sin(np.deg2rad(Lidar.optics.scanner.theta))
        y=Lidar.optics.scanner.focus_dist[i]*np.sin(np.deg2rad(Lidar.optics.scanner.phi))*np.sin(np.deg2rad(Lidar.optics.scanner.theta)) 
        z=Lidar.optics.scanner.focus_dist[i]*np.cos(np.deg2rad(Lidar.optics.scanner.theta)) 
    return(x,y,z)

#%% NDF function

def to_netcdf(DataXarray,Qlunc_yaml_inputs,Lidar,Atmospheric_Scenario):
    #DataXarray=Lidar.lidar_inputs.dataframe
    """.
    
    Save the project to an netndf file - Location: Qlunc_Help_standAlone.py
    
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