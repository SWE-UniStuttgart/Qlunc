# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 00:24:33 2020

@author: fcosta
"""
#import itertools as itt
#import functools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Qlunc_Help_standAlone as SA
import pdb
# inputs
origin         = [0],[0],[0] #Origin
focus_distance = ([70]) # Focus distance [m] for pulsed we have several distances
sample_rate    = 10 #number of samples per frequency pattern??

# Define pattern in spherical coordinates:
phi    = np.array([np.arange(0,360,sample_rate)])         # Azimuth)
theta  = np.array([10])                                  # Height
rho    = focus_distance/np.cos(np.deg2rad(theta)) # beam length

# Errors:
#calculate error taking into account reference in the focused point
#
stdv_rho    = 0 # define the errors in measuring for each focus distance (stdv)
stdv_theta  = 0 # This is for the focus point
stdv_phi    = 0 # This is for the focus point

# Estimating the error in theta and phi, for each measured height, by 'equivalent triangles' method
vec_stdv_thetas = focus_distance*stdv_theta/np.max(focus_distance)
vec_stdv_phis    = focus_distance*stdv_phi/np.max(focus_distance)

# Function calculating...
X,Y,Z=SA.pointing_error(rho,theta,phi,stdv_rho,vec_stdv_thetas,vec_stdv_phis)
#%% Plotting:
ax=plt.axes(projection='3d')
#for t in range (len(y)):
ax.plot3D(x,y,z,'or')
ax.plot3D(*origin,'ob')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_zlabel('z[m]')
ax.set_title('Qlunc Scanner UQ')
ax.set_xlim3d(-15,15)
ax.set_ylim3d(-15,15)
ax.set_zlim3d(0,100)
#ax.set_legend('stdv_rho  {}'.format(stdv_rho))
#ax.quiver(*origin,xcart,ycart,zcart)
