# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:22:47 2022

@author: fcosta
"""
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

#%% Define inputs
theta =np.linspace(0,86,1000)
stdv_angle = np.radians(.0573)
Vh    = 12.5
alpha =np.round(np.linspace(.1,.5,5),3)

#%% Uncertainty (GUM)

# Uniform flow
U_Vrad,U_Vh=[],[]
# U_Vrad.append([Vh*np.sin(np.radians(theta[ind_u]))*stdv_angle for ind_u in range(len(theta))])
U_Vrad.append([100*np.cos(np.radians(theta[ind_u]))*np.tan(np.radians(theta[ind_u]))*stdv_angle for ind_u in range(len(theta))])
# pdb.set_trace()
U_Vh.append([100*np.tan(np.radians(theta[ind_u]))*stdv_angle for ind_u in range(len(theta))])

# Including shear:
U_Vrad_sh,U_Vh_sh = [],[]
for ind_alpha in alpha:
    # 
    U_Vrad_sh.append([100*np.cos(np.radians(theta[ind_u]))*stdv_angle*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])
    U_Vh_sh.append([100*stdv_angle*abs((ind_alpha/math.tan(np.radians(theta[ind_u])))-np.tan(np.radians(theta[ind_u])) ) for ind_u in range(len(theta))])

    # U_Vh_sh.append([(Vh*np.tan(np.radians(theta[ind_u]))+ind_alpha*math.atan(np.radians(theta[ind_u])))*stdv_angle for ind_u in range(len(theta))])
    # U_Vh_sh.append([(ind_alpha*math.atan(np.radians(theta[ind_u])))*stdv_angle for ind_u in range(len(theta))])
    # pdb.set_trace()
    # U_Vh_sh.append([ind_alpha*(()**alpha) for ind_u in range(len(theta))])
    


#%% Plot errors
color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))
fig,axs1 = plt.subplots()  
axs1.plot(theta,U_Vrad[0],'b-',label='Uniform flow')
for ind_a in range(len(alpha)):
    c=next(color)
    axs1.plot(theta,U_Vrad_sh[ind_a],'r-.',label='Shear ({})'.format(alpha[ind_a]),c=c)

axs1.set_xlabel('theta [°]',fontsize=25)
axs1.set_ylabel('U [%]',fontsize=25)
plt.title('Uncertainty in radial velocity',fontsize=29)
axs1.legend()
axs1.grid(axis='both')

color=iter(cm.rainbow(np.linspace(0,1,len(alpha))))

fig,axs2 = plt.subplots()  
axs2.plot(theta,U_Vh[0],'b-',label='Uniform flow')
for ind_a in range(len(alpha)):
    c=next(color)
    axs2.plot(theta,U_Vh_sh[ind_a],'r-.',label='Shear ({})'.format(alpha[ind_a]),c=c)

axs2.set_xlabel('theta [°]',fontsize=25)
axs2.set_ylabel('U [%]',fontsize=25)

plt.title('Uncertainty in horizontal velocity',fontsize=29)
axs2.legend()
axs2.grid(axis='both')