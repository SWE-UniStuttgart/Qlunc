# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:09:12 2022

@author: fcosta
"""
###############   Plot Animation uncertainties   #############################           
def PlotAnimUnc(Wdir_U,Vh_u,Coord):
    from celluloid import Camera
    from matplotlib import animation
    import matplotlib.pyplot as plt
    import time
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    import numpy as np
    import pdb

    fig, axs0 = plt.subplots(subplot_kw={'projection': '3d'})
    axs0.set_xlabel('x[m]',fontsize=20)
    axs0.set_ylabel('y[m]',fontsize=20)
    axs0.set_zlabel('z[m]',fontsize=20,rotation=90)
    # axs0.set_subplots_adjust(left=1.5, bottom=None, right=None, top=None, wspace=None, hspace=None)
    camera1 = Camera(fig)
    summ1,ii1= [],0
    summ2,ii2= [],0
    
    axs2 = fig.add_axes([.75, .4, 0.15, 0.35])
    axs2.get_xaxis().set_visible(False)
    axs2.set_ylim(0,1)
    
    axs3=axs2.twinx()    
    axs3.set_ylim(0,1)
    axs3.get_xaxis().set_visible(False)

    # Plot the lidars locations, the measuremets locations and the trajectories   
    for n in range(len(Wdir_U)):    
        axs0.plot(Coord[0][0][0],Coord[0][0][1],Coord[0][0][2],'sb')
        axs0.plot(Coord[0][4][0],Coord[0][4][1],Coord[0][4][2],'sb')        
        for ini in np.arange(n,-1,-1):
            axs0.plot(Coord[ini][1],Coord[ini][2],Coord[ini][3],'or')    
        axs0.plot([Coord[0][0][0],Coord[n][1]],[Coord[0][0][1],Coord[n][2]],[Coord[0][0][2],Coord[n][3]],'--g')
        axs0.plot([Coord[0][4][0],Coord[n][1]],[Coord[0][4][1],Coord[n][2]],[Coord[0][4][2],Coord[n][3]],'--g')

    # Wind direction uncertainty animation 
        # pdb.set_trace()
        for ini in np.arange(n,-1,-1):
            plt.axhline(Wdir_U[ini],color = 'g', linestyle = '-',alpha=0.1)
        summ1.append([np.sum(Wdir_U[indd]) for indd in range(ii1+1)])
        ii1+=1
        props = dict(boxstyle='round', facecolor='green', alpha=0.4)
        textstr1 = 'mean uncertainty wind direction[°] = '+str(np.round(np.nanmean(summ1[n]),2))+'; stdv[°] = '+str(np.round(np.nanstd(summ1[n]),2))
        plt.axhline(np.nanmean(summ1[n]),color = 'darkgreen', linestyle = '-',linewidth=3)
        plt.errorbar(np.linspace(0,0.1,1),np.nanmean(summ1[n]),yerr=np.round(np.nanstd(summ1[n]),2),ecolor = 'darkgreen')    
        axs2.text(-0.15, -0.25, textstr1, transform=axs2.transAxes, fontsize=9,horizontalalignment='left',verticalalignment='top', bbox=props)
        axs2.set_ylabel('Wind direction Uncertainty [°]', color = 'tab:green',fontsize=11.5)
        # pdb.set_trace()
    # Horizontal velocityh uncertainty animation
    
        for ini in np.arange(n,-1,-1):
            # pdb.set_trace()
            plt.axhline(Vh_u[ini],color = 'b', linestyle = '-',alpha=0.1)
        summ2.append([np.sum(Vh_u[indd]) for indd in range(ii2+1)])
        ii2+=1
        props = dict(boxstyle='round', facecolor='blue', alpha=0.4)
        # pdb.set_trace()
        textstr2 ='mean uncertainty wind speed [m/s] = '+str(np.round(np.nanmean(summ2[n]),2))+'; stdv[m/s] = '+str(np.round(np.nanstd(summ2[n]),2))
        plt.axhline(np.nanmean(summ2[n]),color = 'mediumblue', linestyle = '-',linewidth=3)
        plt.errorbar(np.linspace(0,0.1,1),np.nanmean(summ2[n]),yerr=np.round(np.nanstd(summ2[n]),2),ecolor = 'mediumblue')    
        axs3.text(-0.15, -.10, textstr2, transform=axs3.transAxes, fontsize=9,horizontalalignment='left',verticalalignment='top', bbox=props)
        axs3.set_ylabel('Wind velocity Uncertainty [m/s]', color = 'tab:blue',fontsize=11.5)
        camera1.snap()    
    plt.show()
    
    # Play animation
    animation1 = camera1.animate(interval = 500, repeat = True,repeat_delay = 500)

    # Save animation  
    pdb.set_trace()
    animation1.save('C:/SWE_LOCAL/GIT_Qlunc/Lidar_Projects/Unc4P.gif')
    
    return(animation1)
            
            