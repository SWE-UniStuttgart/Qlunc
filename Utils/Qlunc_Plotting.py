# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:18:05 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c) 

"""
#%% import packages:
from Utils.Qlunc_ImportModules import *

#%% Plotting:
def plotting(Lidar,Qlunc_yaml_inputs,Data,flag_plot_measuring_points_pattern,flag_plot_photodetector_noise):
    """
    Plotting. Location: .Utils/Qlunc_plotting.py
    
    Parameters
    ----------
    
    * Lidar
        data...
        
    Returns
    -------
    
    list
    
    """
    # Ploting general parameters:
    plot_param={'axes_label_fontsize' : 25,
                'textbox_fontsize'    : 14,
                'title_fontsize'      : 29,
                'suptitle_fontsize'   : 23,
                'legend_fontsize'     : 15,
                'xlim'                : [-45,45],
                'ylim'                : [-45,45],
                'zlim'                : [-40,45],
                'markersize'          : 5,
                'markersize_lidar'    : 9,
                'marker'              : '.r',
                'markerTheo'          : '.b',
                'tick_labelrotation'  : 45,
                'Qlunc_version'       : 'Qlunc Version - 0.91'
                }
        
##################    Ploting scanner measuring points pattern #######################
    if flag_plot_measuring_points_pattern:              
        # Plotting
        fig,axs4 = plt.subplots()  
        axs4=plt.axes(projection='3d')
        axs4.plot([Lidar.optics.scanner.origin[0]],[Lidar.optics.scanner.origin[1]],[Lidar.optics.scanner.origin[2]],'ob',label='{} coordinates [{},{},{}]'.format(Lidar.LidarID,Lidar.optics.scanner.origin[0],Lidar.optics.scanner.origin[1],Lidar.optics.scanner.origin[2]),markersize=plot_param['markersize_lidar'])
        axs4.plot(Data['MeasPoint_Coordinates'][0],Data['MeasPoint_Coordinates'][1],Data['MeasPoint_Coordinates'][2],plot_param['markerTheo'],markersize=plot_param['markersize'],label='Theoretical measuring point')
        axs4.plot(Data['NoisyMeasPoint_Coordinates'][0],Data['NoisyMeasPoint_Coordinates'][1],Data['NoisyMeasPoint_Coordinates'][2],plot_param['marker'],markersize=plot_param['markersize'],label='Distance error [m] = {0:.3g}$\pm${1:.3g}'.format(np.mean(Data['Simu_Mean_Distance']),np.mean(Data['STDV_Distance'])))
        
        # Setting labels, legend, title and axes limits:
        axs4.set_xlabel('x [m]',fontsize=plot_param['axes_label_fontsize'])#,orientation=plot_param['tick_labelrotation'])
        axs4.set_ylabel('y [m]',fontsize=plot_param['axes_label_fontsize'])#,orientation=plot_param['tick_labelrotation'])
        axs4.set_zlabel('z [m]',fontsize=plot_param['axes_label_fontsize'])        
        axs4.set_title('Lidar pointing accuracy ['+Qlunc_yaml_inputs['Components']['Scanner']['Type']+']',fontsize=plot_param['title_fontsize'])
        axs4.legend()
        axs4.set_xlim3d(plot_param['xlim'][0],plot_param['xlim'][1])
        axs4.set_ylim3d(plot_param['ylim'][0],plot_param['ylim'][1])
        axs4.set_zlim3d(plot_param['zlim'][0],plot_param['zlim'][1])
       
###############   Plot photodetector noise   #############################       
    if flag_plot_photodetector_noise:
        # Quantifying uncertainty from photodetector and interval domain for the plot Psax is define in the photodetector class properties)
        Psax=10*np.log10(Lidar.photonics.photodetector.Power_interval) 
    
        # Plotting:
        fig,ax=plt.subplots()
        label0=['Shot SNR','Thermal SNR','Dark current SNR','Total SNR1','TIA SNR']
        i_label=0
        for i in Data['SNR_data_photodetector']:            
            ax.plot(Psax,Data['SNR_data_photodetector'][i][0],label=label0[i_label])  
            i_label+=1
        # ax.plot(Psax,Data['Total_SNR_data'],label='Total SNR')
        ax.set_xlabel('Input Signal optical power [dBm]',fontsize=plot_param['axes_label_fontsize'])
        ax.set_ylabel('SNR [dB]',fontsize=plot_param['axes_label_fontsize'])
        ax.legend(fontsize=plot_param['legend_fontsize'])
        ax.set_title('SNR Photodetector',fontsize=plot_param['title_fontsize'])
        ax.grid(axis='both')
        ax.text(.90,.05,plot_param['Qlunc_version'],transform=ax.transAxes, fontsize=14,verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    
