# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:18:05 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c) 

"""

from Qlunc_Instantiate import *
#%% Plotting:

# Ploting general parameters:
plot_param={'axes_label_fontsize' : 25,
            'textbox_fontsize'    : 14,
            'title_fontsize'      : 29,
            'suptitle_fontsize'   : 23,
            'legend_fontsize'     : 15,
            'xlim'                : [-10,60],
            'ylim'                : [-40,40],
            'zlim'                : [0,115],
            'markersize'          : 5,
            'markersize_lidar'    : 9,
            'marker'              : '.r',
            'markerTheo'          : '.b',
            'tick_labelrotation'  : 45,
            'Qlunc_version'       : 'Qlunc Version - 0.9'
            }


##############    Ploting scanner measuring points pattern #######################
if flags.flag_plot_measuring_points_pattern:  
    Scanner_Data1 = Lidar.optics.scanner.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs) # Calling Scanner uncertainty to plot the graphics

    
    # Creating the figure and the axes
    fig,axs4 = plt.subplots()  
    axs4=plt.axes(projection='3d')
    
    # Plotting
    axs4.plot([Lidar.optics.scanner.origin[0]],[Lidar.optics.scanner.origin[1]],[Lidar.optics.scanner.origin[2]],'ob',label='{} coordinates [{},{},{}]'.format(Lidar.LidarID,Lidar.optics.scanner.origin[0],Lidar.optics.scanner.origin[1],Lidar.optics.scanner.origin[2]),markersize=plot_param['markersize_lidar'])
    axs4.plot(Scanner_Data1['MeasPoint_Coordinates'][0],Scanner_Data1['MeasPoint_Coordinates'][1],Scanner_Data1['MeasPoint_Coordinates'][2],plot_param['markerTheo'],markersize=plot_param['markersize'],label='Theoretical measuring point')
    axs4.plot(Scanner_Data1['NoisyMeasPoint_Coordinates'][0],Scanner_Data1['NoisyMeasPoint_Coordinates'][1],Scanner_Data1['NoisyMeasPoint_Coordinates'][2],plot_param['marker'],markersize=plot_param['markersize'],label='Distance error [m] = {0:.3g}$\pm${1:.3g}'.format(np.mean(Scanner_Data1['Simu_Mean_Distance']),np.mean(Scanner_Data1['STDV_Distance'])))
    
    # Setting labels, legend, title and axes limits:
    axs4.set_xlabel('x [m]',fontsize=plot_param['axes_label_fontsize'])#,orientation=plot_param['tick_labelrotation'])
    axs4.set_ylabel('y [m]',fontsize=plot_param['axes_label_fontsize'])#,orientation=plot_param['tick_labelrotation'])
    axs4.set_zlabel('z [m]',fontsize=plot_param['axes_label_fontsize'])

    
    axs4.set_title('Lidar pointing accuracy ['+Qlunc_yaml_inputs['Components']['Scanner']['Type']+']',fontsize=plot_param['title_fontsize'])
    axs4.legend()
    axs4.set_xlim3d(plot_param['xlim'][0],plot_param['xlim'][1])
    axs4.set_ylim3d(plot_param['ylim'][0],plot_param['ylim'][1])
    axs4.set_zlim3d(plot_param['zlim'][0],plot_param['zlim'][1])
   
###########   Plot photodetector noise   #############################       
if flags.flag_plot_photodetector_noise:
    # Quantifying uncertainty from photodetector and interval domain for the plot Psax is define in the photodetector class properties)
    UQ_photo = Lidar.photonics.photodetector.Uncertainty(Lidar,Atmospheric_Scenario,cts) # Obtain the UQ photodetector dictionary with SNR and UQ information
    Psax=10*np.log10(Lidar.photonics.photodetector.Power_interval) 

    # Plotting:
    
    fig,ax=plt.subplots()
    ax.plot(Psax,UQ_photo['SNR_data_photodetector']['SNR_Shot_Noise'][0],Psax,UQ_photo['SNR_data_photodetector']['SNR_Thermal'][0],Psax,UQ_photo['SNR_data_photodetector']['SNR_Dark_Current'][0],Psax,UQ_photo['SNR_data_photodetector']['SNR_TIA'][0])
    ax.set_xlabel('Input Signal optical power (dBm)',fontsize=plot_param['axes_label_fontsize'])
    ax.set_ylabel('SNR (dB)',fontsize=plot_param['axes_label_fontsize'])
    ax.legend(['Shot Noise','Thermal Noise','Dark current Noise','TIA Noise'],fontsize=plot_param['legend_fontsize'])#,'Total error [w]'])
    ax.set_title('SNR Photodetector',fontsize=plot_param['title_fontsize'])
    ax.grid(axis='both')
    ax.text(.90,.05,plot_param['Qlunc_version'],transform=ax.transAxes, fontsize=14,verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


