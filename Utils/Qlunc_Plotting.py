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
def plotting(Lidar,Qlunc_yaml_inputs,Data,flag_plot_measuring_points_pattern,flag_plot_photodetector_noise,flag_probe_volume_param):
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
                'xlim'                : [-60,60],
                'ylim'                : [-60,60],
                'zlim'                : [0,80],
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
        fig,axs0 = plt.subplots()  
        axs0=plt.axes(projection='3d')
        axs0.plot([Lidar.optics.scanner.origin[0]],[Lidar.optics.scanner.origin[1]],[Lidar.optics.scanner.origin[2]],'ob',label='{} coordinates [{},{},{}]'.format(Lidar.LidarID,Lidar.optics.scanner.origin[0],Lidar.optics.scanner.origin[1],Lidar.optics.scanner.origin[2]),markersize=plot_param['markersize_lidar'])
        axs0.plot(Data['MeasPoint_Coordinates'][0],Data['MeasPoint_Coordinates'][1],Data['MeasPoint_Coordinates'][2],plot_param['markerTheo'],markersize=plot_param['markersize'],label='Theoretical measuring point')
        axs0.plot(Data['NoisyMeasPoint_Coordinates'][0],Data['NoisyMeasPoint_Coordinates'][1],Data['NoisyMeasPoint_Coordinates'][2],plot_param['marker'],markersize=plot_param['markersize'],label='Distance error [m] = {0:.3g}$\pm${1:.3g}'.format(np.mean(Data['Simu_Mean_Distance']),np.mean(Data['STDV_Distance'])))
        
        # Setting labels, legend, title and axes limits:
        axs0.set_xlabel('x [m]',fontsize=plot_param['axes_label_fontsize'])#,orientation=plot_param['tick_labelrotation'])
        axs0.set_ylabel('y [m]',fontsize=plot_param['axes_label_fontsize'])#,orientation=plot_param['tick_labelrotation'])
        axs0.set_zlabel('z [m]',fontsize=plot_param['axes_label_fontsize'])        
        axs0.set_title('Lidar pointing accuracy ['+Qlunc_yaml_inputs['Components']['Scanner']['Type']+']',fontsize=plot_param['title_fontsize'])
        axs0.legend()
        axs0.set_xlim3d(plot_param['xlim'][0],plot_param['xlim'][1])
        axs0.set_ylim3d(plot_param['ylim'][0],plot_param['ylim'][1])
        axs0.set_zlim3d(plot_param['zlim'][0],plot_param['zlim'][1])
       
###############   Plot photodetector noise   #############################       
    if flag_plot_photodetector_noise:
        # Quantifying uncertainty from photodetector and interval domain for the plot Psax is define in the photodetector class properties)
        Psax=10*np.log10(Lidar.photonics.photodetector.Power_interval) 
    
        # Plotting:
        fig,axs1=plt.subplots()
        label0=['Shot SNR','Thermal SNR','Dark current SNR','Total SNR','TIA SNR']
        i_label=0
        for i in Data['SNR_data_photodetector']:            
            axs1.plot(Psax,Data['SNR_data_photodetector'][i][0],label=label0[i_label])  
            i_label+=1
        # axs1.plot(Psax,Data['Total_SNR_data'],label='Total SNR')
        axs1.set_xlabel('Input Signal optical power [dBm]',fontsize=plot_param['axes_label_fontsize'])
        axs1.set_ylabel('SNR [dB]',fontsize=plot_param['axes_label_fontsize'])
        axs1.legend(fontsize=plot_param['legend_fontsize'])
        axs1.set_title('SNR Photodetector',fontsize=plot_param['title_fontsize'])
        axs1.grid(axis='both')
        axs1.text(.90,.05,plot_param['Qlunc_version'],transform=axs1.transAxes, fontsize=14,verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


###############   Plot Probe Volume parameters    ############################
    if flag_probe_volume_param: 
        typeLidar ="CW"
        wave      = Qlunc_yaml_inputs['Components']['Laser']['Wavelength']  # wavelength
        f_length  = Qlunc_yaml_inputs['Components']['Telescope']['Focal length'] # focal length
        a         = np.arange(2e-3,4e-3,.02e-3) # distance fiber-end--telescope lens
        a0        = Qlunc_yaml_inputs['Components']['Telescope']['Fiber-lens offset'] # the offset (a constant number), to avoid the fiber-end locates at the focal point, otherwise the lights will be parallel to each other
        A         = 20e-3 # beam radius at the output lens
        ext_coef  = 0.085
        effective_radius_telescope  = 16.6e-3

        # The focus distance varies with the distance between the fiber-end and the telescope lens. So that, also the probe length varies with such distance.
        #Calculating focus distance depending on the distance between the fiber-end and the telescope lens:
        focus_distance = 1/((1/f_length)-(1/(a+a0))) # Focus distance
        dist =(np.linspace(0,60,len(a)))  # distance from the focus position along the beam direction
        
        # Rayleigh length variation due to focus_distance variations (due to the distance between fiber-end and telescope lens)
        zr = (wave*(focus_distance**2))/(np.pi*(Qlunc_yaml_inputs['Components']['Telescope']['Effective radius telescope'])**2)# Rayleigh length  (considered as the probe length) # half-width of the weighting function --> FWHM = 2*zr
    
        # Probe volume:
        #Probe_volume = np.pi*(A**2)*((4*(focus_distance**2)*wave)/(Telescope_aperture)) # based on Marijn notes
        vol_zr       = np.pi*(A**2)*(2*zr) # based on the definition of Rayleigh length in Liqin Jin notes (Focus calibration formula)
        
        # Lorentzian weighting function:
        phi = (ext_coef/np.pi)*(1/((1**2)+(36.55-focus_distance)**2))
        
        # Plotting
        fig=plt.figure()
        ax2=fig.add_subplot(2,2,1)
        ax2.plot(dist,phi)
        ax2.set_yscale('log')
        
            
        # fig2=plt.figure()
        ax3=fig.add_subplot(2,2,2)
        ax3.plot(focus_distance,zr)
        
        ax4=fig.add_subplot(2,2,3)
        ax4.plot(a,zr)
        
        ax5=fig.add_subplot(2,2,4)
        ax5.plot(focus_distance,a)
    
    
        # Titles and axes
        ax2.title.set_text('weighting function')
        ax3.title.set_text('Rayleigh Vs focus distance')
        ax4.title.set_text('Rayleigh Vs Fiber-end')
        ax5.title.set_text('Fiber-end Vs focus distance')
    
    
    





###############   Plot optical amplifier noise   #############################    
    # if flag_plot_optical_amplifier_noise:
    #     # Quantifying uncertainty from photodetector and interval domain for the plot Psax is define in the photodetector class properties)
    #     Psax=10*np.log10(Lidar.photonics.photodetector.Power_interval) 
    
    #     # Plotting:
    #     fig,axs1=plt.subplots()
    #     label0=['Optical amplifier OSNR']
    #     i_label=0
    #     for i in Data['SNR_data_photodetector']:            
    #         axs1.plot(Psax,Data['OSNR'][i][0],label=label0[i_label])  
    #         i_label+=1
    #     axs1.set_xlabel('Input Signal optical power [dBm]',fontsize=plot_param['axes_label_fontsize'])
    #     axs1.set_ylabel('SNR [dB]',fontsize=plot_param['axes_label_fontsize'])
    #     axs1.legend(fontsize=plot_param['legend_fontsize'])
    #     axs1.set_title('OSNR Optical Amplifier',fontsize=plot_param['title_fontsize'])
    #     axs1.grid(axis='both')
    #     axs1.text(.90,.05,plot_param['Qlunc_version'],transform=axs1.transAxes, fontsize=14,verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))