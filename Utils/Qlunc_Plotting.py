# -*- coding: utf-8 -*-
""".

Created on Tue Oct 20 21:18:05 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c) 

"""
#%% import packages:
from Utils.Qlunc_ImportModules import *


def scatter3d(x,y,z, Vrad_homo, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(Vrad_homo), vmax=max(Vrad_homo)) #Normalize(vmin=0.005, vmax=.045) # 
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, Vrad_homo, s=75, c=scalarMap.to_rgba(Vrad_homo))
    ax.set_xlabel('theta [°]',fontsize=plot_param['axes_label_fontsize'])
    ax.set_ylabel('psi [°]',fontsize=plot_param['axes_label_fontsize'])
    ax.set_zlabel('rho [m]',fontsize=plot_param['axes_label_fontsize'])
    scalarMap.set_array(Vrad_homo)
    fig.colorbar(scalarMap,label='V_Rad Uncertainty [m/s]',shrink=0.7)
    # fig.colorbar.tick_params(labelsize=10)
    plt.show()
    

#%% Plotting:
def plotting(Lidar,Qlunc_yaml_inputs,Data,flag_plot_measuring_points_pattern,flag_plot_photodetector_noise,flag_probe_volume_param,flag_plot_optical_amplifier_noise, flag_plot_pointing_unc):
    """.
    
    Plotting. Location: .Utils/Qlunc_plotting.py       
    Parameters
    ----------    
    
    * Lidar
        data...            
    Returns
    ------- 
    
    list
    
    
    """
    # pdb.set_trace()
    # Ploting general parameters:
    plot_param={'axes_label_fontsize' : 25,
                'textbox_fontsize'    : 14,
                'title_fontsize'      : 29,
                'suptitle_fontsize'   : 23,
                'legend_fontsize'     : 15,
                'xlim'                : [-280,280],
                'ylim'                : [-280,280],
                'zlim'                : [-280,280],
                'markersize'          : 5,
                'markersize_lidar'    : 9,
                'marker'              : '.r',
                'markerTheo'          : '.b',
                'tick_labelrotation'  : 45,
                'tick_labelfontsize'  : 19,
                'Qlunc_version'       : 'Qlunc Version - 0.91'
                }
        
##################    Ploting scanner measuring points pattern #######################
    if flag_plot_measuring_points_pattern:              
        # Plotting
# =============================================================================
#         # fig,axs0 = plt.subplots()  
#         # axs0=plt.axes(projection='3d')
#         # axs0.plot([Lidar.optics.scanner.origin[0]],[Lidar.optics.scanner.origin[1]],[Lidar.optics.scanner.origin[2]],'ob',label='{} coordinates [{},{},{}]'.format(Lidar.LidarID,Lidar.optics.scanner.origin[0],Lidar.optics.scanner.origin[1],Lidar.optics.scanner.origin[2]),markersize=plot_param['markersize_lidar'])
#         # axs0.plot(Data['MeasPoint_Coordinates'][0],Data['MeasPoint_Coordinates'][1],Data['MeasPoint_Coordinates'][2],plot_param['markerTheo'],markersize=plot_param['markersize'],label='Theoretical measuring point')
#         # axs0.plot(Data['NoisyMeasPoint_Coordinates'][0],Data['NoisyMeasPoint_Coordinates'][1],Data['NoisyMeasPoint_Coordinates'][2],plot_param['marker'],markersize=plot_param['markersize'],label='Distance error [m] = {0:.3g}$\pm${1:.3g}'.format(np.mean(Data['Simu_Mean_Distance_Error']),np.mean(Data['STDV_Distance'])))
#         
#         # # Setting labels, legend, title and axes limits:
#         # axs0.set_xlabel('x [m]',fontsize=plot_param['axes_label_fontsize'])#,orientation=plot_param['tick_labelrotation'])
#         # axs0.set_ylabel('y [m]',fontsize=plot_param['axes_label_fontsize'])#,orientation=plot_param['tick_labelrotation'])
#         # axs0.set_zlabel('z [m]',fontsize=plot_param['axes_label_fontsize'])        
#         # axs0.set_title('Lidar pointing accuracy ['+Qlunc_yaml_inputs['Components']['Scanner']['Type']+']',fontsize=plot_param['title_fontsize'])
#         # axs0.legend()
#         # axs0.set_xlim3d(plot_param['xlim'][0],plot_param['xlim'][1])
#         # axs0.set_ylim3d(plot_param['ylim'][0],plot_param['ylim'][1])
#         # axs0.set_zlim3d(plot_param['zlim'][0],plot_param['zlim'][1])
# =============================================================================
        

        # 1. Plotting scanning points with uncertainty
        # cm = plt.get_cmap('jet')
        # cNorm = matplotlib.colors.Normalize(vmin=min(Data['Vr Uncertainty MC [m/s]'][0]), vmax=max(Data['Vr Uncertainty MC [m/s]'][0]))
        # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.plot([Lidar.optics.scanner.origin[0]],[Lidar.optics.scanner.origin[1]],[Lidar.optics.scanner.origin[2]],'ob')
        # ax.scatter(Data['x'],Data['y'], Data['z'], Data['Vr Uncertainty MC [m/s]'][0], c=scalarMap.to_rgba(Data['Vr Uncertainty MC [m/s]'][0]))
        # ax.set_xlabel('X [m]')
        # ax.set_ylabel('Y [m]')
        # ax.set_zlabel('Z [m]')
        # ax.set_xlim(-250,250)
        # ax.set_ylim(-250,250)
        # ax.set_zlim(0,250)
        # ax.set_title('Relative Uncertainty (MC)',fontsize=plot_param['title_fontsize'])
        # scalarMap.set_array(Data['Vr Uncertainty MC [m/s]'][0])
        # fig.colorbar(scalarMap,label='$V_{rad} ~ Uncertainty ~(m/s)$',shrink=0.5)                
        # plt.show()
        
        # 2. Plot Uncertainty in Vrad with theta
        
        fig,ax2=plt.subplots()        
        ax2.plot(np.degrees(Data['theta']),Data['Vr Uncertainty homo GUM [m/s]'],'b-',label='U Uniform flow GUM')
        ax2.plot(np.degrees(Data['theta']),Data['Vr Uncertainty homo MC [m/s]'],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
        # color=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp']))))   
        # pdb.set_trace()
        for ind_a in range(len(Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp'])):
            # c=next(color)
            ax2.plot(np.degrees(Data['theta']),Data['Vr Uncertainty GUM [m/s]'],'-',label='U Shear GUM (alpha={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp'][ind_a] ))
            ax2.plot(np.degrees(Data['theta']),Data['Vr Uncertainty MC [m/s]'],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')        
        ax2.legend(loc=2, prop={'size': plot_param['legend_fontsize']})
        ax2.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
        ax2.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
        ax2.set_xlim(0,90)
        ax2.set_ylim(0,1)
        plt.title('$V_{LOS}$ Uncertainty',fontsize=plot_param['title_fontsize'])
        
        
        # 3. Plot Uncertainty in Vrad with psi
        
        fig,ax3=plt.subplots()
        ax3.plot(np.degrees(Data['psi']),Data['Vr Uncertainty homo GUM [m/s]'],'b-',label='U Uniform flow GUM')
        ax3.plot(np.degrees(Data['psi']),Data['Vr Uncertainty homo MC [m/s]'],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
        # color=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp']))))   
        
        for ind_a in range(len(Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp'])):
            # c=next(color)
            ax3.plot(np.degrees(Data['psi']),Data['Vr Uncertainty GUM [m/s]'],'-',label='U Shear GUM (alpha={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp'][ind_a] ))
            ax3.plot(np.degrees(Data['psi']),Data['Vr Uncertainty MC [m/s]'],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')        
        ax3.legend(loc=2, prop={'size': plot_param['legend_fontsize']})
        ax3.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
        ax3.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
        ax3.set_xlim(-90,90)
        ax3.set_ylim(0,1)
        plt.title('$V_{LOS}$ Uncertainty',fontsize=plot_param['title_fontsize'])
        
        
        # 4.  Plot Uncertainty in Vrad with rho        
        fig,ax4=plt.subplots()
        ax4.plot(Data['rho'],Data['Vr Uncertainty homo GUM [m/s]'],'b-',label='U Uniform flow GUM')
        ax4.plot(Data['rho'],Data['Vr Uncertainty homo MC [m/s]'],'ob' , markerfacecolor=(1, 1, 0, 0.5),label='U uniform MC')
        # color=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp']))))   
        
        for ind_a in range(len(Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp'])):
            # c=next(color)
            ax4.plot(Data['rho'],Data['Vr Uncertainty GUM [m/s]'],'-',label='U Shear GUM (alpha={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['PL_exp'][ind_a] ))
            ax4.plot(Data['rho'],Data['Vr Uncertainty MC [m/s]'],'or' , markerfacecolor=(1, 1, 0, 0.5),label='U shear MC')      
        ax4.legend(loc=2, prop={'size': plot_param['legend_fontsize']})
        ax4.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
        ax4.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
        ax4.set_xlim(0,5000)
        ax4.set_ylim(0,1) 
        
       # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join((
        r'$\rho~ [m]=%.2f$' % (Data['rho'][0], ),
        r'$\varphi~ [°]=%.2f$' % (np.degrees(Data['psi'][0]), ),
        r'N={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['N_MC'], ),
        r'Href [m]={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['Href'], )))

        
        # place a tex1t box in upper left in axes coords
        ax2.text(0.5, 0.95, textstr, transform=ax2.transAxes, fontsize=14,horizontalalignment='left',verticalalignment='top', bbox=props)
        ax2.set_xlabel('elevation [°]',fontsize=plot_param['axes_label_fontsize'])
        ax2.set_ylabel('Uncertainty [m/s]',fontsize=plot_param['axes_label_fontsize'])
        ax2.grid(axis='both')
        plt.title('$V_{LOS}$ Uncertainty',fontsize=plot_param['title_fontsize'])
        plt.show()
        


        # these are matplotlib.patch.Patch properties
        props3 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr3 = '\n'.join((
        r'$\rho ~[°]=%.2f$' % (Data['rho'][0], ),
        r'$\theta~ [°]=%.2f$' % (np.degrees(Data['theta'][0]), ),
        r'N={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['N_MC'], ),
        r'Href [m]={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['Href'], )))

        ax3.text(0.5, 0.95, textstr3, transform=ax3.transAxes, fontsize=14,horizontalalignment='left',verticalalignment='top', bbox=props3)
        ax3.set_xlabel('Azimuth [°]',fontsize=plot_param['axes_label_fontsize'])
        ax3.set_ylabel('Uncertainty [m/s]',fontsize=plot_param['axes_label_fontsize'])
        ax3.grid(axis='both')
        plt.title('$V_{LOS}$ Uncertainty',fontsize=plot_param['title_fontsize'])
        plt.show()
        
        
        
        # these are matplotlib.patch.Patch properties
        props4 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr4 = '\n'.join((
        r'$\theta~ [°]=%.2f$' % (np.degrees(Data['theta'][0]), ),
        r'$\varphi~ [°]=%.2f$' % (np.degrees(Data['psi'][0]), ),
        r'N={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['N_MC'], ),
        r'Href [m]={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['Href'], )))
        
        ax4.text(0.5, 0.95, textstr4, transform=ax3.transAxes, fontsize=14,horizontalalignment='left',verticalalignment='top', bbox=props4)
        ax4.set_xlabel('range [m]',fontsize=25)
        ax4.set_ylabel('Uncertainty [m/s]',fontsize=25)
        ax4.grid(axis='both')
        plt.title('Vrad Uncertainty',fontsize=plot_param['title_fontsize'])
        plt.show()
        
        # 5.  Plot Uncertainty in VLOS with wind direction 
        fig,ax5=plt.subplots()
        ax5.plot(np.degrees(Data['wind direction']),Data['VLOS1 Uncertainty MC [m/s]'],'.k',np.degrees(Data['wind direction']),Data['VLOS1 Uncertainty GUM [m/s]'],'-k',label='$V_{LOS_{1}}$')
        plt.title('$V_{LOS}$ uncertainty $vs$ wind direction',fontsize=plot_param['title_fontsize'])
        ax5.set_xlabel('Wind Direction (°)',fontsize=plot_param['axes_label_fontsize'])
        ax5.set_ylabel('Vlos uncertainty [m/s]',fontsize=plot_param['axes_label_fontsize'])
        ax5.grid(axis='both')       
        ax5.legend(loc=2, prop={'size': plot_param['legend_fontsize']})  
###############   Plot photodetector noise   #############################       
    if flag_plot_photodetector_noise:
        # Quantifying uncertainty from photodetector and interval domain for the plot Psax is define in the photodetector class properties)
        Psax=(Lidar.photonics.photodetector.Power_interval)*Lidar.photonics.photodetector.Active_Surf
        # Plotting:
        fig,axs1=plt.subplots()
        label0=['Shot SNR','Thermal SNR','Dark current SNR','Total SNR','TIA SNR']
        i_label=0
        for i in Data['SNR_data_photodetector']:            
            axs1.plot(Psax,Data['SNR_data_photodetector'][i][0],label=label0[i_label])  
            i_label+=1
        # axs1.plot(Psax,Data['Total_SNR_data'],label='Total SNR')
        axs1.set_xlabel('Input Signal optical power [W]',fontsize=plot_param['axes_label_fontsize'])
        axs1.set_ylabel('SNR [dB]',fontsize=plot_param['axes_label_fontsize'])
        axs1.legend(fontsize=plot_param['legend_fontsize'])
        axs1.set_title('SNR - Photodetector',fontsize=plot_param['title_fontsize'])
        axs1.grid(axis='both')
        axs1.text(.90,.05,plot_param['Qlunc_version'],transform=axs1.transAxes, fontsize=14,verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


###############   Plot Probe Volume parameters    ############################
    if flag_probe_volume_param: 
        # pdb.set_trace()
        # typeLidar ="CW"
        wave      = Qlunc_yaml_inputs['Components']['Laser']['Wavelength']  # wavelength
        f_length  = Qlunc_yaml_inputs['Components']['Telescope']['Focal length'] # focal length
        a         = np.arange(2e-3,4e-3,.002e-3) # distance fiber-end--telescope lens
        a0        = Qlunc_yaml_inputs['Components']['Telescope']['Fiber-lens offset'] # the offset (a constant number), to avoid the fiber-end locates at the focal point, otherwise the lights will be parallel to each other
        A         = Qlunc_yaml_inputs['Components']['Telescope']['Output beam radius'] # beam radius at the output lens
        ext_coef  = 1
        # effective_radius_telescope  = 16.6e-3
        s = 0 # distance from telescope to the target
        # The focus distance varies with the distance between the fiber-end and the telescope lens. So that, also the probe length varies with such distance.
        #Calculating focus distance depending on the distance between the fiber-end and the telescope lens:
        focus_distance = 1/((1/f_length)-(1/(a+a0))) # Focus distance
        dist =(np.linspace(0,80,len(a)))  # distance from the focus position along the beam direction
        # pdb.set_trace()
        # Rayleigh length variation due to focus_distance variations (due to the distance between fiber-end and telescope lens)
        zr = (wave*(focus_distance**2))/(np.pi*(Qlunc_yaml_inputs['Components']['Telescope']['Effective radius telescope'])**2)# Rayleigh length  (considered as the probe length) # half-width of the weighting function --> FWHM = 2*zr
    
        # Probe volume:
        #Probe_volume = np.pi*(A**2)*((4*(focus_distance**2)*wave)/(Telescope_aperture)) # based on Marijn notes
        vol_zr       = np.pi*(A**2)*(2*zr) # based on the definition of Rayleigh length in Liqin Jin notes (Focus calibration formula)
        
        # Lorentzian weighting function:
        
        phi = (ext_coef/np.pi)*(zr/((zr**2)+(s-focus_distance)**2))
        # phi = (ext_coef/np.pi)*(zr/((zr**2)))
        # pdb.set_trace()
        # Plotting
        fig=plt.figure()
        axs2=fig.add_subplot(2,2,1)
        axs2.plot(dist,phi)
        axs2.set_yscale('log')
        axs2.title.set_text('Weighting function')
        axs2.set_xlabel('focus distance [m]',fontsize=plot_param['axes_label_fontsize'])
        axs2.set_ylabel('$\phi$ [-]',fontsize=plot_param['axes_label_fontsize'])

        axs3=fig.add_subplot(2,2,2)
        axs3.plot(focus_distance,zr)
        # axs3.set_xlabel('focus distance [m]',fontsize=plot_param['axes_label_fontsize'])
        axs3.set_ylabel('{} [m]'.format('$\mathregular{z_{R}}$'),fontsize=plot_param['axes_label_fontsize'])
        
        axs4=fig.add_subplot(2,2,3)
        axs4.plot(a,zr)
        axs4.set_xlabel('(a+a0) [m]',fontsize=plot_param['axes_label_fontsize'])
        axs4.set_ylabel('{} [m]'.format('$\mathregular{z_{R}}$'),fontsize=plot_param['axes_label_fontsize'])
        
        
        axs5=fig.add_subplot(2,2,4)
        axs5.plot(focus_distance,a)
        axs5.set_xlabel('focus distance [m]',fontsize=plot_param['axes_label_fontsize'])
        axs5.set_ylabel('(a+a0) [m]',fontsize=plot_param['axes_label_fontsize'])
    
        # Titles and axes
        
        axs3.title.set_text('Rayleigh Vs focus distance')
        axs4.title.set_text('Rayleigh Vs Fiber-end/lens')
        axs5.title.set_text('Fiber-end/lens distance Vs focus distance')
    
    
    





###############   Plot optical amplifier noise   #############################    
    if flag_plot_optical_amplifier_noise:
        # Quantifying uncertainty from photodetector and interval domain for the plot Psax is define in the photodetector class properties)
        # Psax=10*np.log10(np.linspace(0,20e-3,1000))
        # Psax=(Lidar.photonics.photodetector.Power_interval)*Lidar.photonics.photodetector.Active_Surf
        
        # Plotting:
        fig=plt.figure()
        axs=fig.subplots()
        label0=['Optical amplifier OSNR']
        axs.plot(Lidar.photonics.optical_amplifier.Power_interval,Data['OSNR'],label=label0[0])  
        # axs1.plot(Lidar.photonics.optical_amplifier.Power_interval,Data['OSNR'],label=label0[0],marker='o')  

        axs.set_xlabel('Input Signal optical power [W]',fontsize=plot_param['axes_label_fontsize'])
        axs.set_ylabel('OSNR [dB]',fontsize=plot_param['axes_label_fontsize'])
        axs.legend(fontsize=plot_param['legend_fontsize'])
        axs.set_title('OSNR - Optical Amplifier',fontsize=plot_param['title_fontsize'])
        axs.grid(axis='both')
        axs.text(.90,.05,plot_param['Qlunc_version'],transform=axs.transAxes, fontsize=14,verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        # pdb.set_trace()
        
    if flag_plot_pointing_unc:
        # pdb.set_trace()
        fig,ax1=plt.subplots()       
        # plt.plot(np.degrees(wind_direction),u_wind_GUM,'-g', label='Uncertainty u component - GUM')
        # plt.plot(np.degrees(wind_direction),v_wind_GUM,'-b', label='Uncertainty v component - GUM')
        
        # plt.annotate('psi1_psi2_corr_n='+str(psi1_psi2_corr_n)+'\n'+'theta1_theta2_corr_n='+str(theta1_theta2_corr_n)+'\nrho1_rho2_corr_n='+str(rho1_rho2_corr_n)+'\nU_psi='+str(rho1_rho2_corr_n), xy=(0.05, 0.9), xycoords='axes fraction')
        # pdb.set_trace()
        ax1.plot(np.degrees(Data['wind direction']),Data['Vh Uncertainty GUM [m/s]'],'-r', label='Uncertainty Vh - GUM')
        # plt.annotate('psi1_psi2_corr_n='+str(psi1_psi2_corr_n)+'\n'+'theta1_theta2_corr_n='+str(theta1_theta2_corr_n)+'\nrho1_rho2_corr_n='+str(rho1_rho2_corr_n), xy=(0.05, 0.9), xycoords='axes fraction')
        
        
        ax1.plot(np.degrees(Data['wind direction']),Data['u Uncertainty [m/s]'],'og',alpha=0.3,label=' Uncertainty u component - MC')
        # plt.annotate('Uncertainty U', xy=(0.05, 0.9), xycoords='axes fraction')
        
        # plt.figure()
        ax1.plot(np.degrees(Data['wind direction']),Data['v Uncertainty [m/s]'],'ob',alpha=0.3,label='Uncertainty v component - MC')
        # plt.annotate('Uncertainty V', xy=(0.05, 0.9), xycoords='axes fraction')
        ax1.plot(np.degrees(Data['wind direction']),Data['Vh Uncertainty MC [m/s]'],'or',alpha=0.3, label='Uncertainty Vh - MC')
        ax1.set_xlabel('Wind Direction (°)',fontsize=plot_param['axes_label_fontsize'])
        ax1.set_ylabel('Vh uncertainty [m/s]',fontsize=plot_param['axes_label_fontsize'])
        ax1.legend(loc=2, prop={'size': plot_param['legend_fontsize']})
        ax1.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
        ax1.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
        plt.title('Vh Uncertainty',fontsize=plot_param['title_fontsize'])
        ax1.legend(loc=2, prop={'size': plot_param['legend_fontsize']})
        ax1.grid(axis='both')
    
        
        
        
        
        
        
        
        