# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:18:05 2020

@author: fcosta
"""

#%% Plotting:
    
plot_param={'axes_label_fontsize' : 16,
            'textbox_fontsize'    : 14,
            'title_fontsize'      : 24,
            'suptitle_fontsize'   : 23,
            'legend_fontsize'     : 12,
            'xlim'                : [-50,50],
            'ylim'                : [-50,50],
            'zlim'                : [0,130],
            'markersize'          : 5,
            'markersize_lidar'    : 9,
            'marker'              : '.',
            'markerTheo'          : '.b',
            'tick_labelrotation'  : 45}

# Plot parameters:
if flags.flag_plot_pointing_accuracy_unc:
    print('PACO')
    pdb.set_trace()
    
#########    # Scanner pointing accuracy uncertainty:#################
    
#    Calculating inputs for plotting:
    Scanner_Data1 = Lidar1.optics.scanner.Uncertainty(Lidar1,Atmospheric_Scenario,cts)
    Scanner_Data2 = Lidar2.optics.scanner.Uncertainty(Lidar2,Atmospheric_Scenario,cts)
    Scanner_Data3 = Lidar3.optics.scanner.Uncertainty(Lidar3,Atmospheric_Scenario,cts)
    
    # Creating the figure and the axes
    fig,(axs1,axs2,axs3) = plt.subplots(1,3,sharey=False) 
#    ax.plot.errorbar(Lidar1.optics.scanner.focus_dist,,plot_param['marker'],markersize=6.5,label='stdv Distance')
    
    # fitting the results to a straight line
    z1 = np.polyfit(Lidar1.optics.scanner.focus_dist, Scanner_Data1['Simu_Mean_Distance'], 1) # With '1' is a straight line y=ax+b
    f1 = np.poly1d(z1)
    # calculate new x's and y's
    x_new1 = np.linspace(Lidar1.optics.scanner.focus_dist[0], Lidar1.optics.scanner.focus_dist[-1], 50)
    y_new1 = f1(x_new1)

    
    z2 = np.polyfit(Scanner2.theta, Scanner_Data2['Simu_Mean_Distance'], 1) # With '1' is a straight line y=ax+b
    f2 = np.poly1d(z2)
    # calculate new x's and y's
    x_new2 = np.linspace(Scanner2.theta[0], Scanner2.theta[-1], 50)
    y_new2 = f2(x_new2)

   
    z3 = np.polyfit(Scanner3.phi, Scanner_Data3['Simu_Mean_Distance'], 1) # With '1' is a straight line y=ax+b
    f3 = np.poly1d(z3)
    # calculate new x's and y's
    x_new3 = np.linspace(Scanner3.phi[0], Scanner3.phi[-1], 50)
    y_new3 = f3(x_new3)
    
     # Plotting:
    axs1.plot(x_new1,y_new1,'r-',label='Fitted curve1')
    axs1.errorbar(Lidar1.optics.scanner.focus_dist,Scanner_Data1['Simu_Mean_Distance'],yerr=Scanner_Data1['STDV_Distance'],label='Data1')
    axs2.plot(x_new2,y_new2,'r-',label='Fitted curve2')
    axs2.errorbar(Scanner2.theta,Scanner_Data2['Simu_Mean_Distance'],yerr=Scanner_Data2['STDV_Distance'],label='Data2')
    axs3.plot(x_new3,y_new3,'r-',label='Fitted curve3')
    axs3.errorbar(Scanner3.phi,Scanner_Data3['Simu_Mean_Distance'],yerr=Scanner_Data3['STDV_Distance'],label='Data3')
    
    #Title and axis labels for the different plots
    fig.suptitle('Mean Distance error and stdv of the Distance error [m]',fontsize=plot_param['suptitle_fontsize'])
    axs1.set_title('Variation with f.distance',fontsize=plot_param['title_fontsize'])
    axs1.set(xlabel='Focus Distance [m]',ylabel='Distance error [m]')
    axs1.yaxis.get_label().set_fontsize(plot_param['axes_label_fontsize'])
    axs1.xaxis.get_label().set_fontsize(plot_param['axes_label_fontsize'])
    axs2.set_title(r'Variation with $\theta$',fontsize=plot_param['title_fontsize'])
    axs2.set(xlabel=r'$\theta$'+' [°]')
    axs2.xaxis.get_label().set_fontsize(plot_param['axes_label_fontsize'])
    axs3.set_title(r'Variation with $\phi$',fontsize=plot_param['title_fontsize'])
    axs3.set(xlabel=r'$\phi$'+' [°]')
    axs3.xaxis.get_label().set_fontsize(plot_param['axes_label_fontsize'])
    axs1.set_ylim(0,10)
    axs2.set_ylim(0,10)
    axs3.set_ylim(0,10)
    # text box with the fitted polynomial on the plot
    axs1.text(np.min(x_new1),np.max(y_new1),'$y$={0:.3g}$x$+{1:.3g}'.format(z1[0],z1[1]),fontsize=plot_param['textbox_fontsize'])
    axs2.text(np.min(x_new2),np.max(y_new2),'$y$={0:.3g}$x$+{1:.3g}'.format(z2[0],z2[1]),fontsize=plot_param['textbox_fontsize'])
    axs3.text(np.min(x_new3),np.max(y_new3),'$y$={0:.3g}$x$+{1:.3g}'.format(z3[0],z3[1]),fontsize=plot_param['textbox_fontsize'])
#    

##############    Ploting scanner measuring points pattern #######################
if flags.flag_plot_measuring_points_pattern:
    # Creating the figure and the axes
    #    Calculating inputs for plotting:
    Scanner_Data1 = Lidar1.optics.scanner.Uncertainty(Lidar1,Atmospheric_Scenario,cts)
    Scanner_Data2 = Lidar2.optics.scanner.Uncertainty(Lidar2,Atmospheric_Scenario,cts)
    Scanner_Data3 = Lidar3.optics.scanner.Uncertainty(Lidar3,Atmospheric_Scenario,cts)
    
    fig,axs4 = plt.subplots()  
    axs4=plt.axes(projection='3d')
    
    axs4.plot([Lidar1.optics.scanner.origin[0]],[Lidar1.optics.scanner.origin[1]],[Lidar1.optics.scanner.origin[2]],'ob',label='{} coordinates [{},{},{}]'.format(Lidar1.LidarID,Lidar1.optics.scanner.origin[0],Lidar1.optics.scanner.origin[1],Lidar1.optics.scanner.origin[2]),markersize=plot_param['markersize_lidar'])
    axs4.plot(Scanner_Data1['MeasPoint_Coordinates'][0],Scanner_Data1['MeasPoint_Coordinates'][1],Scanner_Data1['MeasPoint_Coordinates'][2],plot_param['markerTheo'],markersize=plot_param['markersize'],label='Theoretical measuring point')
    axs4.plot(Scanner_Data1['NoisyMeasPoint_Coordinates'][0],Scanner_Data1['NoisyMeasPoint_Coordinates'][1],Scanner_Data1['NoisyMeasPoint_Coordinates'][2],plot_param['marker'],markersize=plot_param['markersize'],label='Distance error [m] = {0:.3g}$\pm${1:.3g}'.format(np.mean(Scanner_Data1['Simu_Mean_Distance']),np.mean(Scanner_Data1['STDV_Distance'])))

    axs4.plot(Scanner_Data2['MeasPoint_Coordinates'][0],Scanner_Data2['MeasPoint_Coordinates'][1],Scanner_Data2['MeasPoint_Coordinates'][2],plot_param['markerTheo'],markersize=plot_param['markersize'])
    axs4.plot(Scanner_Data2['NoisyMeasPoint_Coordinates'][0],Scanner_Data2['NoisyMeasPoint_Coordinates'][1],Scanner_Data2['NoisyMeasPoint_Coordinates'][2],plot_param['marker'],markersize=plot_param['markersize'],label='Distance error [m] = {0:.3g}$\pm${1:.3g}'.format(np.mean(Scanner_Data2['Simu_Mean_Distance']),np.mean(Scanner_Data2['STDV_Distance'])))
    
    axs4.plot(Scanner_Data3['MeasPoint_Coordinates'][0],Scanner_Data3['MeasPoint_Coordinates'][1],Scanner_Data3['MeasPoint_Coordinates'][2],plot_param['markerTheo'],markersize=plot_param['markersize'])
    axs4.plot(Scanner_Data3['NoisyMeasPoint_Coordinates'][0],Scanner_Data3['NoisyMeasPoint_Coordinates'][1],Scanner_Data3['NoisyMeasPoint_Coordinates'][2],plot_param['marker'],markersize=plot_param['markersize'],label='Distance error [m] = {0:.3g}$\pm${1:.3g}'.format(np.mean(Scanner_Data3['Simu_Mean_Distance']),np.mean(Scanner_Data3['STDV_Distance'])))
    
    axs4.set_xlabel('x [m]',fontsize=plot_param['axes_label_fontsize'])#,orientation=plot_param['tick_labelrotation'])
    axs4.set_ylabel('y [m]',fontsize=plot_param['axes_label_fontsize'])#,orientation=plot_param['tick_labelrotation'])
    axs4.set_zlabel('z [m]',fontsize=plot_param['axes_label_fontsize'])
#    
    axs4.set_title('Scanner pointing accuracy',fontsize=plot_param['title_fontsize'])
#    axs4.text(np.min(x_new3),np.max(y_new3),'$y$={0:.3g}$x$+{1:.3g}'.format(z3[0],z3[1]),fontsize=plot_param['textbox_fontsize'])
    axs4.legend()
#    axs4.text(np.min(x_new1),np.max(y_new1),'$y$={0:.3g}$x$+{1:.3g}'.format(z1[0],z1[1]))
    axs4.set_xlim3d(plot_param['xlim'][0],plot_param['xlim'][1])
    axs4.set_ylim3d(plot_param['ylim'][0],plot_param['ylim'][1])
    axs4.set_zlim3d(plot_param['zlim'][0],plot_param['zlim'][1])
#    plt.rcParams['legend.fontsize'] = plot_param['legend_fontsize']
    
   
###########   Plot photodetector noise   #############################       
if flags.flag_plot_photodetector_noise:
    UQ_photo = Lidar1.photonics.photodetector.Uncertainty(Lidar1,Atmospheric_Scenario,cts) # Obtain the UQ photodetector dictionary wit SNR and UQ information
    
    Psax=10*np.log10(Lidar1.photonics.photodetector.Power_interval) 
    plt.figure()
    #plt.xscale('log',basex=10)
    #plt.yscale('log',basey=10)
    
    plt.plot(Psax,UQ_photo['SNR_data_photo']['SNR_Shot_Noise'][0],Psax,UQ_photo['SNR_data_photo']['SNR_Thermal'][0],Psax,UQ_photo['SNR_data_photo']['SNR_Dark_Current'][0],Psax,UQ_photo['SNR_data_photo']['SNR_TIA'][0])
    plt.xlabel('Input Signal optical power (dBm)',fontsize=plot_param['axes_label_fontsize'])
    plt.ylabel('SNR (dB)',fontsize=plot_param['axes_label_fontsize'])
    plt.legend(['Shot Noise','Thermal Noise','Dark current Noise','TIA Noise'],fontsize=plot_param['legend_fontsize'])#,'Total error [w]'])
    plt.title('SNR Photodetector',fontsize=plot_param['title_fontsize'])
    plt.grid(axis='both')
       




