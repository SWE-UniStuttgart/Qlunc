# -*- coding: utf-8 -*-
""".

Created on Tue Oct 20 21:18:05 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c) 

"""
#%% import packages:
from Utils.Qlunc_ImportModules import *

#%% Plotting:
def plotting(Lidar,Atmospheric_Scenario,Qlunc_yaml_inputs,Data,flag_plot_photodetector_noise, flag_plot_wind_direction_unc,flag_plot_wind_velocity_unc,flag_plot_LOS_unc,flag_plot_PDFs,flag_plot_CIs):
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
    """
    # save whole figure 
    pickle.dump(fig, open("C:/Users/fcosta/Desktop/test_fig.pickle", "wb"))

    # load figure from file
    fig = pickle.load(open("C:/Users/fcosta/Desktop/test_fig.pickle", "rb"))
    """
    # Ploting general parameters:
    plot_param={'axes_label_fontsize' : 40,
                'textbox_fontsize'    : 14,
                'title_fontsize'      : 29,
                'suptitle_fontsize'   : 23,
                'legend_fontsize'     : 23,
                'xlim'                : [-280,280],
                'ylim'                : [-280,280],
                'zlim'                : [-280,280],
                'linewidth'           : 2.25,
                'markersize'          : 5,
                'markersize_lidar'    : 9,
                'marker'              : '.r',
                'marker_face_color'   : [1,1,0,.39],
                'markerTheo'          : '.b',
                'tick_labelrotation'  : 45,
                'tick_labelfontsize'  : 30,
                'tick_labelfontsize_scy'  : 20,

                'Qlunc_version'       : 'Qlunc Version - 1.0'
                }
        


    # ##########################
    # Wind direction uncertainty 
    ############################
    if flag_plot_wind_direction_unc:
        # pdb.set_trace()
        # 0. Plot Uncertainty in /Omega against wind direction             
        color1   = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))
        
        
        #Sort the elements of wind direction (necessary when working with TimeSeries)
        
        # Sort wind direction and get the sorting indexes
        windDirection_sortI =  np.argsort(Data['wind direction']) 
        WindDir             =  np.sort(Data['wind direction'])        
        
        # Sort Uncertainty vectors with the indexes WindDirection_sortI
        # U_WindVel_p   = [np.concatenate( Data['Vh Unc [°]']['Uncertainty Vh GUM'][i], axis=0 ) for i in range(len(Data['Vh Unc [°]']['Uncertainty Vh GUM']))] # Transforming format        
        # UWindVel_GUM  = [U_WindVel_p[i][windDirection_sortI.argsort()] for i in range(len(U_WindVel_p))]
        # UWindVel_MCM  = [np.array(Data['Vh Unc [°]']['Uncertainty Vh MCM'][i])[windDirection_sortI.argsort()] for i in range(len(Data['Vh Unc [°]']['Uncertainty Vh MCM'])) ]


        U_WindDir_p   = [np.concatenate( Data['WinDir Unc [°]']['Uncertainty wind direction GUM'][i], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))] # Transforming format        
        UWindDir_GUM  = [U_WindDir_p[i][windDirection_sortI.argsort()] for i in range(len(U_WindDir_p))]
        UWindDir_MCM  = [np.array(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'][i])[windDirection_sortI.argsort()] for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])) ]
        
        Sens_coeff_Vlos_W1=[np.concatenate( Data['Sens coeff Vlos']['W1'][i], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]
        Sens_coeff_Vlos_W2=[np.concatenate( Data['Sens coeff Vlos']['W2'][i], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]
        Sens_coeff_Vlos_W12=[np.concatenate( Data['Sens coeff Vlos']['W1W2'][i], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]
        Sens_coeff_Vlos_W4=[np.concatenate( Data['Sens coeff Vlos']['W4'][i], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]
        Sens_coeff_Vlos_W5=[np.concatenate( Data['Sens coeff Vlos']['W5'][i], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]
        Sens_coeff_Vlos_W6=[np.concatenate( Data['Sens coeff Vlos']['W6'][i], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]

        Sens_coeff_Vlos_W1  = [Sens_coeff_Vlos_W1[i][windDirection_sortI.argsort()] for i in range(len(U_WindDir_p))]
        Sens_coeff_Vlos_W2  = [Sens_coeff_Vlos_W2[i][windDirection_sortI.argsort()] for i in range(len(U_WindDir_p))]
        Sens_coeff_Vlos_W12  = [Sens_coeff_Vlos_W12[i][windDirection_sortI.argsort()] for i in range(len(U_WindDir_p))]
        Sens_coeff_Vlos_W4  = [Sens_coeff_Vlos_W4[i][windDirection_sortI.argsort()] for i in range(len(U_WindDir_p))]
        Sens_coeff_Vlos_W5  = [Sens_coeff_Vlos_W5[i][windDirection_sortI.argsort()] for i in range(len(U_WindDir_p))]
        Sens_coeff_Vlos_W6  = [Sens_coeff_Vlos_W6[i][windDirection_sortI.argsort()] for i in range(len(U_WindDir_p))]



        if len(Lidar.optics.scanner.origin)==3:
            fig0,ax0 = plt.subplots(3,1)
            fig0.tight_layout()
            # legt = [r'$\frac{\partial^2{\Omega}}{\partial{V_{LOS_1}}}\sigma^2_{V_{LOS_{1}}}$',r'$\frac{\partial^2{\Omega}}{\partial{V_{LOS_2}}}\sigma^2_{V_{LOS_{2}}}$',r'$\frac{\partial^2{\Omega}}{\partial{V_{LOS_{3}}}}\sigma^2_{V_{LOS_{2,3}}}$'
            #         ,r'$\frac{\partial{\Omega}}{\partial{V_{LOS_{1,2}}}}\sigma_{V_{LOS_{1,2}}}$',r'$\frac{\partial{\Omega}}{\partial{V_{LOS_{1,3}}}}\sigma_{V_{LOS_{1,3}}}$',r'$\frac{\partial{\Omega}}{\partial{V_{LOS_{2,3}}}}\sigma_{V_{LOS_{2,3}}}$']
            
            legt = [r'$T_{\Omega,V_{LOS_1}}$',r'$T_{\Omega,V_{LOS_2}}$',r'$T_{\Omega,V_{LOS_3}}$',r'$T_{\Omega,V_{LOS_{12}}}$',r'$T_{\Omega,V_{LOS_{13}}}$',r'$T_{\Omega,V_{LOS_{23}}}$']
           
            
            ax0[1].plot(np.degrees(WindDir),Sens_coeff_Vlos_W1[0],'-',marker='^',markevery=3,linewidth=plot_param['linewidth'], color='darkgrey',label=legt[0])
            ax0[1].plot(np.degrees(WindDir),Sens_coeff_Vlos_W2[0],'-',marker='o',markevery=3,linewidth=plot_param['linewidth'], color='dimgray',label=legt[1])     
            ax0[1].plot(np.degrees(WindDir),Sens_coeff_Vlos_W12[0],'-',marker='X',markevery=3,linewidth=plot_param['linewidth'], color='black',label=legt[2])
            ax0[2].plot(np.degrees(WindDir),Sens_coeff_Vlos_W4[0],'-',marker='^',markevery=3,linewidth=plot_param['linewidth'], color='darkgrey',label=legt[3])
            ax0[2].plot(np.degrees(WindDir),Sens_coeff_Vlos_W5[0],'-',marker='o',markevery=3,linewidth=plot_param['linewidth'], color='dimgray',label=legt[4])
            ax0[2].plot(np.degrees(WindDir),Sens_coeff_Vlos_W6[0],'-',marker='X',markevery=3,linewidth=plot_param['linewidth'], color='black',label=legt[5])
                    
            
            # pdb.set_trace()
            	# Axes:
                    
            ax0[0].set_ylabel(r'$u_{\Omega}$ [°]',fontsize=plot_param['axes_label_fontsize'])          
            ax0[0].tick_params(axis='both', labelsize=plot_param['tick_labelfontsize'])
            ax0[0].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))
            ax0[0].set_ylim(0,5)
            ax0[0].grid(axis='both')
            ax0[0].tick_params(axis='x',label1On=False)
            
            
            ax0[1].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax0[1].set_ylabel(r'$T_{\Omega}$ [°]',fontsize=plot_param['axes_label_fontsize']-2.3)
            ax0[1].ticklabel_format(axis='y',style='sci', scilimits=(0,0))          
            ax0[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
            ax0[1].tick_params(axis='both', labelsize=plot_param['tick_labelfontsize'])
            ax0[1].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))
            ax0[1].grid(axis='both')
            ax0[1].tick_params(axis='x',label1On=False)

            ax0[2].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax0[2].set_ylabel(r'$T_{\Omega}$ [°]',fontsize=plot_param['axes_label_fontsize']-2.3)
            ax0[2].ticklabel_format(axis='y',style='sci', scilimits=(0,0))          
            ax0[2].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
            ax0[2].set_xlabel('Wind direction[°]',fontsize=plot_param['axes_label_fontsize'])
            ax0[2].tick_params(axis='both', labelsize=plot_param['tick_labelfontsize'])
            ax0[2].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))
            ax0[2].grid(axis='both')

            props0 = dict(boxstyle='round', facecolor='wheat', alpha=0.4)        
            textstr0 = '\n'.join((
            r'$r_{\theta_{1},\theta_{2}} ~=%.2f$' % ( Lidar.optics.scanner.correlations[3] ),
            r'$r_{\theta_{1},\theta_{3}} ~=%.2f$' % ( Lidar.optics.scanner.correlations[4] ),
            r'$r_{\theta_{2},\theta_{3}} ~=%.2f$' % ( Lidar.optics.scanner.correlations[5] ),
            
            r'$r_{\varphi_{1},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[0] ),
            r'$r_{\varphi_{1},\varphi_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[1] ),
            r'$r_{\varphi_{2},\varphi_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[2] ),
            
            r'$r_{\rho_{1},\rho_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[6]),
            r'$r_{\rho_{1},\rho_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[7]),
            r'$r_{\rho_{2},\rho_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[8]),
            
            r'$r_{\theta_{1},\varphi_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[9]),
            r'$r_{\theta_{2},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[10]),
            r'$r_{\theta_{3},\varphi_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[11]),
            
            r'$r_{\varphi_{1},\theta_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[12]),
            r'$r_{\varphi_{1},\theta_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[13]),
            r'$r_{\varphi_{2},\theta_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[14]),
            
            r'$r_{\varphi_{2},\theta_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[15]),
            r'$r_{\varphi_{3},\theta_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[16]),
            r'$r_{\varphi_{3},\theta_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[17])))    
            # Place textbox
            # ax0[0].text(.92, 0.80, textstr0,  fontsize = 16,horizontalalignment = 'left',verticalalignment = 'top', bbox = props0, transform=plt.gcf().transFigure)     
            # Size of the graphs
            plt.subplots_adjust(left=0.075, right=0.995, bottom=0.11, top=0.975, wspace=0.3, hspace=0.15)            
            
            # Legend
            ax0[1].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
        
        else:
            fig0,ax0 = plt.subplots(2,1)
            fig0.tight_layout()
            legt = [r'$T_{\Omega,V_{LOS_1}}$',r'$T_{\Omega,V_{LOS_2}}$',r'$T_{\Omega,V_{LOS_{12}}}$']
            ax0[1].plot(np.degrees(WindDir),Sens_coeff_Vlos_W1[0],'-',marker='^',markevery=3,linewidth=plot_param['linewidth'], color='black',label=legt[0])
            ax0[1].plot(np.degrees(WindDir),Sens_coeff_Vlos_W2[0],'-',marker='^',markevery=3,linewidth=plot_param['linewidth'], color='dimgray',label=legt[1])     
            ax0[1].plot(np.degrees(WindDir),Sens_coeff_Vlos_W12[0],'-',marker='^',markevery=3,linewidth=plot_param['linewidth'], color='cadetblue',label=legt[2])
                        
        

            	# Axes:
                    
            ax0[0].set_ylabel(r'$u_{\Omega}$ [°]',fontsize=plot_param['axes_label_fontsize'])          
            ax0[0].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax0[0].tick_params(axis='both', labelsize=plot_param['tick_labelfontsize'])
            ax0[0].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))
            ax0[0].set_ylim(0.,2)
            ax0[0].grid(axis='both')
            ax0[0].tick_params(axis='x',label1On=False)

            ax0[1].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax0[1].set_ylabel(r'$T_{\Omega}$ [°]',fontsize=plot_param['axes_label_fontsize']-2.3)
            ax0[1].ticklabel_format(axis='y',style='sci', scilimits=(0,0))          
            ax0[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
            ax0[1].set_xlabel('Wind direction[°]',fontsize=plot_param['axes_label_fontsize'])
            ax0[1].tick_params(axis='both', labelsize=plot_param['tick_labelfontsize'])
            ax0[1].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))
            ax0[1].grid(axis='both')

            # Print the box with the correlation coefficients
            props0 = dict(boxstyle='round', facecolor='wheat', alpha=0.4)        
            textstr0 = '\n'.join((
            r'$r_{\theta_{1},\theta_{2}} ~=%.2f$' % ( Lidar.optics.scanner.correlations[3] ),               
            r'$r_{\varphi_{1},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[0] ),               
            r'$r_{\rho_{1},\rho_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[6]),              
            r'$r_{\theta_{1},\varphi_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[9]),              
            r'$r_{\varphi_{1},\theta_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[12]),
            r'$r_{\varphi_{2},\theta_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[14])))    

            # ax0[0].text(.92, 0.80, textstr0,  fontsize = 16,horizontalalignment = 'left',verticalalignment = 'top', bbox = props0, transform=plt.gcf().transFigure) 
        
        
        
        for ind_plot in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])):
            
            cc=next(color1)
            ax0[0].plot(np.degrees(WindDir),UWindDir_GUM[ind_plot],'-', color=cc,label=r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
            ax0[0].plot(np.degrees(WindDir),UWindDir_MCM[ind_plot],'o', markerfacecolor=cc,markeredgecolor='lime',alpha=0.4,label='MCM')        
        
        # Legend
        ax0[0].legend(loc=1, prop={'size': plot_param['legend_fontsize']-5})
        plt.subplots_adjust(left=0.075, right=0.995, bottom=0.085, top=0.975, wspace=0.3, hspace=0.11)            
        plt.show()                

    # sdf_MCM[sort_in.argsort()]
    # #######################################
    # Wind velocity uncertainty (Vh or Vwind) 
    #########################################
    if flag_plot_wind_velocity_unc:
        #Sort the elements of wind direction (necessary when working with TimeSeries)
        
        # Sort wind direction and get the sorting indexes
        windDirection_sortI =  np.argsort(Data['wind direction']) 
        WindDir             =  np.sort(Data['wind direction'])        
        
        # Sort Uncertainty vectors with the indexes WindDirection_sortI
        U_Vh_p   = [np.concatenate( Data['Vh Unc [m/s]']['Uncertainty Vh GUM'][i], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))] # Transforming format        
        UVh_GUM  = [U_Vh_p[i][windDirection_sortI.argsort()] for i in range(len(U_Vh_p))]
        UVh_MCM  = [np.array(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'][i])[windDirection_sortI.argsort()] for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])) ]
        CorrelationsV12_GUM=[np.array(Data['Correlations']['V12_GUM'][i])[windDirection_sortI.argsort()] for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])) ]
        CorrelationsV12_MCM=[np.array(Data['Correlations']['V12_MCM'][i])[windDirection_sortI.argsort()] for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])) ]
        
             
        Sens_coeff_Vlos_V1  = [np.concatenate( Data['Sens coeff Vh'][i]['dV1'], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]
        Sens_coeff_Vlos_V2  = [np.concatenate( Data['Sens coeff Vh'][i]['dV2'], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]
        Sens_coeff_Vlos_V12 = [np.concatenate( Data['Sens coeff Vh'][i]['dV1V2'], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]


        Sens_coeff_Vlos_V1   = [Sens_coeff_Vlos_V1[i][windDirection_sortI.argsort()] for i in range(len(U_Vh_p))]
        Sens_coeff_Vlos_V2   = [Sens_coeff_Vlos_V2[i][windDirection_sortI.argsort()] for i in range(len(U_Vh_p))]
        Sens_coeff_Vlos_V12  = [Sens_coeff_Vlos_V12[i][windDirection_sortI.argsort()] for i in range(len(U_Vh_p))]





        if Lidar.optics.scanner.pattern in ['None']:               
            
            # If triple solution
            if len(Lidar.optics.scanner.origin)==3:
                Sens_coeff_Vlos_V13 = [np.concatenate( Data['Sens coeff Vh'][i]['dV1V3'], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]
                Sens_coeff_Vlos_V23 = [np.concatenate( Data['Sens coeff Vh'][i]['dV2V3'], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]
                Sens_coeff_Vlos_V3  = [np.concatenate( Data['Sens coeff Vh'][i]['dV3'], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]
                
                Sens_coeff_Vlos_V13  = [Sens_coeff_Vlos_V13[i][windDirection_sortI.argsort()] for i in range(len(U_Vh_p))]
                Sens_coeff_Vlos_V23  = [Sens_coeff_Vlos_V23[i][windDirection_sortI.argsort()] for i in range(len(U_Vh_p))]
                Sens_coeff_Vlos_V3   = [Sens_coeff_Vlos_V3[i][windDirection_sortI.argsort()] for i in range(len(U_Vh_p))]
                CorrelationsV13_GUM=[np.array(Data['Correlations']['V13_GUM'][i])[windDirection_sortI.argsort()] for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])) ]
                CorrelationsV23_GUM=[np.array(Data['Correlations']['V23_GUM'][i])[windDirection_sortI.argsort()] for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])) ]

                CorrelationsV13_MCM=[np.array(Data['Correlations']['V13_MCM'][i])[windDirection_sortI.argsort()] for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])) ]
                CorrelationsV23_MCM=[np.array(Data['Correlations']['V23_MCM'][i])[windDirection_sortI.argsort()] for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])) ]
           
                #Create subplots
                color1   = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))
                fig0,ax0 = plt.subplots(3,1)
                fig0.tight_layout()
                
                #Plot sensitivity coefficients
                SensCoeff1 = [Sens_coeff_Vlos_V1,Sens_coeff_Vlos_V2,Sens_coeff_Vlos_V3]
                SensCoeff2 = [Sens_coeff_Vlos_V12,Sens_coeff_Vlos_V13,Sens_coeff_Vlos_V23]
                c5 = ['darkgrey','dimgray','black']
                # legt1 = [r'$\frac{\partial^2{V_{wind}}}{\partial{V_{LOS_1}}}\sigma^2_{V_{LOS_{1}}}$',r'$\frac{\partial^2{V_{wind}}}{\partial{V_{LOS_2}}}\sigma^2_{V_{LOS_{2}}}$',r'$\frac{\partial^2{V_{wind}}}{\partial{V_{LOS_{3}}}}\sigma^2_{V_{LOS_{3}}}$']
                # legt2 = [r'$\frac{\partial{V_{wind}}}{\partial{V_{LOS_{1,2}}}}\sigma_{V_{LOS_{1,2}}}$',r'$\frac{\partial{V_{wind}}}{\partial{V_{LOS_{1,3}}}}\sigma_{V_{LOS_{1,3}}}$',r'$\frac{\partial{V_{wind}}}{\partial{V_{LOS_{2,3}}}}\sigma_{V_{LOS_{2,3}}}$']
                legt1 = [r'$T_{V_{wind},V_{LOS_1}}$',r'$T_{V_{wind},V_{LOS_2}}$',r'$T_{V_{wind},V_{LOS_3}}$']
                legt2 = [r'$T_{V_{wind},V_{LOS_{12}}}$',r'$T_{V_{wind},V_{LOS_{13}}}$',r'$T_{V_{wind},V_{LOS_{23}}}$']
                                
                
                markers = ['^','o','X']
                for ind_plot in range(len(SensCoeff1)):                                                 
                    ax0[1].plot(np.degrees(WindDir),SensCoeff1[ind_plot][0],'-',marker=markers[ind_plot],markevery=3,c = c5[ind_plot],linewidth=plot_param['linewidth'],label = legt1[ind_plot])
                for ind_plot in range(len(SensCoeff2)):                                                 
                    ax0[2].plot(np.degrees(WindDir),SensCoeff2[ind_plot][0],'-',marker=markers[ind_plot],markevery=3,c = c5[ind_plot],linewidth=plot_param['linewidth'],label = legt2[ind_plot])                                    

                
                # pdb.set_trace()
                for ind_plot in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])):
                    c2=next(color1)
                    # pdb.set_trace()
                    ax0[0].plot(np.degrees(WindDir),UVh_GUM[ind_plot],'-', color = c2,linewidth = plot_param['linewidth'],label = r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
                    ax0[0].plot(np.degrees(WindDir),UVh_MCM[ind_plot],'o' , markerfacecolor = c2,markeredgecolor = 'lime',alpha = 0.4,label = 'MCM')
                    # ax11.set_ylim([.095, .15])

                	# Axes:
                        
                ax0[0].set_ylabel(r'$u_{V_{wind}}$ [m/s]',fontsize=30)          
                ax0[0].tick_params(axis='both', labelsize=plot_param['tick_labelfontsize'])
                ax0[0].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))
                # ax0[0].set_ylim(0.65,1)
                ax0[0].grid(axis='both')
                ax0[0].legend(loc=1, prop={'size': plot_param['legend_fontsize']-5})
                ax0[0].tick_params(axis='x',label1On=False)
    
                ax0[1].legend(loc=1, prop={'size': plot_param['legend_fontsize']+5})
                ax0[1].set_ylabel(r'$T_{V_{wind}}$ [m/s]',fontsize=30)
                ax0[1].ticklabel_format(axis='y',style='sci', scilimits=(0,0))          
                ax0[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])

                ax0[1].tick_params(axis='both', labelsize=plot_param['tick_labelfontsize'])
                ax0[1].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))
                ax0[1].grid(axis='both')
                ax0[1].tick_params(axis='x',label1On=False)

                ax0[2].legend(loc=1, prop={'size': plot_param['legend_fontsize']+5})
                ax0[2].set_ylabel(r'$T_{V_{wind}}$ [m/s]',fontsize=30)
                ax0[2].ticklabel_format(axis='y',style='sci', scilimits=(0,0))          
                ax0[2].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
                ax0[2].set_xlabel('Wind direction[°]',fontsize=plot_param['axes_label_fontsize'])
                ax0[2].tick_params(axis='both', labelsize=plot_param['tick_labelfontsize'])
                ax0[2].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))
                ax0[2].grid(axis='both')
                # pdb.set_trace()                
                props0 = dict(boxstyle='round', facecolor='wheat', alpha=0.4)        
                textstr0 = '\n'.join((
                r'$r_{\theta_{1},\theta_{2}} ~=%.2f$' % ( Lidar.optics.scanner.correlations[3] ),
                r'$r_{\theta_{1},\theta_{3}} ~=%.2f$' % ( Lidar.optics.scanner.correlations[4] ),
                r'$r_{\theta_{2},\theta_{3}} ~=%.2f$' % ( Lidar.optics.scanner.correlations[5] ),
                
                r'$r_{\varphi_{1},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[0] ),
                r'$r_{\varphi_{1},\varphi_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[1] ),
                r'$r_{\varphi_{2},\varphi_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[2] ),
                
                r'$r_{\rho_{1},\rho_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[6]),
                r'$r_{\rho_{1},\rho_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[7]),
                r'$r_{\rho_{2},\rho_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[8]),
                
                r'$r_{\theta_{1},\varphi_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[9]),
                r'$r_{\theta_{2},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[10]),
                r'$r_{\theta_{3},\varphi_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[11]),
                
                r'$r_{\varphi_{1},\theta_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[12]),
                r'$r_{\varphi_{1},\theta_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[13]),
                r'$r_{\varphi_{2},\theta_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[14]),
                
                r'$r_{\varphi_{2},\theta_{3}}~ =%.2f$' % (Lidar.optics.scanner.correlations[15]),
                r'$r_{\varphi_{3},\theta_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[16]),
                r'$r_{\varphi_{3},\theta_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[17])))    
                
                # Place textbox
                # ax0[0].text(.92, 0.80, textstr0,  fontsize = 16,horizontalalignment = 'left',verticalalignment = 'top', bbox = props0, transform=plt.gcf().transFigure)     
                # Size of the graphs
                plt.subplots_adjust(left=0.085, right=0.995, bottom=0.11, top=0.995, wspace=0.3, hspace=0.15)            
                
                # Legend
                ax0[1].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
                
                # Plot correlations
                fig1,ax1 = plt.subplots(3,1)
                CorrelationsGUM = [CorrelationsV12_GUM,CorrelationsV13_GUM,CorrelationsV23_GUM]
                CorrelationsMCM = [CorrelationsV12_MCM,CorrelationsV13_MCM,CorrelationsV23_MCM]                
                color3 = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))
                # pdb.set_trace()
                for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                    c3 = next(color3)
                    # Plot:
                    ax1[0].plot(np.degrees(WindDir),CorrelationsGUM[0][ind_plot],'-',c = c3,linewidth = plot_param['linewidth'],label = r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))                      
                    ax1[0].plot(np.degrees(WindDir),CorrelationsMCM[0][ind_plot],'o', markerfacecolor = c3,markeredgecolor='lime',alpha=0.4,label='MCM')                    
                    ax1[1].plot(np.degrees(WindDir),CorrelationsGUM[1][ind_plot],'-',c = c3,linewidth = plot_param['linewidth'],label = r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))                      
                    ax1[1].plot(np.degrees(WindDir),CorrelationsMCM[1][ind_plot],'o', markerfacecolor = c3,markeredgecolor='lime',alpha=0.4,label='MCM')
                    ax1[2].plot(np.degrees(WindDir),CorrelationsGUM[2][ind_plot],'-',c = c3,linewidth = plot_param['linewidth'],label = r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))                      
                    ax1[2].plot(np.degrees(WindDir),CorrelationsMCM[2][ind_plot],'o', markerfacecolor = c3,markeredgecolor='lime',alpha=0.4,label='MCM')
                
                
                # Axes:
                ax1[0].set_ylim(-1,1)
                ax1[0].grid(axis = 'both')
                ax1[0].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))            
                ax1[0].tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
                ax1[1].set_ylim(-1,1)
                ax1[1].grid(axis = 'both')
                ax1[1].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))            
                ax1[1].tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
                ax1[2].set_ylim(-1,1)
                ax1[2].grid(axis = 'both')
                ax1[2].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))            
                ax1[2].tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])


                ax1[2].set_xlabel('Wind direction[°]',fontsize = plot_param['axes_label_fontsize'])
                ax1[0].set_ylabel('$r_{V_{LOS_{12}}}$ [-]',fontsize=plot_param['axes_label_fontsize'])
                ax1[1].set_ylabel('$r_{V_{LOS_{13}}}$ [-]',fontsize=plot_param['axes_label_fontsize'])
                ax1[2].set_ylabel('$r_{V_{LOS_{23}}}$ [-]',fontsize=plot_param['axes_label_fontsize'])
                plt.subplots_adjust(left=0.08, right=0.995, bottom=0.11, top=0.98, wspace=0.3, hspace=0.15)            
                ax1[0].legend(loc = 1, prop = {'size': plot_param['legend_fontsize']-5})
                ax1[0].tick_params(axis='x',label1On=False)
                ax1[1].tick_params(axis='x',label1On=False)
                # pdb.set_trace()
            # else dual solution
            else:
                c5=['black','dimgray','cadetblue']
                fig0,ax0 = plt.subplots(2,1)
                fig0.tight_layout()
                legt = [r'$T_{V_h,V_{LOS_1}}$',r'$T_{V_h,V_{LOS_2}}$',r'$T_{V_h,V_{LOS_{12}}}$']
                SensCoeff1=[Sens_coeff_Vlos_V1,Sens_coeff_Vlos_V2,Sens_coeff_Vlos_V12]

                for ind_plot in range(3):                 
                    ax0[1].plot(np.degrees(WindDir),SensCoeff1[ind_plot][0],'-',c = c5[ind_plot],linewidth = plot_param['linewidth'],label = legt[ind_plot])
                    
                	# Axes:
                        
                ax0[0].set_ylabel(r'$u_{V_h}$ [m/s]',fontsize=plot_param['axes_label_fontsize'])          
                ax0[0].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
                ax0[0].tick_params(axis='both', labelsize=plot_param['tick_labelfontsize'])
                ax0[0].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))
                # ax0[0].set_ylim(0.65,1)
                ax0[0].grid(axis='both')
    
                ax0[1].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
                ax0[1].set_ylabel(r'$T_{V_h}$ [m/s]',fontsize=plot_param['axes_label_fontsize']-2.3)
                ax0[1].ticklabel_format(axis='y',style='sci', scilimits=(0,0))          
                ax0[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
                ax0[1].set_xlabel('Wind direction[°]',fontsize=plot_param['axes_label_fontsize'])
                ax0[1].tick_params(axis='both', labelsize=plot_param['tick_labelfontsize'])
                ax0[1].set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))
                ax0[1].grid(axis='both')
                
                
                # Put a text box with correlation coefficients
                props0 = dict(boxstyle='round', facecolor='wheat', alpha=0.4)        
                textstr0 = '\n'.join((
                r'$r_{\theta_{1},\theta_{2}} ~=%.2f$' % ( Lidar.optics.scanner.correlations[3] ),               
                r'$r_{\varphi_{1},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[0] ),               
                r'$r_{\rho_{1},\rho_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[6]),              
                r'$r_{\theta_{1},\varphi_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[9]),              
                r'$r_{\varphi_{1},\theta_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[12]),
                r'$r_{\varphi_{2},\theta_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[14])))                    
                # ax0[0].text(.92, 0.80, textstr0,  fontsize = 16,horizontalalignment = 'left',verticalalignment = 'top', bbox = props0, transform=plt.gcf().transFigure) 
                
                
                # Plot  uncertainties
                color2 = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))
                for ind_plot in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])):
                    
                    c2=next(color2)
                    ax0[0].plot(np.degrees(WindDir),UVh_GUM[ind_plot],'-', color = c2,linewidth = plot_param['linewidth'],label = r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
                    ax0[0].plot(np.degrees(WindDir),UVh_MCM[ind_plot],'o' , markerfacecolor = c2,markeredgecolor = 'lime',alpha = 0.4,label = 'MCM')
                # Legend
                ax0[0].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
                plt.subplots_adjust(left=0.075, right=0.995, bottom=0.085, top=0.975, wspace=0.3, hspace=0.175)            
                plt.show()   
                
                # Plot correlations
                fig1,ax1 = plt.subplots()
                CorrelationsGUM = []
                CorrelationsMCM = []                
                color3 = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))
                # pdb.set_trace()
                for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                    c3 = next(color3)
                    # Plot:
                    ax1.plot(np.degrees(WindDir),CorrelationsV12_GUM[ind_plot],'-',c = c3,linewidth = plot_param['linewidth'],label = r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))                      
                    ax1.plot(np.degrees(WindDir),CorrelationsV12_MCM[ind_plot],'o', markerfacecolor = c3,markeredgecolor='lime',alpha=0.4,label='MCM')                    

                    # Axes:
                    ax1.set_ylim(-1,1)
                    
                    ax1.set_xlim(np.degrees(WindDir.min()),np.degrees(WindDir.max()))            
                    ax1.tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
                ax1.set_xlabel('Wind direction[°]',fontsize = plot_param['axes_label_fontsize'])
                ax1.set_ylabel('$r_{V_{LOS_{12}}}$ [-]',fontsize=plot_param['axes_label_fontsize'])
                ax1.legend(loc = 1, prop = {'size': plot_param['legend_fontsize']})
                ax1.grid(axis = 'both')
                plt.subplots_adjust(left=0.085, right=0.995, bottom=0.11, top=0.975, wspace=0.3, hspace=0.24)                            
                
            
                
        # #######################################
        # Plot the vertical/horizontal plane 
        #########################################
        
        
        
        # pdb.set_trace()
        elif Lidar.optics.scanner.pattern in ['vertical plane'] or Lidar.optics.scanner.pattern in ['horizontal plane']:
            V=[]
            Dir=[]
            for i in range(int((len(Data['Sens coeff Vh'])/len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'])))):
                V.append(Data['Vh Unc [m/s]']['Uncertainty Vh GUM'][i][0])
                Dir.append(Data['WinDir Unc [°]']['Uncertainty wind direction GUM'][i][0])         
            
            # Reshape V and avoid nans and infinit values
            VV=np.reshape(V,[int(np.sqrt(len(V))),int(np.sqrt(len(V)))])
            DirD=np.reshape(Dir,[int(np.sqrt(len(Dir))),int(np.sqrt(len(Dir)))])

            # VV[VV>5]=10
            # DirD[DirD>10]=10

            # Horizontal wind velocity
            col ='coolwarm' 
            cmaps = matplotlib.cm.get_cmap(col)  # viridis is the default colormap for imshow
            # #Horizontal plane
            cmap0 = matplotlib.cm.ScalarMappable(norm = mcolors.Normalize(vmin = 0.1, vmax = .55),cmap = plt.get_cmap(col))
            cmap1 = matplotlib.cm.ScalarMappable(norm = mcolors.Normalize(vmin = 0.9, vmax = 4),cmap = plt.get_cmap(col))
            #Vertical plane
            # cmap0 = matplotlib.cm.ScalarMappable(norm = mcolors.Normalize(vmin = 0.12, vmax = .14),cmap = plt.get_cmap(col))
            # cmap1 = matplotlib.cm.ScalarMappable(norm = mcolors.Normalize(vmin = 0.64, vmax = .91),cmap = plt.get_cmap(col))    
            
            fig00,ax00 = plt.subplots()
            fig01,ax01 = plt.subplots()
            if  Lidar.optics.scanner.pattern in ['vertical plane']:
                XX = np.reshape(Data['lidars']['Coord_Out'][1],[int(np.sqrt(len(V))),int(np.sqrt(len(V)))])
                YY = np.reshape(Data['lidars']['Coord_Out'][2],[int(np.sqrt(len(V))),int(np.sqrt(len(V)))])
                ax00.set_xlabel('Y [m]', fontsize = plot_param['tick_labelfontsize']+20, labelpad = 15)
                ax00.set_ylabel('Z [m]', fontsize = plot_param['tick_labelfontsize']+20, labelpad = 15)
                ax01.set_xlabel('Y [m]', fontsize = plot_param['tick_labelfontsize']+20, labelpad = 15)
                ax01.set_ylabel('Z [m]', fontsize = plot_param['tick_labelfontsize']+20, labelpad = 15)

            elif  Lidar.optics.scanner.pattern in ['horizontal plane']:
                XX=np.reshape(Data['lidars']['Coord_Out'][0],[int(np.sqrt(len(V))),int(np.sqrt(len(V)))])
                YY=np.reshape(Data['lidars']['Coord_Out'][1],[int(np.sqrt(len(V))),int(np.sqrt(len(V)))])
                ax00.set_xlabel('X [m]', fontsize = plot_param['tick_labelfontsize']+20, labelpad = 15)
                ax00.set_ylabel('Y [m]', fontsize = plot_param['tick_labelfontsize']+20, labelpad = 15)
                ax01.set_xlabel('X [m]', fontsize = plot_param['tick_labelfontsize']+20, labelpad = 15)
                ax01.set_ylabel('Y [m]', fontsize = plot_param['tick_labelfontsize']+20, labelpad = 15)
                
                for ind_len in range(len(Lidar.optics.scanner.origin)):
                    ax00.plot(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_len][0],Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_len][1],'sk', ms=8, mec='white', mew=1.5)
                    ax01.plot(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_len][0],Qlunc_yaml_inputs['Components']['Scanner']['Origin'][ind_len][1],'sk', ms=8, mec='white', mew=1.5)
            
            # #Horizontal plane
            ax01.contourf(XX,YY, DirD,50,cmap = cmaps,vmin =.9, vmax = 4)
            ax00.contourf(XX,YY, VV,50,cmap = cmaps,vmin = 0.1, vmax = .55)
            #Vertical plane
            # ax01.contourf(XX,YY, DirD,50,cmap = cmaps,vmin = .64, vmax = .91)
            # ax00.contourf(XX,YY, VV,50,cmap = cmaps,vmin = .12, vmax = .14) 
            
            cmap0.set_array([]) 
            cmap1.set_array([]) 
            colorbar0 = fig00.colorbar(cmap0, ax = ax00) 
            colorbar1 = fig01.colorbar(cmap1, ax = ax01)                        
            colorbar0.set_label(label = 'Uncertainty [m/s]', size = plot_param['tick_labelfontsize']+15, labelpad = 15)
            colorbar0.ax.tick_params(labelsize = 25)
            colorbar1.set_label(label = 'Uncertainty [°]', size = plot_param['tick_labelfontsize']+15, labelpad = 15)
            colorbar1.ax.tick_params(labelsize = 25)

            ax00.set_aspect('equal')
            ax00.ticklabel_format(useOffset=False)
            ax01.set_aspect('equal')
            ax01.ticklabel_format(useOffset=False)
            ax00.locator_params(axis='x', nbins=5)
            ax00.locator_params(axis='y', nbins=5)
            ax01.locator_params(axis='x', nbins=5)
            ax01.locator_params(axis='y', nbins=5)                
            ax00.xaxis.set_tick_params(labelsize = plot_param['tick_labelfontsize']+14)
            ax00.yaxis.set_tick_params(labelsize = plot_param['tick_labelfontsize']+14)
            ax01.xaxis.set_tick_params(labelsize = plot_param['tick_labelfontsize']+14)
            ax01.yaxis.set_tick_params(labelsize = plot_param['tick_labelfontsize']+14)
            plt.show()
        
        
        
        # ##########################
        # Time series 
        ############################    
        # pdb.set_trace()
        if Qlunc_yaml_inputs['Atmospheric_inputs']['TimeSeries']:
            Vh_p  = [np.concatenate( Data['Vh']['V1_GUM'][0], axis=0 ) for i in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction GUM']))]

            fig9,ax9 = plt.subplots(4,1,sharex=True)
            ax9[0].plot(Atmospheric_Scenario.date,Atmospheric_Scenario.PL_exp[0],'.',label=r'$\alpha$')
            ax9[1].plot(Atmospheric_Scenario.date,Atmospheric_Scenario.wind_direction,'.',linewidth=2.64)
            # ax9[1].plot(Atmospheric_Scenario.date,Data['Wind direction']['V1_GUM'][0],'-',color='gold',label='lidar',linewidth=.61)
            # ax9[1].fill_between(Atmospheric_Scenario.date,Data['Wind direction']['V1_GUM'][0]-2*UWindDir_GUM[0],Data['Wind direction']['V1_GUM'][0]+2*UWindDir_GUM[0],color='red', alpha=0.35,label='uncertainty')    

            # ax9[1].legend(loc=1, fontsize=16.23)
            ax9[2].plot(Atmospheric_Scenario.date,UVh_GUM[0],'.',label='Vh Uncertainty GUM')
            ax9[2].plot(Atmospheric_Scenario.date,UVh_MCM[0],'.',label='Vh Uncertainty MCM')
            ax9[2].legend()
            ax9[3].plot(Atmospheric_Scenario.date,Atmospheric_Scenario.Vref,'dimgrey',linewidth=2.64,label='cup')
            ax9[3].plot(Atmospheric_Scenario.date,Data['Vh']['V1_GUM'][0],'gold',linewidth=.8,label='lidar')
            
            ax9[3].fill_between(Atmospheric_Scenario.date,[Vh_p-2*UVh_GUM[0]][0][0],[Vh_p+2*UVh_GUM[0]][0][0],color='red', alpha=0.35,label='uncertainty')    
            # pdb.set_trace()
            ax9[3].legend(loc=4, fontsize=16.23)
            
            
            ### Axes ###############################
            ax9[0].grid(axis = 'both')
            ax9[0].set_xlim(Atmospheric_Scenario.date[0],Atmospheric_Scenario.date[-1])            
            ax9[0].tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
            
            ax9[1].grid(axis = 'both')
            ax9[1].tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
            
            ax9[2].grid(axis = 'both')
            ax9[2].tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
            
            ax9[3].grid(axis = 'both')
            ax9[3].tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])

            ax9[3].set_xlabel('Date [mm-dd hh]',fontsize=30)

            ax9[0].set_ylabel('[-]',fontsize=30)
            ax9[1].set_ylabel(r'$\Omega$[°]',fontsize=30)
            ax9[2].set_ylabel(r'[ms$^{-1}$]',fontsize=30)
            ax9[2].set_ylabel('[m/s]',fontsize=30)
            plt.subplots_adjust(left=0.075, right=0.995, bottom=0.16, top=0.975, wspace=0.3, hspace=0.15)         
            plt.xticks(rotation=20)

                    ############################################################           
            ## Plots the 3D figure
            # fig = plt.figure()
            # ax = Axes3D(fig)
                            
            # plt.scatter(Data['lidars']['Coord_Out'][0],Data['lidars']['Coord_Out'][1], s=20,c=scalarMap.to_rgba(V))
            # # ax.scatter(Data['lidars']['Coord_Out'][0],Data['lidars']['Coord_Out'][1], Data['lidars']['Coord_Out'][2], V, c=scalarMap.to_rgba(V))
            # ax00.plot(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0][0],Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0][1],'sk', ms=5, mec='black', mew=1.5)
            # ax00.plot(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1][0],Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1][1],'sk', ms=5, mec='white', mew=1.5)
            # ax00.set_xlabel('Y [m]', fontsize=plot_param['tick_labelfontsize']+20, labelpad=15)
            # ax00.set_ylabel('Z [m]', fontsize=plot_param['tick_labelfontsize']+20, labelpad=15)
            # ax.set_xlabel('X [m]', fontsize=plot_param['tick_labelfontsize'], labelpad=15)
            # ax.set_ylabel('Y [m]', fontsize=plot_param['tick_labelfontsize'], labelpad=15)
            # ax.set_zlabel('Z [m]', fontsize=plot_param['tick_labelfontsize'], labelpad=15)
            
            # ax.plot(Data['lidars']['Lidar0_Rectangular']['LidarPosX'],Data['lidars']['Lidar0_Rectangular']['LidarPosY'],Data['lidars']['Lidar0_Rectangular']['LidarPosZ'],'sb')
            # ax.plot(Data['lidars']['Lidar1_Rectangular']['LidarPosX'],Data['lidars']['Lidar1_Rectangular']['LidarPosY'],Data['lidars']['Lidar1_Rectangular']['LidarPosZ'],'sb')
            # ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
            # ax.xaxis.set_tick_params(labelsize=plot_param['tick_labelfontsize']-3)
            # ax.yaxis.set_tick_params(labelsize=plot_param['tick_labelfontsize']-3)
            # ax.zaxis.set_tick_params(labelsize=plot_param['tick_labelfontsize']-3)
            # ax00.set_box_aspect([ub - lb for lb, ub in (getattr(ax00, f'get_{a}lim')() for a in 'yz')])
           # lab_Xaxis=ax00.get_xticks()
           # lab_Yaxis=ax00.get_yticks()
           # # ax00.set_xticks([lab_Xaxis[0],lab_Xaxis[2],lab_Xaxis[4],lab_Xaxis[6],lab_Xaxis[8]])
           # # ax00.set_yticks([lab_Yaxis[0],lab_Yaxis[2],lab_Yaxis[4],lab_Yaxis[6]])

            #############################################################                   
            

        
    #######################################################
    # 2. Plot Uncertainty in Vlos with theta       
    
    if flag_plot_LOS_unc:
        fig,ax2 = plt.subplots() 
        color   = iter(cm.rainbow(np.linspace(0, 1, len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))
        
        for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
            cc=next(color)          
            ax2.plot(np.degrees(Data['lidars']['Coord_Test']['TESTt'][0]),Data['VLOS Unc [m/s]']['VLOS Uncertainty GUM theta [m/s]'][ind_plot],c=cc,label=r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot]))
            ax2.plot(np.degrees(Data['lidars']['Coord_Test']['TESTt'][0]),Data['VLOS Unc [m/s]']['VLOS Uncertainty MC theta [m/s]'][ind_plot],'or' , markerfacecolor=cc,markeredgecolor='lime',alpha=0.3,label='MC')        
    
        ax2.legend(loc = 4, prop = {'size': plot_param['legend_fontsize']})
        ax2.tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
        ax2.set_xlim(0,90)
        ax2.set_ylim(0,0.04)
        ax2.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,0))
      
        # these are matplotlib.patch.Patch properties
        props   = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.4)
        textstr = '\n'.join((
        r'$\rho~ [m]=%.1f$' % (Data['lidars']['Lidar0_Spherical']['rho'], ),
        r'$\varphi~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['psi']), ),
        r'N ={:.1e}'.format(Decimal(Qlunc_yaml_inputs['Components']['Scanner']['N_MC']) )
        ))
        ax2.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
        plt.tight_layout()                    
        # place a tex1t box in upper left in axes coords
        ax2.text(0.5, 0.7, textstr, transform = ax2.transAxes, fontsize = 18, bbox = props)
        ax2.set_xlabel('Elevation angle [°]',fontsize = plot_param['axes_label_fontsize'])
        ax2.set_ylabel(r'$u_{V_{LOS}}$ [m/s]',fontsize = plot_param['axes_label_fontsize'])
        ax2.grid(axis = 'both')
        plt.show()
        
        
        
        # 3. Plot Uncertainty in Vlos with psi
        fig,ax3 = plt.subplots()
        color   = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))              
        for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
            cc = next(color)
            ax3.plot(np.degrees(Data['lidars']['Coord_Test']['TESTp'][0]),Data['VLOS Unc [m/s]']['VLOS Uncertainty GUM psi [m/s]'][ind_plot],c = cc,label = r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
            ax3.plot(np.degrees(Data['lidars']['Coord_Test']['TESTp'][0]),Data['VLOS Unc [m/s]']['VLOS Uncertainty MC psi [m/s]'][ind_plot],'or' , markerfacecolor = cc,markeredgecolor = 'lime',alpha = 0.4,label = 'MC')        
        ax3.legend(loc = 1, prop={'size': plot_param['legend_fontsize']})
        ax3.tick_params(axis = 'both', labelsize=plot_param['tick_labelfontsize'])
        ax3.set_xlim(0,359)
        ax3.set_ylim(0,0.04)
        ax3.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,0))
        # these are matplotlib.patch.Patch properties
        props3   = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.4)
        textstr3 = '\n'.join((
        r'$\rho ~[m]=%.1f$' % (Data['lidars']['Lidar0_Spherical']['rho'], ),
        r'$\theta~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['theta']), ),
        r'N ={:.1e}'.format(Decimal(Qlunc_yaml_inputs['Components']['Scanner']['N_MC']))))
        
        ax3.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
        plt.tight_layout()
        ax3.text(0.5,0.7, textstr3, transform = ax3.transAxes, fontsize = 18, bbox = props3)
        ax3.set_xlabel('Azimuth angle [°]',fontsize = plot_param['axes_label_fontsize'])
        ax3.set_ylabel(r'$u_{V_{LOS}}$ [m/s]',fontsize = plot_param['axes_label_fontsize'])
        ax3.grid(axis = 'both')
        plt.show()



        # 4.  Plot Uncertainty in Vrad with rho                   
        fig,ax4 = plt.subplots()
        color   = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))          
        for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
            cc = next(color)
            ax4.plot(Data['lidars']['Coord_Test']['TESTr'][0],Data['VLOS Unc [m/s]']['VLOS Uncertainty GUM rho [m/s]'][ind_plot],c = cc,label=r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
            ax4.plot(Data['lidars']['Coord_Test']['TESTr'][0],Data['VLOS Unc [m/s]']['VLOS Uncertainty MC rho [m/s]'][ind_plot],'or' , markerfacecolor = cc,markeredgecolor = 'lime',alpha = 0.4,label = 'MC')      
        ax4.legend(loc = 1, prop = {'size': plot_param['legend_fontsize']})
        ax4.tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
        ax4.set_xlim(0,5000)
        ax4.set_ylim(0,0.04) 
        ax4.ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))
        ax4.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])

        props4   = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.4)
        textstr4 = '\n'.join((
        r'$\theta~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['theta']), ),
        r'$\varphi~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['psi']), ),
         r'N ={:.1e}'.format(Decimal(Qlunc_yaml_inputs['Components']['Scanner']['N_MC']))))
    
        ax4.text(0.5,0.7, textstr4, transform = ax3.transAxes, fontsize = 18, bbox = props4)
        ax4.set_xlabel('Focus distance [m]',fontsize=plot_param['axes_label_fontsize'])
        ax4.set_ylabel(r'$u_{V_{LOS}}$ [m/s]',fontsize=plot_param['axes_label_fontsize'])
        ax4.grid(axis='both')
        ax4.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
        plt.tight_layout()
        plt.show() 
    
   
        # ##############################################
        # Plot  Vlos1, Vlos2 and Vlos3 uncertainties
        ################################################           
        # 5.  Plot Uncertainty in VLOS1 with wind direction 
        fig5,ax5 = plt.subplots(2,1)            
        fig5.tight_layout()
        color=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))   
        
        for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
            cc=next(color)
            ax5[0].plot(np.degrees(Data['wind direction']),Data['VLOS Unc [m/s]']['VLOS1 Uncertainty GUM [m/s]'][ind_plot],'-',c = cc,label = r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
            ax5[0].plot(np.degrees(Data['wind direction']),Data['VLOS Unc [m/s]']['VLOS1 Uncertainty MC [m/s]'][ind_plot],'o', markerfacecolor = cc,markeredgecolor = 'lime',alpha = 0.4,label = 'Montecarlo')                       
        
        
        # Plot with sensitivity coefficients:               
        
        Cont_Theta1         = (np.array(Data['Sens coeff Vlos']['V1_theta'][0]*np.array(np.radians(Data['STDVs'][0][0]))))**2
        Cont_Psi1           = (np.array(Data['Sens coeff Vlos']['V1_psi'][0]*np.array(np.radians(Data['STDVs'][1][0]))))**2
        Cont_Rho1           = (np.array(Data['Sens coeff Vlos']['V1_rho'][0]*np.array(Data['STDVs'][2][0])))**2     
        Cont_Corr1          = 2*Lidar.optics.scanner.correlations[9]*np.array(Data['Sens coeff Vlos']['V1_theta'][0])*np.array(Data['Sens coeff Vlos']['V1_psi'][0])*np.array(np.radians(Data['STDVs'][0][0]))*np.array(np.radians(Data['STDVs'][1][0]))
        # pdb.set_trace()
        ax5[1].plot(np.degrees(Data['wind direction']),Cont_Theta1,'-',c = 'black',    linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS},\theta_1}$')
        ax5[1].plot(np.degrees(Data['wind direction']),Cont_Psi1 ,'-', c = 'dimgray',  linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS},\varphi_1}$')
        ax5[1].plot(np.degrees(Data['wind direction']),Cont_Rho1,'-',  c = 'darkgrey',linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS},\rho_1}$')
        ax5[1].plot(np.degrees(Data['wind direction']),Cont_Corr1 ,'-',c = 'cadetblue',linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS},\theta_1,\varphi_1}$')
        ax5[1].set_xlabel('Wind Direction [°]',fontsize = plot_param['axes_label_fontsize'])
        ax5[0].set_ylabel('$u_{V_{LOS_1}}$ [m/s]',fontsize = plot_param['axes_label_fontsize'])
        ax5[1].set_ylabel(r'$T_{V_{LOS}}$ [m/s]',fontsize = plot_param['axes_label_fontsize'])
        ax5[0].set_xlim(0,359)
        ax5[1].set_xlim(0,359)            
        
        
        ax5[0].grid(axis = 'both') 
        ax5[1].grid(axis = 'both') 
        ax5[0].legend(loc = 1, prop = {'size': plot_param['legend_fontsize']})
        ax5[1].legend(loc = 1, prop = {'size': 5+plot_param['legend_fontsize']})  
        ax5[0].tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
        ax5[1].tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
     
        props5   = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.4)
        textstr5 = '\n'.join((
        r'$\rho ~[m]=%.1f$' % (Data['lidars']['Lidar0_Spherical']['rho'] ),
        r'$\theta~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['theta'])),
        r'$\varphi~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['psi'])),
        r'$r_{\theta,\varphi}~ =%.1f$' % (Lidar.optics.scanner.correlations[9]),
        r'N ={:.1e}'.format(Decimal(Qlunc_yaml_inputs['Components']['Scanner']['N_MC'])),           
        ))           
        ax5[0].text(0.5, 0.95, textstr5, transform = ax5[0].transAxes, fontsize = 14,horizontalalignment = 'left',verticalalignment = 'top', bbox = props5)
        ax5[0].ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))
        ax5[1].ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))
        ax5[0].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
        ax5[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
 


       # 6.  Plot Uncertainty in VLOS2 with wind direction 
        fig6,ax6 = plt.subplots(2,1)  
        fig6.tight_layout()
        color = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))   
        for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
            cc = next(color)
            ax6[0].plot(np.degrees(Data['wind direction']),Data['VLOS Unc [m/s]']['VLOS2 Uncertainty GUM [m/s]'][ind_plot],'-',c = cc,label = r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
            ax6[0].plot(np.degrees(Data['wind direction']),Data['VLOS Unc [m/s]']['VLOS2 Uncertainty MC [m/s]'][ind_plot],'o', markerfacecolor = cc,markeredgecolor = 'lime',alpha = 0.4,label = 'Montecarlo')
                    
        # Plot with sensitivity coefficients:           
        Cont_Theta2         = (np.array(Data['Sens coeff Vlos']['V2_theta'][0]*np.array(np.radians(Data['STDVs'][0][1]))))**2
        Cont_Psi2           = (np.array(Data['Sens coeff Vlos']['V2_psi'][0]*np.array(np.radians(Data['STDVs'][1][1]))))**2
        Cont_Rho2           = (np.array(Data['Sens coeff Vlos']['V2_rho'][0]*np.array(Data['STDVs'][2][1])))**2     
        Cont_Corr2          = 2*Lidar.optics.scanner.correlations[10]*np.array(Data['Sens coeff Vlos']['V2_theta'][0])*np.array(Data['Sens coeff Vlos']['V2_psi'][0])*np.array(np.radians(Data['STDVs'][1][1]))*np.array(np.radians(Data['STDVs'][0][1]))

        # Plotting contributors:
        ax6[1].plot(np.degrees(Data['wind direction']),Cont_Theta2,'-',c = 'black',    linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS_2},\theta_2}$')
        ax6[1].plot(np.degrees(Data['wind direction']),Cont_Psi2  ,'-',c = 'dimgray',  linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS_2},\varphi_2}$')
        ax6[1].plot(np.degrees(Data['wind direction']),Cont_Rho2  ,'-',c = 'darkgrey',linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS_2},\rho_2}$')
        ax6[1].plot(np.degrees(Data['wind direction']),Cont_Corr2 ,'-',c = 'cadetblue',linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS_2},\theta_2,\varphi_2}$')

        ax6[1].set_xlabel('Wind Direction [°]',fontsize = plot_param['axes_label_fontsize'])
        ax6[0].set_ylabel('$u_{V_{LOS_2}}$ [m/s]',fontsize = plot_param['axes_label_fontsize'])
        ax6[1].set_ylabel(r'$T_{V_{LOS}}$ [m/s]',fontsize = plot_param['axes_label_fontsize'])
        ax6[0].set_xlim(0,359)
        ax6[1].set_xlim(0,359)
        ax6[0].grid(axis = 'both') 
        ax6[1].grid(axis = 'both') 
        ax6[0].legend(loc = 1, prop = {'size': plot_param['legend_fontsize']})
        ax6[1].legend(loc = 1, prop = {'size': 5+plot_param['legend_fontsize']})  
        ax6[0].tick_params(axis = 'both', labelsize=plot_param['tick_labelfontsize'])
        ax6[1].tick_params(axis = 'both', labelsize=plot_param['tick_labelfontsize'])

        
        props5 = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.4)
        textstr5 = '\n'.join((
        r'$\rho ~[m]=%.1f$' % (Data['lidars']['Lidar1_Spherical']['rho']  ),
        r'$\theta~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar1_Spherical']['theta'] ), ),
        r'$\varphi~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar1_Spherical']['psi'] ), ),
        r'$r_{\theta,\varphi}~ =%.1f$' % (Lidar.optics.scanner.correlations[10]),
        r'N ={:.1e}'.format(Decimal(Qlunc_yaml_inputs['Components']['Scanner']['N_MC']))))
        
        ax6[0].text(0.5, 0.95, textstr5, transform = ax6[0].transAxes, fontsize = 14,horizontalalignment = 'left',verticalalignment = 'top', bbox = props5)
        ax6[0].ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))
        ax6[1].ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))    
        
        ax6[0].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
        ax6[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
 
        # pdb.set_trace()
       
        if len(Lidar.optics.scanner.origin)==3: 
            # 7.  Plot Uncertainty in VLOS3 with wind direction 
            fig7,ax7 = plt.subplots(2,1)  
            fig7.tight_layout()
            color = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))   
            for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                cc = next(color)
                ax7[0].plot(np.degrees(Data['wind direction']),Data['VLOS Unc [m/s]']['VLOS3 Uncertainty GUM [m/s]'][ind_plot],'-',c = cc,label = r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
                ax7[0].plot(np.degrees(Data['wind direction']),Data['VLOS Unc [m/s]']['VLOS3 Uncertainty MC [m/s]'][ind_plot],'o', markerfacecolor = cc,markeredgecolor = 'lime',alpha = 0.4,label = 'Montecarlo')
                        
            # Plot with sensitivity coefficients:             
            Cont_Theta3         = (np.array(Data['Sens coeff Vlos']['V3_theta'][0]*np.array(np.radians(Data['STDVs'][0][2]))))**2
            Cont_Psi3           = (np.array(Data['Sens coeff Vlos']['V3_psi'][0]*np.array(np.radians(Data['STDVs'][1][2]))))**2
            Cont_Rho3           = (np.array(Data['Sens coeff Vlos']['V3_rho'][0]*np.array(Data['STDVs'][2][2])))**2     
            Cont_Corr3          = 2*Lidar.optics.scanner.correlations[11]*np.array(Data['Sens coeff Vlos']['V3_theta'][0])*np.array(Data['Sens coeff Vlos']['V3_psi'][0])*np.array(np.radians(Data['STDVs'][1][2]))*np.array(np.radians(Data['STDVs'][0][2]))

            # Plotting contributors:
            ax7[1].plot(np.degrees(Data['wind direction']),Cont_Theta3,'-',c = 'black',    linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS_3},\theta_3}$')
            ax7[1].plot(np.degrees(Data['wind direction']),Cont_Psi3  ,'-',c = 'dimgray',  linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS_3},\varphi_3}$')
            ax7[1].plot(np.degrees(Data['wind direction']),Cont_Rho3  ,'-',c = 'darkgrey',linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS_3},\rho_3}$')
            ax7[1].plot(np.degrees(Data['wind direction']),Cont_Corr3 ,'-',c = 'cadetblue',linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS_3},\theta_3,\varphi_3}$')

            ax7[1].set_xlabel('Wind Direction [°]',fontsize = plot_param['axes_label_fontsize'])
            ax7[0].set_ylabel('$u_{V_{LOS_3}}$ [m/s]',fontsize = plot_param['axes_label_fontsize'])
            ax7[1].set_ylabel(r'$T_{V_{LOS}}$ [m/s]',fontsize = plot_param['axes_label_fontsize']+.5)
            ax7[0].set_xlim(0,359)
            ax7[1].set_xlim(0,359)
            ax7[0].grid(axis = 'both') 
            ax7[1].grid(axis = 'both') 
            ax7[0].legend(loc = 1, prop = {'size': plot_param['legend_fontsize']})
            ax7[1].legend(loc = 1, prop = {'size': 5+plot_param['legend_fontsize']})  
            ax7[0].tick_params(axis = 'both', labelsize=plot_param['tick_labelfontsize'])
            ax7[1].tick_params(axis = 'both', labelsize=plot_param['tick_labelfontsize'])

            
            props5 = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.4)
            textstr6 = '\n'.join((
            r'$\rho ~[m]=%.1f$' % (Data['lidars']['Lidar2_Spherical']['rho']  ),
            r'$\theta~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar2_Spherical']['theta'] ), ),
            r'$\varphi~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar2_Spherical']['psi'] ), ),
            r'$r_{\theta,\varphi}~ =%.1f$' % (Lidar.optics.scanner.correlations[10]),
            r'N ={:.1e}'.format(Decimal(Qlunc_yaml_inputs['Components']['Scanner']['N_MC']), ),           
            ))           
            ax7[0].text(0.5, 0.95, textstr6, transform = ax7[0].transAxes, fontsize = 14,horizontalalignment = 'left',verticalalignment = 'top', bbox = props5)
            ax7[0].ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))
            ax7[1].ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))            
            ax7[0].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
            ax7[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])

            # ##############################################
            # Plot  Vlos cross-correlation terms
            ################################################     
            fig8,ax8 = plt.subplots(3,1)  
            fig8.tight_layout()
            color = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))   
            markers_plot = 6

            # Plot with sensitivity coefficients:                                             
            # Vlos1Vlos2
            Corr_psi1psi2     = 2*Lidar.optics.scanner.correlations[0]*np.array(Data['Sens coeff Vlos']['V1_psi'][0])*np.array(Data['Sens coeff Vlos']['V2_psi'][0])*np.array(np.radians(Data['STDVs'][1][0]))*np.array(np.radians(Data['STDVs'][1][1]))
            Corr_theta1theta2 = 2*Lidar.optics.scanner.correlations[3]*np.array(Data['Sens coeff Vlos']['V1_theta'][0])*np.array(Data['Sens coeff Vlos']['V2_theta'][0])*np.array(np.radians(Data['STDVs'][0][0]))*np.array(np.radians(Data['STDVs'][0][1]))
            Corr_rho1rho2     = 2*Lidar.optics.scanner.correlations[6]*np.array(Data['Sens coeff Vlos']['V1_rho'][0])*np.array(Data['Sens coeff Vlos']['V2_rho'][0])*np.array(Data['STDVs'][2][0])*np.array(Data['STDVs'][2][1])
            Corr_psi1theta2   = 2*Lidar.optics.scanner.correlations[12]*np.array(Data['Sens coeff Vlos']['V1_psi'][0])*np.array(Data['Sens coeff Vlos']['V2_theta'][0])*np.array(np.radians(Data['STDVs'][1][0]))*np.array(np.radians(Data['STDVs'][0][1]))
            Corr_psi2theta1   = 2*Lidar.optics.scanner.correlations[14]*np.array(Data['Sens coeff Vlos']['V2_psi'][0])*np.array(Data['Sens coeff Vlos']['V1_theta'][0])*np.array(np.radians(Data['STDVs'][1][1]))*np.array(np.radians(Data['STDVs'][0][0]))
            
            # Vlos1Vlos3
            Corr_psi1psi3     = 2*Lidar.optics.scanner.correlations[1]*np.array(Data['Sens coeff Vlos']['V1_psi'][0])*np.array(Data['Sens coeff Vlos']['V3_psi'][0])*np.array(np.radians(Data['STDVs'][1][0]))*np.array(np.radians(Data['STDVs'][1][2]))
            Corr_theta1theta3 = 2*Lidar.optics.scanner.correlations[4]*np.array(Data['Sens coeff Vlos']['V1_theta'][0])*np.array(Data['Sens coeff Vlos']['V3_theta'][0])*np.array(np.radians(Data['STDVs'][0][0]))*np.array(np.radians(Data['STDVs'][0][2]))
            Corr_rho1rho3     = 2*Lidar.optics.scanner.correlations[7]*np.array(Data['Sens coeff Vlos']['V1_rho'][0])*np.array(Data['Sens coeff Vlos']['V3_rho'][0])*np.array(Data['STDVs'][2][0])*np.array(Data['STDVs'][2][2])
            Corr_psi1theta3   = 2*Lidar.optics.scanner.correlations[13]*np.array(Data['Sens coeff Vlos']['V1_psi'][0])*np.array(Data['Sens coeff Vlos']['V3_theta'][0])*np.array(np.radians(Data['STDVs'][1][0]))*np.array(np.radians(Data['STDVs'][0][2]))
            Corr_psi3theta1   = 2*Lidar.optics.scanner.correlations[16]*np.array(Data['Sens coeff Vlos']['V3_psi'][0])*np.array(Data['Sens coeff Vlos']['V1_theta'][0])*np.array(np.radians(Data['STDVs'][1][2]))*np.array(np.radians(Data['STDVs'][0][0]))

            # Vlos1Vlos3
            Corr_psi2psi3     = 2*Lidar.optics.scanner.correlations[2]*np.array(Data['Sens coeff Vlos']['V2_psi'][0])*np.array(Data['Sens coeff Vlos']['V3_psi'][0])*np.array(np.radians(Data['STDVs'][1][1]))*np.array(np.radians(Data['STDVs'][1][2]))
            Corr_theta2theta3 = 2*Lidar.optics.scanner.correlations[5]*np.array(Data['Sens coeff Vlos']['V2_theta'][0])*np.array(Data['Sens coeff Vlos']['V3_theta'][0])*np.array(np.radians(Data['STDVs'][0][1]))*np.array(np.radians(Data['STDVs'][0][2]))
            Corr_rho2rho3     = 2*Lidar.optics.scanner.correlations[8]*np.array(Data['Sens coeff Vlos']['V2_rho'][0])*np.array(Data['Sens coeff Vlos']['V3_rho'][0])*np.array(Data['STDVs'][2][1])*np.array(Data['STDVs'][2][2])
            Corr_psi2theta3   = 2*Lidar.optics.scanner.correlations[15]*np.array(Data['Sens coeff Vlos']['V2_psi'][0])*np.array(Data['Sens coeff Vlos']['V3_theta'][0])*np.array(np.radians(Data['STDVs'][1][1]))*np.array(np.radians(Data['STDVs'][0][2]))
            Corr_psi3theta2   = 2*Lidar.optics.scanner.correlations[17]*np.array(Data['Sens coeff Vlos']['V3_psi'][0])*np.array(Data['Sens coeff Vlos']['V2_theta'][0])*np.array(np.radians(Data['STDVs'][1][2]))*np.array(np.radians(Data['STDVs'][0][1]))
            # pdb.set_trace()
            
            # Plotting contributors:
            ax8[0].plot(np.degrees(Data['wind direction']),Corr_psi1psi2,'-d',markersize=8,c = 'black', markevery=markers_plot,   linewidth = 3,label = r'$T_{V_{LOS},\varphi_{ij}}$')
            ax8[0].plot(np.degrees(Data['wind direction']),Corr_theta1theta2  ,'-s',markersize=8,c = 'dimgray', markevery=markers_plot, linewidth = 3,label = r'$T_{V_{LOS},\theta_{ij}}$')
            ax8[0].plot(np.degrees(Data['wind direction']),Corr_rho1rho2  ,'-^',markersize=8,c = 'darkgrey',markevery=markers_plot,linewidth = 3,label = r'$T_{V_{LOS},\rho_{ij}}$')
            ax8[0].plot(np.degrees(Data['wind direction']),Corr_psi1theta2 ,'-X',markersize=8,c = 'cadetblue',markevery=markers_plot,linewidth = 3,label = r'$T_{V_{LOS},\varphi_i\theta_j}$')
            ax8[0].plot(np.degrees(Data['wind direction']),Corr_psi2theta1 ,'-o',markersize=8,c = 'gold',markevery=markers_plot,linewidth = 3,label = r'$T_{V_{LOS},\theta_i\varphi_j}$')
            ax8[0].tick_params(axis='x',label1On=False)

            ax8[1].plot(np.degrees(Data['wind direction']),Corr_psi1psi3,'-d',markersize=8,c = 'black', markevery=markers_plot,   linewidth = 3,label = r'$T_{V_{LOS},\varphi_{ij}}$')
            ax8[1].plot(np.degrees(Data['wind direction']),Corr_theta1theta3  ,'-s',markersize=8,c = 'dimgray',markevery=markers_plot,  linewidth = 3,label = r'$T_{V_{LOS},\theta_{ij}}$')
            ax8[1].plot(np.degrees(Data['wind direction']),Corr_rho1rho3  ,'-^',markersize=8,c = 'darkgrey',markevery=markers_plot,linewidth = 3,label = r'$T_{V_{LOS},\rho_{ij}}$')
            ax8[1].plot(np.degrees(Data['wind direction']),Corr_psi1theta3 ,'-X',markersize=8,c = 'cadetblue',markevery=markers_plot,linewidth = 3,label = r'$T_{V_{LOS},\varphi_i\theta_j}$')
            ax8[1].plot(np.degrees(Data['wind direction']),Corr_psi3theta1 ,'-o',markersize=8,c = 'gold',markevery=markers_plot,linewidth = 3,label = r'$T_{V_{LOS},\theta_i\varphi_j}$')
            ax8[1].tick_params(axis='x',label1On=False)

            ax8[2].plot(np.degrees(Data['wind direction']),Corr_psi2psi3,'-d',markersize=8,c = 'black',  markevery=markers_plot,  linewidth = 3,label = r'$T_{V_{LOS},\varphi_{ij}}$')
            ax8[2].plot(np.degrees(Data['wind direction']),Corr_theta2theta3  ,'-s',markersize=8,c = 'dimgray',markevery=markers_plot,  linewidth = 3,label =  r'$T_{V_{LOS},\theta_{ij}}$')
            ax8[2].plot(np.degrees(Data['wind direction']),Corr_rho2rho3  ,'-^',markersize=8,c = 'darkgrey',markevery=markers_plot,linewidth = 3,label = r'$T_{V_{LOS},\rho_{ij}}$')
            ax8[2].plot(np.degrees(Data['wind direction']),Corr_psi2theta3 ,'-X',markersize=8,c = 'cadetblue',markevery=markers_plot,linewidth = 3,label = r'$T_{V_{LOS},\varphi_i\theta_j}$')
            ax8[2].plot(np.degrees(Data['wind direction']),Corr_psi3theta2 ,'-o',markersize=8,c = 'gold',markevery=markers_plot,linewidth = 3,label = r'$T_{V_{LOS},\theta_i\varphi_j}$')    

            ax8[2].set_xlabel('Wind Direction [°]',fontsize = plot_param['axes_label_fontsize'])
            ax8[0].set_ylabel(r'$T_{V_{LOS,12}}$ [m/s]',fontsize = 35)
            ax8[1].set_ylabel(r'$T_{V_{LOS,13}}$ [m/s]',fontsize = 35)
            ax8[2].set_ylabel(r'$T_{V_{LOS,23}}$ [m/s]',fontsize = 35)
            
            ax8[0].set_xlim(0,359)
            ax8[1].set_xlim(0,359)
            ax8[2].set_xlim(0,359)
            ax8[0].grid(axis = 'both') 
            ax8[1].grid(axis = 'both')
            ax8[2].grid(axis = 'both')
            
            ax8[0].legend(loc = 'center right', prop = {'size': plot_param['legend_fontsize']})
            # ax8[1].legend(loc = 1, prop = {'size': plot_param['legend_fontsize']})  
            # ax8[2].legend(loc = 1, prop = {'size': plot_param['legend_fontsize']})  
            
            ax8[0].tick_params(axis = 'both', labelsize=plot_param['tick_labelfontsize'])
            ax8[1].tick_params(axis = 'both', labelsize=plot_param['tick_labelfontsize'])
            ax8[2].tick_params(axis = 'both', labelsize=plot_param['tick_labelfontsize'])
            

            ax8[0].ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))
            ax8[1].ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))
            ax8[2].ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))            
            
            ax8[0].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])
            ax8[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])    
            ax8[2].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])    
            plt.subplots_adjust(right=0.995,left = 0.07,top = 0.975,bottom = 0.11,wspace = 0.3,hspace = 0.15)    
        
        else:
            pdb.set_trace()

            # ##############################################
            # Plot  Vlos cross-correlation terms
            ################################################
            fig8,ax8 = plt.subplots()  
            fig8.tight_layout()
            color = iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))   
            markers_plot =  0 + np.arange(0, 120)* 3

            # Vlos1Vlos2
            Corr_psi1psi2     = 2*Lidar.optics.scanner.correlations[0]*np.array(Data['Sens coeff Vlos']['V1_psi'][0])*np.array(Data['Sens coeff Vlos']['V2_psi'][0])*np.array(np.radians(Data['STDVs'][1][0]))*np.array(np.radians(Data['STDVs'][1][1]))
            Corr_theta1theta2 = 2*Lidar.optics.scanner.correlations[3]*np.array(Data['Sens coeff Vlos']['V1_theta'][0])*np.array(Data['Sens coeff Vlos']['V2_theta'][0])*np.array(np.radians(Data['STDVs'][0][0]))*np.array(np.radians(Data['STDVs'][0][1]))
            Corr_rho1rho2     = 2*Lidar.optics.scanner.correlations[6]*np.array(Data['Sens coeff Vlos']['V1_rho'][0])*np.array(Data['Sens coeff Vlos']['V2_rho'][0])*np.array(Data['STDVs'][2][0])*np.array(Data['STDVs'][2][1])
            Corr_psi1theta2   = 2*Lidar.optics.scanner.correlations[12]*np.array(Data['Sens coeff Vlos']['V1_psi'][0])*np.array(Data['Sens coeff Vlos']['V2_theta'][0])*np.array(np.radians(Data['STDVs'][1][0]))*np.array(np.radians(Data['STDVs'][0][1]))
            Corr_psi2theta1   = 2*Lidar.optics.scanner.correlations[14]*np.array(Data['Sens coeff Vlos']['V2_psi'][0])*np.array(Data['Sens coeff Vlos']['V1_theta'][0])*np.array(np.radians(Data['STDVs'][1][1]))*np.array(np.radians(Data['STDVs'][0][0]))

            # Plotting contributors:
            ax8.plot(np.degrees(Data['wind direction']),Corr_psi1psi2,'-d',c = 'black', markevery=markers_plot,   linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS},\varphi_{12}}$')
            ax8.plot(np.degrees(Data['wind direction']),Corr_theta1theta2  ,'-s',c = 'dimgray', markevery=markers_plot, linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS},\theta_{12}}$')
            ax8.plot(np.degrees(Data['wind direction']),Corr_rho1rho2  ,'-^',c = 'darkgrey',markevery=markers_plot,linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS},\rho_{12}}$')
            ax8.plot(np.degrees(Data['wind direction']),Corr_psi1theta2 ,'-X',c = 'cadetblue',markevery=markers_plot,linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS},\varphi_1\theta_2}$')
            ax8.plot(np.degrees(Data['wind direction']),Corr_psi2theta1 ,'-o',c = 'gold',markevery=markers_plot,linewidth = plot_param['linewidth'],label = r'$T_{V_{LOS},\varphi_2\theta_1}$')

            ax8.set_xlabel('Wind Direction [°]',fontsize = plot_param['axes_label_fontsize'])
            ax8.set_ylabel('$T_{V_{LOS}}$ [m/s]',fontsize = plot_param['axes_label_fontsize'])
            ax8.set_xlim(0,359)
            ax8.grid(axis = 'both')             
            ax8.legend(loc = 1, prop = {'size': plot_param['legend_fontsize']})
            ax8.tick_params(axis = 'both', labelsize=plot_param['tick_labelfontsize'])           
            ax8.ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))            
            ax8.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize_scy'])   
            plt.subplots_adjust(right=0.995,left = 0.075,top = 0.975,bottom = 0.11,wspace = 0.3,hspace = 0.24)

    # pdb.set_trace()
    ##################################################################################################
    ######################### Plot seaborn graphs ####################################################
    ##################################################################################################
    if flag_plot_PDFs:         
 
        ################################################### 
        # Plot velocities
        # Create the DataFrame
        from matplotlib.ticker import FormatStrFormatter
        import matplotlib.ticker as ticker
        
        def corrfunc_V(x, y, **kwds):
            cmap = ListedColormap(['white'])
            norm = kwds['norm']
            ax = plt.gca()
            # ax.tick_params(bottom=False, top=False, left=False, right=False)
            g_V.tick_params(axis='both',labelsize=20) 
            sns.despine(ax=ax, bottom=False, top=False, left=False, right=False)
            r, _ = pearsonr(x, y)
            facecolor = cmap(norm(r))
            ax.set_facecolor(facecolor)
            lightness = (max(facecolor[:3]) + min(facecolor[:3]) ) / 2
            ax.annotate(f"r={r:.2f}", xy=(.5, .5), xycoords=ax.transAxes,
                        color='white' if lightness < 0.7 else 'black', size=26, ha='center', va='center')        

        def corrfunc_Param(x, y, **kwds):
            cmap = ListedColormap(['white'])
            norm = kwds['norm']
            ax = plt.gca()
            # ax.tick_params(bottom=False, top=False, left=False, right=False)
            g_Param.tick_params(axis='both',labelsize=19) 
            sns.despine(ax=ax, bottom=False, top=False, left=False, right=False)
            r, _ = pearsonr(x, y)
            facecolor = cmap(norm(r))
            ax.set_facecolor(facecolor)
            lightness = (max(facecolor[:3]) + min(facecolor[:3]) ) / 2
            ax.annotate(f"r={r:.2f}", xy=(.5, .5), xycoords=ax.transAxes,
                        color='white' if lightness < 0.7 else 'black', size=26, ha='center', va='center')   
        
        def corrfunc_Unc(x, y, **kwds):
            cmap = ListedColormap(['white'])
            norm = kwds['norm']
            ax = plt.gca()
            # ax.tick_params(bottom=False, top=False, left=False, right=False)
            ax.tick_params(axis='both',labelsize=19) 
            sns.despine(ax=ax, bottom=False, top=False, left=False, right=False)
            r, _ = pearsonr(x, y)
            facecolor = cmap(norm(r))
            ax.set_facecolor(facecolor)
            lightness = (max(facecolor[:3]) + min(facecolor[:3]) ) / 2
            ax.annotate(f"r={r:.2f}", xy=(.5, .5), xycoords=ax.transAxes,
                        color='white' if lightness < 0.7 else 'black', size=26, ha='center', va='center')      


   
        
        
        def hide_ticks(*args, **kwds):
            plt.gca().tick_params(axis='both',bottom=False, top=False,right=False,left=False)        
        
        # Data frame velocities
        df_V     = pd.DataFrame(
                                 {r"$V_1$":Data['Mult param'][0][0],
                                  r"$V_2$":Data['Mult param'][1][0],
                                  r"$V_3$":Data['Mult param'][2][0]})   
        

                
        g_V = sns.PairGrid(df_V,aspect=1)
        g_V = g_V.map_diag(sns.distplot,fit=norm,kde=False)
        g_V = g_V.map_lower(sns.kdeplot,fill=True) 
        for ax in plt.gcf().axes:
            l = ax.get_xlabel()
            ax.set_xlabel(l, fontsize=40)    
            ll = ax.get_ylabel()
            ax.set_ylabel(ll, fontsize=40)     
            ax.xaxis.set_tick_params('both',labelsize=plot_param['tick_labelfontsize'])
            ax.yaxis.set_tick_params('both',labelsize=plot_param['tick_labelfontsize'])
        g_V.map_upper(corrfunc_V, cmap=ListedColormap(['white']), norm=plt.Normalize(vmin=-1, vmax=1))       
        g_V.map_upper(hide_ticks)        
        g_V.fig.subplots_adjust(top=0.985,bottom=0.086,right=.998,left=0.450,wspace=0.06, hspace=0.06) # equal spacing in both directions
        
        # DAtaframe angles
        df_Param = pd.DataFrame(
                                 {r"$\theta_1$":Data['Mult param'][12][0],
                                  r"$\theta_2$":Data['Mult param'][12][1],
                                  r"$\theta_3$":Data['Mult param'][12][2],
                                  r"$\varphi_1$":Data['Mult param'][13][0],
                                  r"$\varphi_2$":Data['Mult param'][13][1],
                                  r"$\varphi_3$":Data['Mult param'][13][2],})
                          # r"$\rho_1$":Mult_param[14][0],
                          # r"$\rho_2$":Mult_param[14][1],
                          # r"$\rho_3$":Mult_param[14][2]})   
        g_Param = sns.PairGrid(df_Param, aspect=1)
        g_Param = g_Param.map_diag(sns.distplot,fit=norm,kde=False)
        g_Param = g_Param.map_lower(sns.kdeplot,fill=True)
        
        
        for ax in plt.gcf().axes:
            ax.xaxis.set_tick_params(rotation=0)
            l = ax.get_xlabel()
            ax.set_xlabel(l, fontsize=30)    
            ll = ax.get_ylabel()
            ax.set_ylabel(ll, fontsize=30)     
            
            # ax.ticklabel_format(axis = 'x',style = 'sci', scilimits = (0,0))            
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))                
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))                
        
        
        
        g_Param.map_upper(corrfunc_Param, cmap=ListedColormap(['white']), norm=plt.Normalize(vmin=-1, vmax=1))
        g_Param.map_upper(hide_ticks)        
        g_Param.fig.subplots_adjust(top=0.99,bottom=0.430,right=.99,left=0.32,wspace=0.06, hspace=0.17) # equal spacing in both directions
        plt.show()

    
        
        # Plot uncertainty correlations
        for ind_a in range(len(Data['VLOS Unc [m/s]']['VLOS1 Uncertainty GUM [m/s]'])):
            # UV1,UV2,UV3=[],[],[]
            
            # Uncertainties data frame
            df_Unc=pd.DataFrame(
                                {"UV1":Data['VLOS Unc [m/s]']['VLOS1 Uncertainty GUM [m/s]'][ind_a],
                                 "UV2":Data['VLOS Unc [m/s]']['VLOS2 Uncertainty GUM [m/s]'][ind_a],
                                 "UV3":Data['VLOS Unc [m/s]']['VLOS3 Uncertainty GUM [m/s]'][ind_a]})
           
            
            g_Unc = sns.PairGrid(df_Unc, aspect=1)
            g_Unc = g_Unc.map_diag(sns.distplot)
            g_Unc = g_Unc.map_lower(plt.scatter)
            g_Unc = g_Unc.map_upper(corrfunc_Unc, cmap=ListedColormap(['white']), norm=plt.Normalize(vmin=-1, vmax=1))
        
        for ax in plt.gcf().axes:
            ax.xaxis.set_tick_params(rotation=0)
            l = ax.get_xlabel()
            ax.set_xlabel(l, fontsize=30)    
            ll = ax.get_ylabel()
            ax.set_ylabel(ll, fontsize=30)     
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))                
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))                
      
        g_Unc.map_upper(hide_ticks) 
   
        g_Unc.fig.subplots_adjust(top=0.99,bottom=0.430,right=.99,left=0.32,wspace=0.06, hspace=0.17) # equal spacing in both directions
        plt.show()
        


    
    #%% CI plotting
    if flag_plot_CIs:          
                   
         ###################################################################################
         ## VLOS ###########################################################################
         ###################################################################################
         
         
         # Vlos          = [l.tolist()[0] for l in Data['Vlos_GUM']['V{}'.format(Data['Tolerance'][-1])]]
         
         
         # CI_VLOS_L_GUM = [l.tolist()[0] for l in Data['CI'][0]]
         # CI_VLOS_H_GUM = [l.tolist()[0] for l in Data['CI'][1]]
        
         # CI_VLOS_L_MCM = [l.tolist()[0] for l in Data['CI'][2]]
         # CI_VLOS_H_MCM = [l.tolist()[0] for l in Data['CI'][3]]
        
        
         # # Limits of the CI for GUM and MCM
         # y1_GUM = [Vlos[inf0]-CI_VLOS_L_GUM[inf0] for inf0 in range(len(Vlos))]
         # y2_GUM = [Vlos[inf0]-CI_VLOS_H_GUM[inf0] for inf0 in range(len(Vlos))]
         # y1_MCM = [Vlos[inf0]-CI_VLOS_L_MCM[inf0] for inf0 in range(len(Vlos))]
         # y2_MCM = [Vlos[inf0]-CI_VLOS_H_MCM[inf0] for inf0 in range(len(Vlos))]    
        
         # #Percentage of MCM data within the calculated CI
         # percentage_VLOS=[]
         # for ind_per in range(len(Data['Mult param'][Data['Tolerance'][-1]])):
         #     percentage_VLOS.append( 100 * len([i for i in Data['Mult param'][Data['Tolerance'][-1]][ind_per] if i > CI_VLOS_L_GUM[ind_per] and i < CI_VLOS_H_GUM[ind_per]]) / len(Data['Mult param'][Data['Tolerance'][-1]][0]) )
         # CI_final = np.round(np.mean(percentage_VLOS),2)
        
        
         # #Plot results
         # fig, ax = plt.subplots()
         # ax.plot(np.degrees(Data['wind direction']), Data['Vlos_GUM']['V{}'.format(Data['Tolerance'][-1])], '-',color='goldenrod',zorder = 12,linewidth = 2.45,label = r'$V_{LOS}$ GUM')
         # # plt.plot(np.degrees(Data['wind direction']), Data['Mult param'][0], 'o',color = 'cornflowerblue',markersize = 3.7,markeredgecolor = 'royalblue',zorder=0)    
         # ax.plot(np.degrees(Data['wind direction']), np.array(Vlos) - np.array(y1_MCM),'--',color = 'dimgray',linewidth = 2.7,zorder = 12,label = 'CI MCM')  
         # ax.plot(np.degrees(Data['wind direction']), np.array(Vlos) - np.array(y2_MCM), '--',color = 'dimgray',linewidth = 2.7,zorder = 12)      
         # ax.fill_between(np.degrees(Data['wind direction']), np.array(Vlos)-np.array(y1_GUM), np.array(Vlos)-np.array(y2_GUM), alpha = 0.83,color = 'darkgrey',zorder = 11,label = 'CI - {}%'.format(CI_final))
         # ax.grid('both')
         # plt.legend(loc=3, prop = {'size': plot_param['legend_fontsize']})
        
         # ax.set_xlabel('Wind Direction [°]',fontsize = plot_param['axes_label_fontsize'])
         # ax.set_ylabel('[m/s]',fontsize = plot_param['axes_label_fontsize'])
         # ax.set_xlim(0,359)               
         # ax.tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])           

         # plt.subplots_adjust(right = 0.995,left = 0.07,top = 0.975,bottom = 0.085,wspace = 0.3,hspace = 0.24)            
         
         
         #####################################################################################
         ## V wind ###########################################################################    
         #####################################################################################
         # pdb.set_trace()
         Vh=[l.tolist()[0] for l in Data['Vh']['V{}_GUM'.format(Data['Tolerance'][-1])][0]]
         # GUM
         CI_Vh_L_GUM = [l.tolist()[0] for l in Data['CI'][4]]
         CI_Vh_H_GUM = [l.tolist()[0] for l in Data['CI'][5]]    
         y1_Vh_GUM   = [Vh[inf0]-CI_Vh_L_GUM[inf0] for inf0 in range(len(Vh))]
         y2_Vh_GUM   = [Vh[inf0]-CI_Vh_H_GUM[inf0] for inf0 in range(len(Vh))]
        
         # MCM
         CI_Vh_L_MCM = [l.tolist() for l in Data['CI'][6]]
         CI_Vh_H_MCM = [l.tolist() for l in Data['CI'][7]]
        
         y1_Vh_MCM = [Vh[inf0] - CI_Vh_L_MCM[inf0] for inf0 in range(len(Vh))]
         y2_Vh_MCM = [Vh[inf0] - CI_Vh_H_MCM[inf0] for inf0 in range(len(Vh))]    
        
         #Percentage of MCM data within the calculated CI
         percentage_Vh = []
         for ind_per in range(len(Data['Vh']['V{}_MCM'.format(Data['Tolerance'][-1])][0])):
             percentage_Vh.append( 100 * len([i for i in Data['Vh']['V{}_MCM'.format(Data['Tolerance'][-1])][0][ind_per] if i > CI_Vh_L_GUM[ind_per] and i < CI_Vh_H_GUM[ind_per]]) / len(Data['Vh']['V{}_MCM'.format(Data['Tolerance'][-1])][0][0]))
         CI_final_Vh = np.round(np.mean(percentage_Vh),2)
        

         # colors=['darkcyan','powderblue', 'silver','dimgrey','gold','darkorange','navy', 'cornflowerblue']
         #Plot results
         fig, ax = plt.subplots()
         ax.plot(np.degrees(Data['wind direction']), Data['Vh']['V{}_GUM'.format(Data['Tolerance'][-1])][0], '-',color = 'darkred',zorder = 2,linewidth = 3,label = r'$GUM - \overline{V}_{wind}$')
         # # ax.plot(np.degrees(Data['wind direction']), Data['Vh']['V1_MCM'][0], 'o',color = 'cornflowerblue',markersize = 3.7,markeredgecolor = 'royalblue',zorder=0)  
         ax.plot(np.degrees(Data['wind direction']), Data['Vh']['V{}_MCM_mean'.format(Data['Tolerance'][-1])][0],'o',color = 'lightcoral',markersize = 8,zorder = 1,label = 'MCM')
         plt.plot(np.degrees(Data['wind direction']), np.array(Vh) - np.array(y2_Vh_GUM),'-',color = 'darkorange',linewidth = 3,zorder = 2,label = 'GUM - CI = {}%'.format(CI_final_Vh))  
         plt.plot(np.degrees(Data['wind direction']), np.array(Vh) - np.array(y1_Vh_GUM),'-',color = 'darkorange',linewidth = 3,zorder = 2)  
         
         plt.plot(np.degrees(Data['wind direction']), np.array(Vh) - np.array(y1_Vh_MCM),'o',color = 'gold',markersize = 8,zorder = 1,label = 'MCM')  
         plt.plot(np.degrees(Data['wind direction']), np.array(Vh) - np.array(y2_Vh_MCM),'o',color = 'gold',markersize = 8,zorder = 1)  
         

         
         ax.grid('both')
         ax.set_xlabel('Wind Direction [°]',fontsize = plot_param['axes_label_fontsize'])
         ax.set_ylabel('CI [m/s]',fontsize = plot_param['axes_label_fontsize'])
         ax.set_xlim(0,359)         
         ax.tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])           
         # ax.ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))            
         # ax.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)   
         plt.subplots_adjust(right = 0.995,left = 0.090,top = 0.975,bottom = 0.11,wspace = 0.3,hspace = 0.24)            
        
         plt.legend(loc = 'center left', prop = {'size': plot_param['legend_fontsize']})
         
         
         # # save whole figure 
         # savepath="C:/SWE_LOCAL/Papers_FCG/Journals/Uncertainty_3D/Figures/CI/"
         # pickle.dump(fig, open(savepath+"CI_WindVelocity.pickle", "wb"))
        
         # # load figure from file
         # fig = pickle.load(open(savepath+"CI_WindVelocity.pickle", "rb"))
         
         
         #####################################################################################
         ## Wind Direction ###########################################################################    
         #####################################################################################
         WindDirection=Data['Wind direction']['V{}_GUM'.format(Data['Tolerance'][-1])][0]
         # GUM
         CI_WD_L_GUM = [l.tolist()[0] for l in Data['CI'][8]]
         CI_WD_H_GUM = [l.tolist()[0] for l in Data['CI'][9]]    
         y1_WD_GUM   = [WindDirection[inf0]-np.radians(CI_WD_L_GUM[inf0]) for inf0 in range(len(WindDirection))]
         y2_WD_GUM   = [WindDirection[inf0]-np.radians(CI_WD_H_GUM[inf0]) for inf0 in range(len(WindDirection))]
        
         # MCM
         CI_WD_L_MCM = [l.tolist() for l in Data['CI'][10]]
         CI_WD_H_MCM = [l.tolist() for l in Data['CI'][11]]
        
         y1_WD_MCM = [WindDirection[inf0] - np.radians(CI_WD_L_MCM[inf0]) for inf0 in range(len(WindDirection))]
         y2_WD_MCM = [WindDirection[inf0] - np.radians(CI_WD_H_MCM[inf0]) for inf0 in range(len(WindDirection))]    
        
         #Percentage of MCM data within the calculated CI
         percentage_WD = []
         for ind_per in range(len(Data['Wind direction']['V{}_MCM'.format(Data['Tolerance'][-1])][0])):
             percentage_WD.append( 100 * len([i for i in Data['Wind direction']['V{}_MCM'.format(Data['Tolerance'][-1])][0][ind_per] if i > CI_WD_L_GUM[ind_per] and i < CI_WD_H_GUM[ind_per]]) / len(Data['Wind direction']['V{}_MCM'.format(Data['Tolerance'][-1])][0][0]))
         CI_final_WD = np.round(np.mean(percentage_Vh),2)
        

         
         #Plot results
         #            colors=['darkcyan','powderblue', 'silver','dimgrey','gold','darkorange','navy', 'cornflowerblue']
         fig, ax = plt.subplots()
         ax.plot(np.degrees(Data['wind direction'][0:179]),( Data['Wind direction']['V{}_GUM'.format(Data['Tolerance'][-1])][0][0:179]), '-',color = 'darkred',zorder = 2,linewidth = 3,label = r'$GUM - \overline{\Omega}$')
         # # ax.plot(np.degrees(Data['wind direction']), Data['WindDirection']['V1_MCM'][0], 'o',color = 'cornflowerblue',markersize = 3.7,markeredgecolor = 'royalblue',zorder=0)  
         ax.plot(np.degrees(Data['wind direction'][0:179]), (Data['Wind direction']['V{}_MCM_mean'.format(Data['Tolerance'][-1])][0][0:179]),'o',color = 'lightcoral',markersize = 8,zorder = 1,label = 'MCM')
         plt.plot(np.degrees(Data['wind direction'][0:179]), np.degrees(np.array(WindDirection[0:179]) - np.array(y2_WD_GUM[0:179])),'-',color = 'darkcyan',linewidth = 3,zorder = 2,label = 'GUM - CI = {}%'.format(CI_final_WD))  
         plt.plot(np.degrees(Data['wind direction'][0:179]), np.degrees(np.array(WindDirection[0:179]) - np.array(y1_WD_GUM[0:179])),'-',color = 'darkcyan',linewidth = 3,zorder = 2)  
         
         plt.plot(np.degrees(Data['wind direction'][0:179]), np.degrees(np.array(WindDirection[0:179]) - np.array(y1_WD_MCM[0:179])),'o',color = 'powderblue',markersize = 8,zorder = 1,label = 'MCM')  
         plt.plot(np.degrees(Data['wind direction'][0:179]), np.degrees(np.array(WindDirection[0:179]) - np.array(y2_WD_MCM[0:179])),'o',color = 'powderblue',markersize = 8,zorder = 1)  
         

         ax.grid('both')
         ax.set_xlabel('Wind Direction [°]',fontsize = plot_param['axes_label_fontsize'])
         ax.set_ylabel('CI [°]',fontsize = plot_param['axes_label_fontsize'])
         ax.set_xlim(0,179) 
         ax.set_ylim(-150,320)
          
         ax.tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])           
         # ax.ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))            
         # ax.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)   
         plt.subplots_adjust(right = 0.995,left = 0.1,top = 0.975,bottom = 0.11,wspace = 0.3,hspace = 0.24)            
        
         plt.legend(loc =4, prop = {'size': plot_param['legend_fontsize']})

         
         # save whole figure 
         # savepath="C:/SWE_LOCAL/Papers_FCG/Journals/Uncertainty_3D/Figures/CI/"
         # pdb.set_trace()
         # pickle.dump(fig, open(savepath+"CI_WindDirectionZoom.pickle", "wb"))
        
         # # load figure from file
         # fig = pickle.load(open(savepath+"CI_WindDirection.pickle", "rb"))
                  


         
         
         
         
         ######################################################################################################    
         # plot MCM validation and tolerance ###################################################    
         ######################################################################################################    

         # Vh ######################################################################################################    
         
         fig,axtol = plt.subplots()
         axtol.axhline(y=Data['Tolerance'][0], color='red', linestyle='--',linewidth=4,label=r"$\delta$")
         axtol.plot(np.degrees(Data['wind direction']),Data['Tolerance'][2],'-',color = 'silver',markevery=3,linewidth=3,markersize=8,label=r"$d_{low}$")
         axtol.plot(np.degrees(Data['wind direction']),Data['Tolerance'][3],'-',color = 'dimgrey',markevery=3,linewidth=3,markersize=8,label=r"$d_{up}$")             

         axtol.grid('both')
         axtol.set_xlabel('Wind Direction [°]',fontsize = plot_param['axes_label_fontsize']+10)
         axtol.set_ylabel('$d_V$ [m/s]',fontsize = plot_param['axes_label_fontsize']+10)
         axtol.set_xlim(0,359)               
         axtol.tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize']+5)           
         axtol.ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))            
         axtol.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)   
         axtol.set_ylim(0,0.005425)
         plt.subplots_adjust(right = 0.995,left = 0.06,top = 0.965,bottom = 0.13,wspace = 0.3,hspace = 0.24)            
         
         plt.legend(loc = (0.77,0.55), prop = {'size': plot_param['legend_fontsize']+20})             
         

         # Wind dir ######################################################################################################   
         fig,axtol_dir = plt.subplots()
         axtol_dir.axhline(y=Data['Tolerance'][1], color='red', linestyle='--',linewidth=4,label=r"$\delta$")
         axtol_dir.plot(np.degrees(Data['wind direction']),Data['Tolerance'][4],'-',color = 'silver',markevery=3,linewidth=3,markersize=8,label=r"$d_{low}$")
         axtol_dir.plot(np.degrees(Data['wind direction']),Data['Tolerance'][5],'-',color = 'dimgrey',markevery=3,linewidth=3,markersize=8,label=r"$d_{up}$")
             
         axtol_dir.grid('both')
         axtol_dir.set_xlabel('Wind Direction [°]',fontsize = plot_param['axes_label_fontsize'])
         axtol_dir.set_ylabel(r'$d_\Omega$ [°]',fontsize = plot_param['axes_label_fontsize'])
         axtol_dir.set_xlim(0,359)    
         axtol_dir.set_ylim(0,0.05425)           
         axtol_dir.tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])           
         axtol_dir.ticklabel_format(axis = 'y',style = 'sci', scilimits = (0,0))            
         axtol_dir.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)   
         plt.subplots_adjust(right = 0.995,left = 0.05,top = 0.965,bottom = 0.11,wspace = 0.3,hspace = 0.24)            
        
         plt.legend(loc = (0.77,0.55), prop = {'size': plot_param['legend_fontsize']+20})             

    ###################################################    

    # # #Plot Vlos with stdv
    # fig, ax=plt.subplots()
    # Vel,stdvVel=[],[]
    # for ind in range (3):
    #     for ind_V in range(len(Data['Mult param'][0])):
    
    #         Vel.append(np.mean(Data['Mult param'][ind][ind_V]))       
    #         stdvVel.append(np.std(Data['Mult param'][ind][ind_V]))
    
    #     ax.plot(np.linspace(0,359,360),Vel,  'k-')
    #     ax.fill_between(np.linspace(0,359,360),np.array(Vel)-np.array(Data['Vh Unc [m/s]' ]['Uncertainty Vh MCM'][ind]),np.array(Vel)+np.array(Data['Vh Unc [m/s]' ]['Uncertainty Vh MCM'][ind]) ,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    #     plt.show()   
    #     Vel,stdvVel=[],[]
    

    # ## Plot Vh/Vwind with stdv

    # # plt.plot(np.linspace(0,359,360),Vh_MCM_['V1_MCM'][0],  'k-')
    # # plt.fill_between(np.linspace(0,359,360),np.array(Vh_MCM_['V1_MCM'][0])-np.array(U_Vh_MCM_T[0]),np.array(Vh_MCM_['V1_MCM'][0])+np.array(U_Vh_MCM_T[0]) ,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    # # plt.show()   

    
    
    ## Plot uncertainties
    # for ind_a in range(len(Data['VLOS Unc [m/s]']['VLOS1 Uncertainty GUM [m/s]'])):
    #     # UV1,UV2,UV3=[],[],[]
        
    #     df_results2=pd.DataFrame(
    #     {"UV1":Data['VLOS Unc [m/s]']['VLOS1 Uncertainty GUM [m/s]'][ind_a],
    #       "UV2":Data['VLOS Unc [m/s]']['VLOS2 Uncertainty GUM [m/s]'][ind_a],
    #       "UV3":Data['VLOS Unc [m/s]']['VLOS3 Uncertainty GUM [m/s]'][ind_a]})
    #     g = sns.PairGrid(df_results2,aspect=1,layout_pad=0.2)
        # g.map(plt.scatter)
        # sns.pairplot(df_results2,aspect=1)
        # xlabels,ylabels = [],[]
        # plt.title(r'$\alpha$={}'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_a] ))
        # g.map_diag(sns.distplot,kde=False)
        # g.map_lower(plt.scatter) 

        # for ax in g.axes[-1,:]:
        #     xlabel = ax.xaxis.get_label_text()
        #     xlabels.append(xlabel)
        # for ax in g.axes[:,0]:
        #     ylabel = ax.yaxis.get_label_text()
        #     ylabels.append(ylabel)
        
        # for i in range(len(xlabels)):
        #     for j in range(len(ylabels)):
        #         g.axes[j,i].xaxis.set_label_text(xlabels[i])
        #         g.axes[j,i].yaxis.set_label_text(ylabels[j])
        
        # for ax in plt.gcf().axes:
        #     l = ax.get_xlabel()
        #     ax.set_xlabel(l, fontsize=30)    
        #     ll = ax.get_ylabel()
        #     ax.set_ylabel(ll, fontsize=30)     
        #     ax.tick_params(axis='both', labelsize=20)
        #     g_Param.tick_params(axis='both',labelsize=15)   
        # plt.show()


   
    # # Ellipse:
    # plt.plot(UV1,UV3)
    # x=(np.std(UV3)/np.std(UV1))*(np.sqrt(abs(np.array([np.std(UV1)])-np.array(UV1)**2)))
    # plt.plot(x,UV3)
    
    
    # # Plot error ellipse 
    # fig0,ax0=plt.subplots()
    # x = Data['VLOS Unc [m/s]']['VLOS1 Uncertainty GUM [m/s]'][2] #Data['Mult param'][0][0]
    # y = Data['VLOS Unc [m/s]']['VLOS2 Uncertainty GUM [m/s]'][2] #Data['Mult param'][1][0]
    # cov = np.cov(x, y)
    # lambda_, v = np.linalg.eig(cov)
    # lambda_ = np.sqrt(lambda_)
    # from matplotlib.patches import Ellipse
    # ax = plt.subplot(111, aspect='equal')
    # plt.scatter(x, y)
    # for j in range( 4):
    #     ell = Ellipse(xy=(np.mean(x), np.mean(y)),
    #                   width=lambda_[0]*j*2, height=lambda_[1]*j*2,
    #                   angle=np.rad2deg(np.arctan2(*v[:,0][::-1])),alpha=0.7, color='xkcd:wine red',linewidth=2, fill=False, zorder=2)
    #     ell.set_facecolor('grey')
    #     ax.add_artist(ell)
    
    # plt.show()   
                    
    
    ##################################################
             

    #%%###############   Plot photodetector noise   #############################       
    if flag_plot_photodetector_noise:
        # Quantifying uncertainty from photodetector and interval domain for the plot Psax is define in the photodetector class properties)
        Psax = (Lidar.photonics.photodetector.Power_interval)
        # Plotting:
        fig,axs1 = plt.subplots()
        label0   = ['Shot','Thermal','Dark current','TIA','Total']
        i_label  = 0
        col      = ['darkturquoise','darkgoldenrod','slategray','navy','red']
        for i in Data['SNR_data_photodetector']:            
            axs1.plot(Psax,Data['SNR_data_photodetector'][i][0],color = col[i_label],label = label0[i_label], linewidth = 2.3)  
            i_label+= 1
        axs1.tick_params(axis = 'both', labelsize = plot_param['tick_labelfontsize'])
            
        # axs1.plot(Psax,Data['Total_SNR_data'],label='Total SNR')
        axs1.set_xlabel('Input Signal optical power [W]',fontsize = plot_param['axes_label_fontsize'])
        axs1.set_ylabel('SNR [dB]',fontsize = plot_param['axes_label_fontsize'])
        axs1.legend(fontsize = plot_param['legend_fontsize'],loc='upper right')
        # axs1.set_title('SNR - Photodetector',fontsize=plot_param['title_fontsize'])
        axs1.grid(axis = 'both')
        axs1.text(.90,.05,plot_param['Qlunc_version'],transform = axs1.transAxes, fontsize = 14,verticalalignment = 'top',bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.5))
    

#%%################### PLOT COORDINATE SYSTEM DUAL LIDAR ##############################################

    # pdb.set_trace()

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
    
#     r = 89
#     x0 = 500 # To have the tangent at y=0
#     z0 = 119
    
#     # Theta varies only between pi/2 and 3pi/2. to have a half-circle
#     theta = np.linspace(0., 2*np.pi, 161)
    
#     x = np.zeros_like(theta)+x0 # x=0
#     y = r*np.cos(theta)  # y - y0 = r*cos(theta)
#     z = r*np.sin(theta) + z0 # z - z0 = r*sin(theta)
#     ax.plot(x, y, z,'k--',linewidth=2,label='Rotor area')
    
    
#     # # 2lidars
#     # # x1,y1,z1=[500,0],[0,-150],[119,1] 
#     # # x2,y2,z2=[500,0],[ 0,150],[119,1] 
    
    
#     # # 3 lidars
#     x1,y1,z1=[500,0],[0,-0],[119,1] 
#     x2,y2,z2=[500,728.46],[ 0,420.58],[119,1] 
#     x3,y3,z3=[500,728.46],[ 0,-420.58],[119,1] 
    
    
#     # p = Rectangle((Qlunc_yaml_inputs['Components']['Scanner']['Vertical plane parameters'][1], Qlunc_yaml_inputs['Components']['Scanner']['Vertical plane parameters'][3]), Qlunc_yaml_inputs['Components']['Scanner']['Vertical plane parameters'][2]*2,Qlunc_yaml_inputs['Components']['Scanner']['Vertical plane parameters'][1]*2,alpha=0.387,label='Scanned area')
#     # ax.add_patch(p)
#     # art3d.pathpatch_2d_to_3d(p, z=Qlunc_yaml_inputs['Components']['Scanner']['Vertical plane parameters'][0], zdir="x")
    
    
    
#     # x1,y1,z1=[500,0],[0,-150],[119,1] 
#     # x2,y2,z2=[500,0],[ 0,150],[119,1] 
#     # p = Wedge((0, 119), 89.15,0,359,alpha=0.5,label='WT area',width=1.71, ls='--')
#     # ax.add_patch(p)
#     # art3d.pathpatch_2d_to_3d(p, z=500, zdir="x")
#     ax.scatter(500,0, 119, c='r', s=50, marker='o', label=r'$P~(x,y,z)$')
#     ax.scatter(0, -0, 1, c='b', s=50, marker='s', label=r'$Lidars$')
#     ax.scatter(728.46, 420.58, 1, c='b', s=50, marker='s')
#     ax.scatter(728.46, -420.58, 1, c='b', s=50, marker='s')
#     ax.set_box_aspect((np.ptp(x1), np.ptp(x1), np.ptp(x1)))  # aspect ratio is 1:1:1 in data space
#     ax.set_box_aspect((np.ptp(x2), np.ptp(y2), np.ptp(z2)))  # aspect ratio is 1:1:1 in data space
#     ax.set_box_aspect((np.ptp(x3), np.ptp(y3), np.ptp(z3)))  # aspect ratio is 1:1:1 in data space
    
#     ax.plot(x1, y1, z1, color='magenta',linestyle='dashed')
#     ax.plot(x2, y2, z2, color='magenta',linestyle='dashed')
#     ax.plot(x3, y3, z3, color='magenta',linestyle='dashed')
    
    
#     ax.set_xlabel('X [m]', fontsize=21,labelpad=15)
#     ax.set_ylabel('Y [m]', fontsize=21,labelpad=15)
#     ax.set_zlabel('Z [m]', fontsize=21,labelpad=15)
#     ax.set_zlim([0,250])
#     ax.set_xlim([-20,850])
#     plt.legend(loc="best", fontsize=16.23)
    
#     ax.xaxis.set_tick_params(labelsize=15)
#     ax.yaxis.set_tick_params(labelsize=15)
#     ax.zaxis.set_tick_params(labelsize=15)
    


# ##################### Single system #################################################################################################
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
   

# r = 89
# x0 = 500 # To have the tangent at y=0
# z0 = 119

# # Theta varies only between pi/2 and 3pi/2. to have a half-circle
# theta = np.linspace(0., 2*np.pi, 161)

# x = np.zeros_like(theta)+x0 # x=0
# y = r*np.cos(theta)  # y - y0 = r*cos(theta)
# z = r*np.sin(theta) + z0 # z - z0 = r*sin(theta)
# ax.plot(x, y, z,'k--',linewidth=3,label='Rotor area')

# from Utils import Qlunc_Help_standAlone as SA
# x1,y1,z1=[500,0],[0,0],[119,1] 

# # x1,y1,z1=SA.cart2sph(x0,y0,z0)


# p = Wedge((0, 119), 89.15,0,360,alpha=0.9,label='Rotor diameter',width=3, ls='--')
# ax.add_patch(p)
# art3d.pathpatch_2d_to_3d(p, z=500, zdir="x")
# ax.scatter(500,0, 119, c='r', s=50, marker='o', label=r'$P~(x,y,z)$')
# ax.scatter(0, 0, 1, c='b', s=50, marker='s', label=r'$Lidar_1~and~Lidar_2$')

# ax.plot(x1, y1, z1, color='g',linestyle='dashed')

# ax.set_xlabel('X [m]', fontsize=40,labelpad=25)
# ax.set_ylabel('Y [m]', fontsize=40,labelpad=25)
# ax.set_zlabel('Z [m]', fontsize=40,labelpad=30)
# ax.set_zlim([0,250])
# ax.set_xlim([0,550])
# plt.legend(loc="best", fontsize=16.23)

# ax.xaxis.set_tick_params(labelsize=25,pad=5)
# ax.yaxis.set_tick_params(labelsize=25,pad=5)
# ax.zaxis.set_tick_params(labelsize=25,pad=15)





# #%% Print the three cases 
# fig = plt.figure(figsize=plt.figaspect(0.19))
# ax0 = fig.add_subplot(1,3,1,projection='3d')
# ax1 = fig.add_subplot(1,3,2,projection='3d')
# ax2 = fig.add_subplot(1,3,3,projection='3d')


# from Utils import Qlunc_Help_standAlone as SA
# # x1,y1,z1=[500,0],[0,0],[119,1] 
# x0,y0,z0=np.linspace(499.8,499.8,40),np.radians(np.linspace(0,90,40)),np.radians(np.linspace(0,0,40))
# x0,y0,z0=SA.sph2cart(x0,y0,z0)

# x1,y1,z1=np.linspace(499.8,499.8,40),np.radians(np.linspace(13.6,13.6,40)),np.radians(np.linspace(0,359,40))
# x1,y1,z1=SA.sph2cart(x1,y1,z1)

# x2,y2,z2=np.linspace(10,5000,30),np.radians(np.linspace(13.6,13.6,30)),np.radians(np.linspace(0,0,30))
# x2,y2,z2=SA.sph2cart(x2,y2,z2)




# # p = Wedge((0, 119), 89.15,0,360,alpha=0.9,label='Rotor diameter',width=3, ls='--')
# # ax.add_patch(p)
# # art3d.pathpatch_2d_to_3d(p, z=500, zdir="x")
# # ax.scatter(500,0, 119, c='r', s=50, marker='o', label=r'$P~(x,y,z)$')
# ax0.scatter(0, 0, 1, c='b', s=50, marker='s', label='Lidar')
# ax1.scatter(0, 0, 1, c='b', s=50, marker='s', label='Lidar')
# ax2.scatter(0, 0, 1, c='b', s=50, marker='s', label='Lidar')

# ax0.scatter(x0, y0 ,z0, color='r',linestyle='solid', label='scan')
# plt.legend(loc="lower center", fontsize=16.23)
# ax1.scatter(x1, y1, z1, color='r',linestyle='solid', label='scan')
# ax2.scatter(x2, y2, z2, color='r',linestyle='solid', label='scan')


# ax0.set_xlabel('X [m]', fontsize=30,labelpad=15)
# ax0.set_ylabel('Y [m]', fontsize=30,labelpad=25)
# ax0.set_zlabel('Z [m]', fontsize=30,labelpad=15)
# ax0.set_zlim([0,550])
# ax0.set_xlim([0,550])
# plt.legend(loc='lower center', fontsize=25)

# ax0.xaxis.set_tick_params(labelsize=20)
# ax0.yaxis.set_tick_params(labelsize=20)
# ax0.zaxis.set_tick_params(labelsize=20)

# ax1.set_xlabel('X [m]', fontsize=30,labelpad=15)
# ax1.set_ylabel('Y [m]', fontsize=30,labelpad=28)
# ax1.set_zlabel('Z [m]', fontsize=30,labelpad=15)
# ax1.set_zlim([0,150])
# ax1.set_xlim([-550,550])
# # plt.legend(loc="best", fontsize=16.23)

# ax1.xaxis.set_tick_params(labelsize=20)
# ax1.yaxis.set_tick_params(labelsize=20,pad=10)
# ax1.zaxis.set_tick_params(labelsize=20)


# ax2.set_xlabel('X [m]', fontsize=30,labelpad=30)
# ax2.set_ylabel('Y [m]', fontsize=30,labelpad=25)
# ax2.set_zlabel('Z [m]', fontsize=30,labelpad=25)
# ax2.set_zlim([0,1250])
# ax2.set_xlim([0,5500])
# # plt.legend(loc="best", fontsize=16.23)

# ax2.xaxis.set_tick_params(labelsize=20)
# ax2.yaxis.set_tick_params(labelsize=20)
# ax2.zaxis.set_tick_params(labelsize=20,pad=10)
