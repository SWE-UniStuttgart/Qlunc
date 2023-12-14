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
def plotting(Lidar,Qlunc_yaml_inputs,Data,flag_plot_measuring_points_pattern,flag_plot_photodetector_noise,flag_probe_volume_param,flag_plot_optical_amplifier_noise, flag_plot_pointing_unc,flag_plot_wind_dir_unc,flag_plot_correlations):
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
    # Ploting general parameters:
    plot_param={'axes_label_fontsize' : 25,
                'textbox_fontsize'    : 14,
                'title_fontsize'      : 29,
                'suptitle_fontsize'   : 23,
                'legend_fontsize'     : 15,
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
                'tick_labelfontsize'  : 21,

                'Qlunc_version'       : 'Qlunc Version - 1.0'
                }
        
    # pdb.set_trace()
    if flag_plot_measuring_points_pattern:
        if Lidar.optics.scanner.pattern in ['None']:
      
        
            # 0. Plot Uncertainty in /Omega against wind direction             
            color=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))
            fig0,ax0=plt.subplots(2,1)
            fig0.tight_layout()
            legt=[r'$\frac{\partial{\Omega}}{\partial{V_{LOS_1}}}$',r'$\frac{\partial{\Omega}}{\partial{V_{LOS_2}}}$',r'$\frac{\partial{\Omega}}{\partial{V_{LOS_{3}}}}$']

            for ind_plot in range(len(Data['WinDir Unc [°]']['Uncertainty wind direction MCM'])):
                
                cc=next(color)
                ax0[0].plot(np.degrees(Data['wind direction']),Data['WinDir Unc [°]']['Uncertainty wind direction GUM'][ind_plot],'-', color=cc,label=r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
                ax0[0].plot(np.degrees(Data['wind direction']),Data['WinDir Unc [°]']['Uncertainty wind direction MCM'][ind_plot],'o', markerfacecolor=cc,markeredgecolor='lime',alpha=0.4,label='MCM')        
            ax0[1].plot(np.degrees(Data['wind direction']),Data['Sens coeff Vlos']['W1'][-1],'-', color='black',alpha=0.7,label=legt[ind_plot])
            ax0[1].plot(np.degrees(Data['wind direction']),Data['Sens coeff Vlos']['W2'][-1],'-', color='dimgray',alpha=0.7)        
            ax0[1].plot(np.degrees(Data['wind direction']),Data['Sens coeff Vlos']['W1W2'][-1],'-', color='cadetblue',alpha=0.7)        
            
            

            	# Axes:
            ax0[0].set_ylabel('$\Omega$ Uncertainty [°]',fontsize=plot_param['axes_label_fontsize'])          
            ax0[0].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax0[0].tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax0[0].tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            ax0[0].set_xlim(0,359)
            ax0[0].set_ylim(0,4)
            ax0[0].grid(axis='both')

            ax0[1].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax0[1].set_ylabel('Sensitivity coefficients [°]',fontsize=plot_param['axes_label_fontsize']-2.3)
            ax0[1].ticklabel_format(axis='y',style='sci', scilimits=(0,0))          
            ax0[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-4)
            ax0[1].set_xlabel('Wind direction[°]',fontsize=plot_param['axes_label_fontsize'])
            ax0[1].tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax0[1].tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            ax0[1].set_xlim(0,359)
            ax0[1].grid(axis='both')

            plt.show()
            props0 = dict(boxstyle='round', facecolor='wheat', alpha=0.4)        
            textstr0 = '\n'.join((
            r'$r_{\theta_{1},\theta_{2}} ~=%.2f$' % ( Lidar.optics.scanner.correlations[1] ),
            r'$r_{\varphi_{1},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[0]),
            r'$r_{\rho_{1},\rho_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[2]),
            r'$r_{\theta_{1},\varphi_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[3]),
            r'$r_{\theta_{2},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[6])))           
            ax0[0].text(0.95, 0., textstr0, transform=ax0[0].transAxes, fontsize=16,horizontalalignment='left',verticalalignment='top', bbox=props0)     

            # pdb.set_trace()

            
            # 1. Plot Uncertainty in Vh against wind direction
            fig1=plt.figure()
            gs=fig1.add_gridspec(3,1,hspace=0.4,wspace=0.1)
            props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.4)        
            textstr1 = '\n'.join((
            r'$r_{\theta_{1},\theta_{2}} ~=%.2f$' % ( Lidar.optics.scanner.correlations[1] ),
            r'$r_{\varphi_{1},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[0] ),
            r'$r_{\rho_{1},\rho_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[2]),
            r'$r_{\theta_{1},\varphi_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[3]),
            r'$r_{\theta_{2},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[6]),
            r'$r_{\theta_{1},\varphi_{2}}~ =%.2f$' % (Lidar.optics.scanner.correlations[5]),
            r'$r_{\theta_{2},\varphi_{1}}~ =%.2f$' % (Lidar.optics.scanner.correlations[4])))                      
            # pdb.set_trace()
            ax11=fig1.add_subplot(gs[0])        
            color2=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))
            color3=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))               
                       
            
            # plot  VH or Vwind
            for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                c2=next(color2)
                # pdb.set_trace()
                plt.plot(np.degrees(Data['wind direction']),Data['Vh Unc [m/s]']['Uncertainty Vh GUM'][ind_plot],'-', color=c2,linewidth=plot_param['linewidth'],label=r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
                plt.plot(np.degrees(Data['wind direction']),Data['Vh Unc [m/s]']['Uncertainty Vh MCM'][ind_plot],'o' , markerfacecolor=c2,markeredgecolor='lime',alpha=0.4,label='MCM')
                ax11.set_ylim([.0, .5])
 
            # Plot correlations
            CorrelationsGUM=[Data['Correlations']['V12_GUM'],Data['Correlations']['V13_GUM'],Data['Correlations']['V23_GUM']]
            CorrelationsMCM=[Data['Correlations']['V12_MCM'],Data['Correlations']['V13_MCM'],Data['Correlations']['V23_MCM']]
            if len(Lidar.optics.scanner.origin)==3:
                gs_sub=gs[1].subgridspec(1, 3,wspace=0.05)
                ax0=fig1.add_subplot(gs_sub[0,0])
                ax1=fig1.add_subplot(gs_sub[0,1], sharey=ax0)
                ax2=fig1.add_subplot(gs_sub[0,2], sharey=ax0)
                ax11.set_ylabel('$U_{V_{wind}}$ [m/s]',fontsize=plot_param['axes_label_fontsize']+2) 
                for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                    c3=next(color3)
                    # Plot:
                    ax0.plot(np.degrees(Data['wind direction']),CorrelationsGUM[0][ind_plot],'-',c=c3,linewidth=plot_param['linewidth'])                      
                    ax0.plot(np.degrees(Data['wind direction']),CorrelationsMCM[0][ind_plot],'o', markerfacecolor=c3,markeredgecolor='lime',alpha=0.4)                    
                    ax1.plot(np.degrees(Data['wind direction']),CorrelationsGUM[1][ind_plot],'-',c=c3,linewidth=plot_param['linewidth'])                      
                    ax1.plot(np.degrees(Data['wind direction']),CorrelationsMCM[1][ind_plot],'o', markerfacecolor=c3,markeredgecolor='lime',alpha=0.4)
                    ax2.plot(np.degrees(Data['wind direction']),CorrelationsGUM[2][ind_plot],'-',c=c3,linewidth=plot_param['linewidth'])                      
                    ax2.plot(np.degrees(Data['wind direction']),CorrelationsMCM[2][ind_plot],'o', markerfacecolor=c3,markeredgecolor='lime',alpha=0.4)
                    
                    # Axes:
                    ax0.set_ylim(-1,1)
                    ax1.set_ylim(-1,1)
                    ax2.set_ylim(-1,1)                    
                    ax1.tick_params(labelleft=0)
                    ax2.tick_params(labelleft=0)                    
                    ax0.set_ylim(-1,1)
                    ax1.set_ylim(-1,1)
                    ax2.set_ylim(-1,1)
                    ax0.ticklabel_format(axis='y',style='sci')
                    ax1.ticklabel_format(axis='y',style='sci')
                    ax2.ticklabel_format(axis='y',style='sci')
                    ax0.grid(axis='both')
                    ax1.grid(axis='both')
                    ax2.grid(axis='both')
                    ax0.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
                    ax0.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
                    ax1.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
                    ax2.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
                    ax0.set_ylabel('$r_{V_{LOS}}$ [-]',fontsize=plot_param['axes_label_fontsize'])
                    ax0.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-4)
                    ax2.text(0.8, 0.935, textstr1, transform=ax2.transAxes, fontsize=16,horizontalalignment='left',verticalalignment='top', bbox=props1) 
                # Plot sensitivity coefficients
                SensCoeff1=[Data['Sens coeff Vh'][-1]['dV1'],Data['Sens coeff Vh'][-1]['dV2'],Data['Sens coeff Vh'][-1]['dV3']]
                c5=['black','dimgray','lightgray']
                legt=[r'$\frac{\partial{V_{h}}}{\partial{V_{LOS_1}}}$',r'$\frac{\partial{V_{h}}}{\partial{V_{LOS_2}}}$',r'$\frac{\partial{V_{h}}}{\partial{V_{LOS_{3}}}}$']
                for ind_plot in range(3):                 
                    ax13=plt.subplot(gs[2,:])         
                    plt.plot(np.degrees(Data['wind direction']),SensCoeff1[ind_plot],'-',c=c5[ind_plot],linewidth=plot_param['linewidth'],label=legt[ind_plot])
                                    

            else:
                ax2D=fig1.add_subplot(gs[1])
                ax11.set_ylabel('$U_{V_{h}}$ [m/s]',fontsize=plot_param['axes_label_fontsize']+2)
                c5=['black','dimgray','cadetblue']
                legt=[r'$\frac{\partial{V_{h}}}{\partial{V_{LOS_1}}}$',r'$\frac{\partial{V_{h}}}{\partial{V_{LOS_2}}}$',r'$\frac{\partial^2{V_{h}}}{\partial{V_{LOS_{i,j}}}}$']
                for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                    c3=next(color3)
                    ax2D.grid(axis='both')
                    ax2D.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
                    ax2D.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])                   
                    plt.plot(np.degrees(Data['wind direction']),CorrelationsGUM[0][ind_plot],'-',c=c3,linewidth=plot_param['linewidth'])
                    plt.plot(np.degrees(Data['wind direction']),CorrelationsMCM[0][ind_plot],'o', markerfacecolor=c3,markeredgecolor='lime',alpha=0.4)
                    ax2D.set_ylabel('$r_{V_{LOS}}$ [-]',fontsize=plot_param['axes_label_fontsize'])
                    ax2D.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-4)
                    ax2D.text(0.8, 0.935, textstr1, transform=ax2D.transAxes, fontsize=16,horizontalalignment='left',verticalalignment='top', bbox=props1) 
                    ax2D.set_ylim(-1,1)
                # Plot sensitivity coefficients
                SensCoeff1=[Data['Sens coeff Vh'][-1]['dV1'],Data['Sens coeff Vh'][-1]['dV2'],Data['Sens coeff Vh'][-1]['dV1V2']]
                for ind_plot in range(3):                 
                    ax13=plt.subplot(gs[2,:])         
                    plt.plot(np.degrees(Data['wind direction']),SensCoeff1[ind_plot],'-',c=c5[ind_plot],linewidth=plot_param['linewidth'],label=legt[ind_plot])
         
            # Legends
            ax11.legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax13.legend(loc=1, prop={'size': plot_param['legend_fontsize']+4.7})
            
            # Axes
            ax11.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-4)
            ax13.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-4)
            ax13.ticklabel_format(axis='y',style='sci', scilimits=(0,0))          
            ax13.set_xlabel('Wind direction[°]',fontsize=plot_param['axes_label_fontsize'])
            ax13.set_ylabel(r'$ \frac{\partial^2{V_{h}}}{\partial{V_{LOS_{i,j}}}}~\sigma_{V_{LOS_{i,j}}}~$[m/s]',fontsize=plot_param['axes_label_fontsize']-1.51) 
            ax11.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax11.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            ax11.set_xlim(0,359)
            ax11.grid(axis='both')            
            ax13.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax13.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            ax13.set_xlim(0,359)
            ax13.grid(axis='both')
            gs.update(left=0.085,right=0.99,top=0.965,bottom=0.1,wspace=0.3,hspace=0.24)           
            plt.show()
            
            #######################################################
            # 2. Plot Uncertainty in Vlos with theta       
            fig,ax2=plt.subplots() 
            color = iter(cm.rainbow(np.linspace(0, 1, len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))
            
            for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                cc=next(color)          
                # pdb.set_trace()
                ax2.plot(np.degrees(Data['lidars']['Coord_Test']['TESTt'][0]),Data['VLOS Unc [m/s]']['VLOS Uncertainty GUM theta [m/s]'][ind_plot],c=cc,label=r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot]))
                ax2.plot(np.degrees(Data['lidars']['Coord_Test']['TESTt'][0]),Data['VLOS Unc [m/s]']['VLOS Uncertainty MC theta [m/s]'][ind_plot],'or' , markerfacecolor=cc,markeredgecolor='lime',alpha=0.3,label='MC')        
        
            ax2.legend(loc=4, prop={'size': plot_param['legend_fontsize']})
            ax2.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax2.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            ax2.set_xlim(0,90)
            ax2.set_ylim(0,0.04)
            ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
          
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
            textstr = '\n'.join((
            r'$\rho~ [m]=%.1f$' % (Data['lidars']['Lidar0_Spherical']['rho'], ),
            r'$\varphi~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['psi']), ),
            r'N ={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['N_MC'], )
            ))
            ax2.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)
            plt.tight_layout()                    
            # place a tex1t box in upper left in axes coords
            ax2.text(0.5, 0.7, textstr, transform=ax2.transAxes, fontsize=18, bbox=props)
            ax2.set_xlabel('Elevation angle [°]',fontsize=plot_param['axes_label_fontsize'])
            ax2.set_ylabel('$V_{LOS}$ Uncertainty [m/s]',fontsize=plot_param['axes_label_fontsize'])
            ax2.grid(axis='both')
            plt.show()
            
            # pdb.set_trace()
            
            
            # 3. Plot Uncertainty in Vlos with psi
            fig,ax3=plt.subplots()
            color=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))              
            for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                cc=next(color)
                ax3.plot(np.degrees(Data['lidars']['Coord_Test']['TESTp'][0]),Data['VLOS Unc [m/s]']['VLOS Uncertainty GUM psi [m/s]'][ind_plot],c=cc,label=r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
                ax3.plot(np.degrees(Data['lidars']['Coord_Test']['TESTp'][0]),Data['VLOS Unc [m/s]']['VLOS Uncertainty MC psi [m/s]'][ind_plot],'or' , markerfacecolor=cc,markeredgecolor='lime',alpha=0.4,label='MC')        
            ax3.legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax3.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax3.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            ax3.set_xlim(0,359)
            ax3.set_ylim(0,0.04)
            ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            # these are matplotlib.patch.Patch properties
            props3 = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
            textstr3 = '\n'.join((
            r'$\rho ~[m]=%.1f$' % (Data['lidars']['Lidar0_Spherical']['rho'], ),
            r'$\theta~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['theta']), ),
            r'N ={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['N_MC'], )))
            
            ax3.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)
            plt.tight_layout()
            ax3.text(0.5,0.7, textstr3, transform=ax3.transAxes, fontsize=18, bbox=props3)
            ax3.set_xlabel('Azimuth angle [°]',fontsize=plot_param['axes_label_fontsize'])
            ax3.set_ylabel('$V_{LOS}$ Uncertainty [m/s]',fontsize=plot_param['axes_label_fontsize'])
            ax3.grid(axis='both')
            plt.show()
            # pdb.set_trace()



            # 4.  Plot Uncertainty in Vrad with rho                   
            fig,ax4=plt.subplots()
            color=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))          
            for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                cc=next(color)
                ax4.plot(Data['lidars']['Coord_Test']['TESTr'][0],Data['VLOS Unc [m/s]']['VLOS Uncertainty GUM rho [m/s]'][ind_plot],c=cc,label=r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
                ax4.plot(Data['lidars']['Coord_Test']['TESTr'][0],Data['VLOS Unc [m/s]']['VLOS Uncertainty MC rho [m/s]'][ind_plot],'or' , markerfacecolor=cc,markeredgecolor='lime',alpha=0.4,label='MC')      
            ax4.legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax4.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax4.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            ax4.set_xlim(0,5000)
            ax4.set_ylim(0,0.04) 
            ax4.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
            ax4.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)

            # these are matplotlib.patch.Patch properties
            props4 = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
            textstr4 = '\n'.join((
            r'$\theta~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['theta']), ),
            r'$\varphi~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['psi']), ),
             r'N ={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['N_MC'], )
            ))
        
            ax4.text(0.5,0.7, textstr4, transform=ax3.transAxes, fontsize=18, bbox=props4)
            ax4.set_xlabel('Focus distance [m]',fontsize=25)
            ax4.set_ylabel('$V_{LOS}$ Uncertainty [m/s]',fontsize=25)
            ax4.grid(axis='both')
            ax4.yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)
            plt.tight_layout()
            plt.show()
            
           
            
           
            # 5.  Plot Uncertainty in VLOS1 with wind direction 
            fig5,ax5=plt.subplots(2,1)            
            fig5.tight_layout()
            color=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))   
            
            for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                cc=next(color)
                ax5[0].plot(np.degrees(Data['wind direction']),Data['VLOS Unc [m/s]']['VLOS1 Uncertainty GUM [m/s]'][ind_plot],'-',c=cc,label=r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
                ax5[0].plot(np.degrees(Data['wind direction']),Data['VLOS Unc [m/s]']['VLOS1 Uncertainty MC [m/s]'][ind_plot],'o', markerfacecolor=cc,markeredgecolor='lime',alpha=0.4,label='Montecarlo')                       
            
            
            # Plot with sensitivity coefficients: Data['Uncertainty contributors Vlos1']=[contribution theta, contribution varphi, contribution rho] for alpha 0, 0.1 and 0.2. For the plotting we use alpha=0.2                       
            
            Cont_Theta1         = (np.array(Data['Sens coeff Vlos']['V1_theta'][-1]*np.array(np.radians(Data['STDVs'][0][0]))))**2
            Cont_Psi1           = (np.array(Data['Sens coeff Vlos']['V1_psi'][-1]*np.array(np.radians(Data['STDVs'][1][0]))))**2
            Cont_Rho1           = (np.array(Data['Sens coeff Vlos']['V1_rho'][-1]*np.array(Data['STDVs'][2][0])))**2     
            Cont_Corr1          = 2*Lidar.optics.scanner.correlations[9]*np.array(Data['Sens coeff Vlos']['V1_theta'][2])*np.array(Data['Sens coeff Vlos']['V1_psi'][2])*np.array(np.radians(Data['STDVs'][0][0]))*np.array(np.radians(Data['STDVs'][1][0]))

            ax5[1].plot(np.degrees(Data['wind direction']),Cont_Theta1,'-',c='black',linewidth=plot_param['linewidth'],label=r'$\frac{\partial{V_{LOS}}}{\partial{\theta}}$')
            ax5[1].plot(np.degrees(Data['wind direction']),Cont_Psi1 ,'-',c='dimgray',linewidth=plot_param['linewidth'],label=r'$\frac{\partial{V_{LOS}}}{\partial{\varphi}}$')
            ax5[1].plot(np.degrees(Data['wind direction']),Cont_Rho1,'-',c='lightgray',linewidth=plot_param['linewidth'],label=r'$\frac{\partial{V_{LOS}}}{\partial{\rho}}$')
            ax5[1].plot(np.degrees(Data['wind direction']),Cont_Corr1 ,'-',c='cadetblue',linewidth=plot_param['linewidth'],label=r'$\frac{\partial^2{V_{LOS}}}{\partial{\theta}\partial{\varphi}}$')
            ax5[1].set_xlabel('Wind Direction [°]',fontsize=plot_param['axes_label_fontsize'])
            ax5[0].set_ylabel('$V_{LOS}$ uncertainty [m/s]',fontsize=plot_param['axes_label_fontsize'])
            ax5[1].set_ylabel(r'$ \frac{\partial^2{V_{LOS}}}{\partial{\theta}\partial{\varphi}}~\sigma_{\theta \varphi}~$[m/s]',fontsize=plot_param['axes_label_fontsize']+.5)
            ax5[0].set_xlim(0,359)
            ax5[1].set_xlim(0,359)            
            
            
            ax5[0].grid(axis='both') 
            ax5[1].grid(axis='both') 
            ax5[0].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax5[1].legend(loc=1, prop={'size': 5+plot_param['legend_fontsize']})  
            ax5[0].tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax5[1].tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax5[0].tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            ax5[1].tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            props5 = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
            textstr5 = '\n'.join((
            r'$\rho ~[m]=%.1f$' % (Data['lidars']['Lidar0_Spherical']['rho'] ),
            r'$\theta~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['theta'])),
            r'$\varphi~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar0_Spherical']['psi'])),
            r'$r_{\theta,\varphi}~ =%.1f$' % (Lidar.optics.scanner.correlations[3]),
            r'N ={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['N_MC'] ),           
            ))           
            ax5[0].text(0.5, 0.95, textstr5, transform=ax5[0].transAxes, fontsize=14,horizontalalignment='left',verticalalignment='top', bbox=props5)
            ax5[0].ticklabel_format(axis='y',style='sci', scilimits=(0,0))
            ax5[1].ticklabel_format(axis='y',style='sci', scilimits=(0,0))
            ax5[0].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)
            ax5[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)
 


           # 6.  Plot Uncertainty in VLOS2 with wind direction 
            fig6,ax6=plt.subplots(2,1)  
            fig6.tight_layout()
            color=iter(cm.rainbow(np.linspace(0,1,len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent']))))   
            for ind_plot in range(len(Data['Vh Unc [m/s]']['Uncertainty Vh MCM'])):
                cc=next(color)
                ax6[0].plot(np.degrees(Data['wind direction']),Data['VLOS Unc [m/s]']['VLOS2 Uncertainty GUM [m/s]'][ind_plot],'-',c=cc,label=r'GUM ($\alpha$={})'.format(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'][ind_plot] ))
                ax6[0].plot(np.degrees(Data['wind direction']),Data['VLOS Unc [m/s]']['VLOS2 Uncertainty MC [m/s]'][ind_plot],'o', markerfacecolor=cc,markeredgecolor='lime',alpha=0.4,label='Montecarlo')
                        
            # Plot with sensitivity coefficients: Data['Uncertainty contributors Vlos1']=[contribution theta, contribution varphi, contribution rho] for alpha 0, 0.1 and 0.2. For the plotting we use alpha=0.2            
            Cont_Theta2         = (np.array(Data['Sens coeff Vlos']['V2_theta'][-1]*np.array(np.radians(Data['STDVs'][0][1]))))**2
            Cont_Psi2           = (np.array(Data['Sens coeff Vlos']['V2_psi'][-1]*np.array(np.radians(Data['STDVs'][1][1]))))**2
            Cont_Rho2           = (np.array(Data['Sens coeff Vlos']['V2_rho'][-1]*np.array(Data['STDVs'][2][1])))**2     
            Cont_Corr2          = 2*Lidar.optics.scanner.correlations[10]*np.array(Data['Sens coeff Vlos']['V2_theta'][0])*np.array(Data['Sens coeff Vlos']['V2_psi'][0])*np.array(np.radians(Data['STDVs'][1][1]))*np.array(np.radians(Data['STDVs'][0][1]))
            # pdb.set_trace()
            # Plotting contributors:
            ax6[1].plot(np.degrees(Data['wind direction']),Cont_Theta2,'-',c='black',linewidth=plot_param['linewidth'],label=r'$\frac{\partial{V_{LOS}}}{\partial{\theta}}$')
            ax6[1].plot(np.degrees(Data['wind direction']),Cont_Psi2  ,'-',c='dimgray',linewidth=plot_param['linewidth'],label=r'$\frac{\partial{V_{LOS}}}{\partial{\varphi}}$')
            ax6[1].plot(np.degrees(Data['wind direction']),Cont_Rho2  ,'-',c='lightgray',linewidth=plot_param['linewidth'],label=r'$\frac{\partial{V_{LOS}}}{\partial{\rho}}$')
            ax6[1].plot(np.degrees(Data['wind direction']),Cont_Corr2 ,'-',c='cadetblue',linewidth=plot_param['linewidth'],label=r'$\frac{\partial^2{V_{LOS}}}{\partial{\theta}\partial{\varphi}}$')

            ax6[1].set_xlabel('Wind Direction [°]',fontsize=plot_param['axes_label_fontsize'])
            ax6[0].set_ylabel('$V_{LOS}$ uncertainty [m/s]',fontsize=plot_param['axes_label_fontsize'])
            ax6[1].set_ylabel(r'$ \frac{\partial^2{V_{LOS}}}{\partial{\theta}\partial{\varphi}}~\sigma_{\theta\varphi}$[m/s]',fontsize=plot_param['axes_label_fontsize']+.5)
            ax6[0].set_xlim(0,359)
            ax6[1].set_xlim(0,359)
            ax6[0].grid(axis='both') 
            ax6[1].grid(axis='both') 
            ax6[0].legend(loc=1, prop={'size': plot_param['legend_fontsize']})
            ax6[1].legend(loc=1, prop={'size': 5+plot_param['legend_fontsize']})  
            ax6[0].tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax6[1].tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
            ax6[0].tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            ax6[1].tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            props5 = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
            textstr5 = '\n'.join((
            r'$\rho ~[m]=%.1f$' % (Data['lidars']['Lidar1_Spherical']['rho']  ),
            r'$\theta~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar1_Spherical']['theta'] ), ),
            r'$\varphi~ [°]=%.1f$' % (np.degrees(Data['lidars']['Lidar1_Spherical']['psi'] ), ),
            r'$r_{\theta,\varphi}~ =%.1f$' % (Lidar.optics.scanner.correlations[6]),
            r'N ={}'.format(Qlunc_yaml_inputs['Components']['Scanner']['N_MC'], ),           
            ))           
            ax6[0].text(0.5, 0.95, textstr5, transform=ax6[0].transAxes, fontsize=14,horizontalalignment='left',verticalalignment='top', bbox=props5)
            ax6[0].ticklabel_format(axis='y',style='sci', scilimits=(0,0))
            ax6[1].ticklabel_format(axis='y',style='sci', scilimits=(0,0))            
            ax6[0].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)
            ax6[1].yaxis.get_offset_text().set_fontsize(plot_param['tick_labelfontsize']-3)
           

        #%% Plot the vertical/horizontal plane
        # pdb.set_trace()
        if Lidar.optics.scanner.pattern in ['vertical plane'] or Lidar.optics.scanner.pattern in ['horizontal plane']:
            # pdb.set_trace()
            V=[]
            Dir=[]
            for i in range(int((len(Data['Sens coeff Vh'])/len(Qlunc_yaml_inputs['Atmospheric_inputs']['Power law exponent'])))):
                V.append(Data['Vh Unc [m/s]']['Uncertainty Vh GUM'][i][0])
                Dir.append(Data['WinDir Unc [°]']['Uncertainty wind direction GUM'][i][0])         
            
            # Reshape V and avoid nans and infinit values
            VV=np.reshape(V,[int(np.sqrt(len(V))),int(np.sqrt(len(V)))])
            VV[VV>15]=15
            
            # Horizontal wind velocity
            col ='binary' #'binary' #
            cmaps = matplotlib.cm.get_cmap(col)  # viridis is the default colormap for imshow
            cmap = matplotlib.cm.ScalarMappable(norm = mcolors.Normalize(vmin=VV.min(), vmax=VV.max()),cmap = plt.get_cmap(col))

            fig00,ax00=plt.subplots()
            if  Lidar.optics.scanner.pattern in ['vertical plane']:
                XX=np.reshape(Data['lidars']['Coord_Out'][1],[int(np.sqrt(len(V))),int(np.sqrt(len(V)))])
                YY=np.reshape(Data['lidars']['Coord_Out'][2],[int(np.sqrt(len(V))),int(np.sqrt(len(V)))])
                ax00.set_xlabel('Y [m]', fontsize=plot_param['tick_labelfontsize']+20, labelpad=15)
                ax00.set_ylabel('Z [m]', fontsize=plot_param['tick_labelfontsize']+20, labelpad=15)
            elif  Lidar.optics.scanner.pattern in ['horizontal plane']:
                XX=np.reshape(Data['lidars']['Coord_Out'][0],[int(np.sqrt(len(V))),int(np.sqrt(len(V)))])
                YY=np.reshape(Data['lidars']['Coord_Out'][1],[int(np.sqrt(len(V))),int(np.sqrt(len(V)))])
                ax00.set_xlabel('X [m]', fontsize=plot_param['tick_labelfontsize']+20, labelpad=15)
                ax00.set_ylabel('Y [m]', fontsize=plot_param['tick_labelfontsize']+20, labelpad=15)
                ax00.plot(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0][0],Qlunc_yaml_inputs['Components']['Scanner']['Origin'][0][1],'sk', ms=5, mec='white', mew=1.5)
                ax00.plot(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1][0],Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1][1],'sk', ms=5, mec='white', mew=1.5)
                ax00.plot(Qlunc_yaml_inputs['Components']['Scanner']['Origin'][1][0],Qlunc_yaml_inputs['Components']['Scanner']['Origin'][2][1],'sk', ms=5, mec='white', mew=1.5)
     
            plt.contourf(XX,YY, VV,50,cmap=cmaps,vmin=VV.min(), vmax=VV.max())
            cmap.set_array([]) # or alternatively cmap._A = []

            colorbar=fig00.colorbar(cmap, ax = ax00)                        
            colorbar.set_label(label='Uncertainty [m/s]', size=plot_param['tick_labelfontsize']+12, labelpad=15)
            colorbar.ax.tick_params(labelsize=19)
            ax00.set_aspect('equal')
            ax00.ticklabel_format(useOffset=False)
           

            ax00.xaxis.set_tick_params(labelsize=plot_param['tick_labelfontsize']+14)
            ax00.yaxis.set_tick_params(labelsize=plot_param['tick_labelfontsize']+14)
            plt.show()
            
            # pdb.set_trace()
           
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
            
            
            #%% Wind direction
            # colorsMap='jet'
            # cm2 = plt.get_cmap(colorsMap)
            # cNorm2 = matplotlib.colors.Normalize(vmin=0.15, vmax=0.2)
            # scalarMap1 = cmx.ScalarMappable(norm=cNorm2, cmap=cm2)
            
            # fig1 = plt.figure()
            # ax1 = Axes3D(fig1)
            # ax1.scatter(Data['lidars']['Coord_Out'][0],Data['lidars']['Coord_Out'][1], Data['lidars']['Coord_Out'][2], Dir, c=scalarMap1.to_rgba(Dir))
            # ax1.set_xlabel('X [m]', fontsize=plot_param['tick_labelfontsize'], labelpad=15)
            # ax1.set_ylabel('Y [m]', fontsize=plot_param['tick_labelfontsize'], labelpad=15)
            # ax1.set_zlabel('Z [m]', fontsize=plot_param['tick_labelfontsize'], labelpad=15)
            
            # ax1.plot(Data['lidars']['Lidar0_Rectangular']['LidarPosX'],Data['lidars']['Lidar0_Rectangular']['LidarPosY'],Data['lidars']['Lidar0_Rectangular']['LidarPosZ'],'sb')
            # ax1.plot(Data['lidars']['Lidar1_Rectangular']['LidarPosX'],Data['lidars']['Lidar1_Rectangular']['LidarPosY'],Data['lidars']['Lidar1_Rectangular']['LidarPosZ'],'sb')
            # # pdb.set_trace()
            # scalarMap1.set_array(Data['WinDir Unc [°]']['Uncertainty wind direction GUM'])
            # cb1=plt.colorbar(scalarMap1, shrink=0.5)
            # cb1.set_label(label='$\Omega$ Uncertainty [°]', size=plot_param['tick_labelfontsize'])
            # # cb1.ax1.tick_params(labelsize=13)
            # # cb.ax.tick_params(labelsize=13)
            # ax1.ticklabel_format(useOffset=False)
            # plt.show()
            # ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
            # ax1.xaxis.set_tick_params(labelsize=plot_param['tick_labelfontsize']-3)
            # ax1.yaxis.set_tick_params(labelsize=plot_param['tick_labelfontsize']-3)
            # ax1.zaxis.set_tick_params(labelsize=plot_param['tick_labelfontsize']-3)

            # ax1.set_box_aspect([ub - lb for lb, ub in (getattr(ax1, f'get_{a}lim')() for a in 'xyz')])
        
    ###############   Plot photodetector noise   #############################       
    if flag_plot_photodetector_noise:
        # Quantifying uncertainty from photodetector and interval domain for the plot Psax is define in the photodetector class properties)
        Psax=(Lidar.photonics.photodetector.Power_interval)
        # Plotting:
        fig,axs1=plt.subplots()
        label0=['Shot','Thermal','Dark current','TIA','Total']
        i_label=0
        col=['darkturquoise','darkgoldenrod','slategray','navy','red']
        for i in Data['SNR_data_photodetector']:            
            axs1.plot(Psax,Data['SNR_data_photodetector'][i][0],color=col[i_label],label=label0[i_label], linewidth=2.3)  
            i_label+=1
        axs1.tick_params(axis='x', labelsize=plot_param['tick_labelfontsize'])
        axs1.tick_params(axis='y', labelsize=plot_param['tick_labelfontsize'])
            
        # axs1.plot(Psax,Data['Total_SNR_data'],label='Total SNR')
        axs1.set_xlabel('Input Signal optical power [W]',fontsize=plot_param['axes_label_fontsize'])
        axs1.set_ylabel('SNR [dB]',fontsize=plot_param['axes_label_fontsize'])
        axs1.legend(fontsize=plot_param['legend_fontsize'],loc='upper right')
        # axs1.set_title('SNR - Photodetector',fontsize=plot_param['title_fontsize'])
        axs1.grid(axis='both')
        axs1.text(.90,.05,plot_param['Qlunc_version'],transform=axs1.transAxes, fontsize=14,verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    
    ###############   Plot Probe Volume parameters    ############################
    if flag_probe_volume_param: 
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
        # Rayleigh length variation due to focus_distance variations (due to the distance between fiber-end and telescope lens)
        zr = (wave*(focus_distance**2))/(np.pi*(Qlunc_yaml_inputs['Components']['Telescope']['Effective radius telescope'])**2)# Rayleigh length  (considered as the probe length) # half-width of the weighting function --> FWHM = 2*zr
    
        # Probe volume:
        #Probe_volume = np.pi*(A**2)*((4*(focus_distance**2)*wave)/(Telescope_aperture)) # based on Marijn notes
        vol_zr       = np.pi*(A**2)*(2*zr) # based on the definition of Rayleigh length in Liqin Jin notes (Focus calibration formula)
        
        # Lorentzian weighting function:
        
        phi = (ext_coef/np.pi)*(zr/((zr**2)+(s-focus_distance)**2))
        # phi = (ext_coef/np.pi)*(zr/((zr**2)))
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




################### PLOT COORDINATE SYSTEM DUAL LIDAR ##############################################
# from matplotlib.patches import Circle,Wedge
# import mpl_toolkits.mplot3d.art3d as art3d
# from matplotlib.patches import FancyArrowPatch
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# r = 89
# x0 = 500 # To have the tangent at y=0
# z0 = 119

# # Theta varies only between pi/2 and 3pi/2. to have a half-circle
# theta = np.linspace(0., 2*np.pi, 161)

# x = np.zeros_like(theta)+x0 # x=0
# y = r*np.cos(theta)  # y - y0 = r*cos(theta)
# z = r*np.sin(theta) + z0 # z - z0 = r*sin(theta)
# ax.plot(x, y, z,'k--',linewidth=2,label='Rotor area')



# x1,y1,z1=[500,0],[0,-150],[119,1] 
# x2,y2,z2=[500,0],[ 0,150],[119,1] 
# p = patches.Rectangle((89.15, 29.85), -89.15*2,89.15*2,alpha=0.387,label='Scanned area')
# ax.add_patch(p)
# art3d.pathpatch_2d_to_3d(p, z=500, zdir="x")



# x1,y1,z1=[500,0],[0,-150],[119,1] 
# x2,y2,z2=[500,0],[ 0,150],[119,1] 
# # p = Wedge((0, 119), 89.15,0,359,alpha=0.5,label='WT area',width=1.71, ls='--')
# # ax.add_patch(p)
# # art3d.pathpatch_2d_to_3d(p, z=500, zdir="x")
# ax.scatter(500,0, 119, c='r', s=50, marker='o', label=r'$P~(x,y,z)$')
# ax.scatter(0, -150, 1, c='b', s=50, marker='s', label=r'$Lidar_1~and~Lidar_2$')
# ax.scatter(0, 150, 1, c='b', s=50, marker='s')

# ax.plot(x1, y1, z1, color='magenta',linestyle='dashed')
# ax.plot(x2, y2, z2, color='magenta',linestyle='dashed')

# ax.set_xlabel('X [m]', fontsize=21,labelpad=15)
# ax.set_ylabel('Y [m]', fontsize=21,labelpad=15)
# ax.set_zlabel('Z [m]', fontsize=21,labelpad=15)
# ax.set_zlim([0,250])
# ax.set_xlim([0,570])
# plt.legend(loc="best", fontsize=16.23)

# ax.xaxis.set_tick_params(labelsize=15)
# ax.yaxis.set_tick_params(labelsize=15)
# ax.zaxis.set_tick_params(labelsize=15)


# fig = plt.figure()
# ax = fig.gca(projection='3d')


# x1,y1,z1=[500,0],[0,-150],[119,1] 
# x2,y2,z2=[500,0],[ 0,150],[119,1] 
# p = patches.Rectangle((89.15, 29.84), -89.15*2,89.15*2,alpha=0.37)
# ax.add_patch(p)
# art3d.pathpatch_2d_to_3d(p, z=500, zdir="x")



# x1,y1,z1=[500,0],[0,-150],[119,1] 
# x2,y2,z2=[500,0],[ 0,150],[119,1] 
# p = Wedge((0, 119), 89.15,0,360,alpha=0.9,label='Rotor diameter',width=3, ls='--')
# ax.add_patch(p)
# art3d.pathpatch_2d_to_3d(p, z=500, zdir="x")
# ax.scatter(500,0, 119, c='r', s=50, marker='o', label=r'$P~(x,y,z)$')
# ax.scatter(0, -150, 1, c='b', s=50, marker='s', label=r'$Lidar_1~and~Lidar_2$')
# ax.scatter(0, 150, 1, c='b', s=50, marker='s')

# ax.arrow
# ax.plot(x1, y1, z1, color='g',linestyle='dashed')
# ax.plot(x2, y2, z2, color='g',linestyle='dashed')

# ax.set_xlabel('X [m]', fontsize=21,labelpad=15)
# ax.set_ylabel('Y [m]', fontsize=21,labelpad=15)
# ax.set_zlabel('Z [m]', fontsize=21,labelpad=15)
# ax.set_zlim([0,250])
# ax.set_xlim([0,550])
# plt.legend(loc="best", fontsize=16.23)

# ax.xaxis.set_tick_params(labelsize=15)
# ax.yaxis.set_tick_params(labelsize=15)
# ax.zaxis.set_tick_params(labelsize=15)




#%% Plotting correlations
    # flag_plot_correlations =1
    # if flag_plot_correlations ==1:
        
    #     fig0,az0=plt.subplots(3,sharex=True)
    #     fig0.add_subplot(111, frameon=False)
    #     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        
    #     plt.xlabel("Samples",fontsize=25)
    #     plt.ylabel("Correlation",fontsize=25)
    #     az0[0].plot(Data['Correlations'][3])
    #     az0[1].plot(Data['Correlations'][7])
    #     az0[2].plot(Data['Correlation uv'])
    #     # fig0.suptitle('Correlations Vlos, uv',fontsize=25)
    #     az0[0].set_ylim(-1,1)
    #     az0[1].set_ylim(-1,1)
    #     az0[2].set_ylim(-1,1)
    #     az0[0].title.set_text('$V_{LOS_{1}} - V_{LOS_{2}}$ - 1st Multivariate')
    #     az0[1].title.set_text('$V_{LOS_{1}} - V_{LOS_{2}}$ - 2nd Multivariate')
    #     az0[2].title.set_text('$u - v$')
    #     az0[0].grid(axis='both')
    #     az0[1].grid(axis='both')
    #     az0[2].grid(axis='both')
    #     plt.xlabel("Wind direction [°]",fontsize=25)
    #     plt.ylabel("Correlation",fontsize=25)
    #     # pdb.set_trace()
        
        
    #     ###1st Multi
    #     fig1,az1=plt.subplots(2,3)  
    #     fig1.add_subplot(111, frameon=False)
    #     # hide tick and tick label of the big axes
    #     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #     plt.grid(False)
    #     plt.xlabel("Samples",fontsize=25)
    #     plt.ylabel("Correlation",fontsize=25)
    #     az1[0][0].plot(Data['Correlations'][2])
    #     az1[0][1].plot(Data['Correlations'][0])
    #     az1[0][2].plot(Data['Correlations'][5])    
    #     az1[1][0].plot(Data['Correlations'][4])
    #     az1[1][1].plot(Data['Correlations'][1])
    #     az1[1][2].plot(Data['Correlations'][15])    
    #     az1[0][0].title.set_text('theta1theta2')
    #     az1[0][1].title.set_text('theta1psi1')
    #     az1[0][2].title.set_text('theta1psi2')
    #     az1[1][0].title.set_text('psi1psi2')
    #     az1[1][1].title.set_text('theta2psi2')
    #     az1[1][2].title.set_text('theta2psi1')    
    #     fig1.suptitle('$1^{st}$ Multivariate',fontsize=25)
    #     az1[0][0].grid(axis='both')
    #     az1[0][1].grid(axis='both')
    #     az1[0][2].grid(axis='both')
    #     az1[1][0].grid(axis='both')
    #     az1[1][1].grid(axis='both')
    #     az1[1][2].grid(axis='both')
    #     # pdb.set_trace()
        
    #     ###2nd Multi
    #     fig2,az2=plt.subplots(2,3)  
    #     fig2.add_subplot(111, frameon=False)
    #     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #     plt.grid(False)
    #     plt.xlabel("Samples",fontsize=25)
    #     plt.ylabel("Correlation",fontsize=25)
    #     az2[0][0].plot(Data['Correlations'][8])
    #     az2[0][1].plot(Data['Correlations'][11])
    #     az2[0][2].plot(Data['Correlations'][12])
        
    #     az2[1][0].plot(Data['Correlations'][9])
    #     az2[1][1].plot(Data['Correlations'][13])
    #     az2[1][2].plot(Data['Correlations'][14])
        
    #     az2[0][0].title.set_text('theta1theta2')
    #     az2[0][1].title.set_text('theta1psi1')
    #     az2[0][2].title.set_text('theta1psi2')
    #     az2[1][0].title.set_text('psi1psi2')
    #     az2[1][1].title.set_text('theta2psi2')
    #     az2[1][2].title.set_text('theta2psi1')    
    #     fig2.suptitle('$2^{nd}$ Multivariate',fontsize=25)    
    #     az2[0][0].grid(axis='both')
    #     az2[0][1].grid(axis='both')
    #     az2[0][2].grid(axis='both')
    #     az2[1][0].grid(axis='both')
    #     az2[1][1].grid(axis='both')
    #     az2[1][2].grid(axis='both')
    #     # pdb.set_trace()
        
    #     fig2,az2=plt.subplots(2)   
    #     az2[0].plot(CorrCoef_U_VLOS)
    #     az2[1].plot(CorrCoef_U_uv)
    #     az2[0].set_ylabel(r'$r_{U_{V1V2}}$',fontsize=21)
    #     az2[1].set_ylabel(r'$r_{U_{uv}}$',fontsize=21)
        
    #     fig3,az3=plt.subplots(1,3)
    #     fig3.add_subplot(111, frameon=False)
    #     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #     plt.grid(False)
    #     plt.xlabel("Samples",fontsize=25)
    #     plt.ylabel("Correlation",fontsize=25)
    #     az3[0].plot(Vlos1_MC_cr2,Vlos2_MC_cr2,'bo',alpha=0.4)
    #     az3[1].plot(Theta1_cr2,Theta2_cr2,'bo',alpha=0.4)
    #     az3[2].plot(Psi1_cr2,Psi2_cr2,'bo',alpha=0.4)
    #     az3[0].set_aspect(1) 
    #     az3[1].set_aspect(1) 
    #     az3[2].set_aspect(1)    
    #     az3[0].title.set_text('Vlos_MC_cr2')
    #     az3[1].title.set_text('Theta_cr2')
    #     az3[2].title.set_text('Psi_cr2')
        
    #     # pdb.set_trace()
        

    #     Correlations in uncertainties
    #     fig4,az4=plt.subplots(1,2)
    #     fig4.add_subplot(111, frameon=False)
    #     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #     plt.grid(False)
    #     plt.xlabel("Samples",fontsize=25)
    #     plt.ylabel("Correlation",fontsize=25)

        
    #     # az4[0][0].plot(Theta1_cr2,Theta2_cr2,'bo',alpha=0.4)
    #     # az4[1][0].plot(Psi1_cr2,Psi2_cr2,'bo',alpha=0.4)
        

        
    #     az4[0].plot(Data['VLOS1 Uncertainty MC [m/s]'],Data['VLOS2 Uncertainty MC [m/s]'],'bo',alpha=0.4)  
    #     # Indexes for reference
    #     az4[0].plot(Data['VLOS1 Uncertainty MC [m/s]'][0],Data['VLOS2 Uncertainty MC [m/s]'][0],'ro')
    #     az4[0].plot(Data['VLOS1 Uncertainty MC [m/s]'][179],Data['VLOS2 Uncertainty MC [m/s]'][179],'go')
    #     az4[0].plot(Data['VLOS1 Uncertainty MC [m/s]'][89],Data['VLOS2 Uncertainty MC [m/s]'][89],'ko')
    #     az4[0].plot(Data['VLOS1 Uncertainty MC [m/s]'][270],Data['VLOS2 Uncertainty MC [m/s]'][270],'yo')
        
    #     az4[1].plot(Data['Uncertainty u wind component MCM'],Data['Uncertainty v wind component MCM'],'bo',alpha=0.4)  
    #     az4[1].plot(Data['Uncertainty u wind component MCM'][0],Data['Uncertainty v wind component MCM'][0],'ro')
    #     az4[1].plot(Data['Uncertainty u wind component MCM'][179],Data['Uncertainty v wind component MCM'][179],'go')
    #     az4[1].plot(Data['Uncertainty u wind component MCM'][89],Data['Uncertainty v wind component MCM'][89],'ko')
    #     az4[1].plot(Data['Uncertainty u wind component MCM'][270],Data['Uncertainty v wind component MCM'][270],'yo')
        
    #     az4[0].set_aspect(1) 
    #     # az4[1][0].set_aspect(1) 
    #     az4[1].set_aspect(1) 
    #     # az4[1][1].set_aspect(1) 
        
    #     az4[0].set_xlabel('$U_{V_{LOS1}}$',fontsize=21)
    #     az4[1].set_xlabel('$U_u$',fontsize=21)
    #     az4[0].set_ylabel('$U_{V_{LOS2}}$',fontsize=21)
    #     az4[1].set_ylabel('$U_v$',fontsize=21)
    #     az4[0].grid(axis='both')
    #     az4[1].grid(axis='both')

    #     az4[0][0].set_xlabel(r'$\theta_{1}$',fontsize=21)
    #     az4[1][0].set_xlabel(r'$\varphi{1}$',fontsize=21)
    #     az4[0][0].set_ylabel(r'$\theta_{2}}$',fontsize=21)
    #     az4[1][0].set_ylabel(r'$\varphi{2}$',fontsize=21)
    
           
