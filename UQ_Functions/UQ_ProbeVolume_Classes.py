# # -*- coding: utf-8 -*-
# """
# Created on Wed May 19 12:48:10 2021

# @author: fcosta
# """
# '''
# This works for a CW monostatic coherent lidar 
# '''


from Utils.Qlunc_ImportModules import *
from Utils import Qlunc_Help_standAlone as SA
from Utils import Scanning_patterns as SP
from Utils import Qlunc_Plotting as QPlot

def UQ_Probe_volume (Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    
    # f_length  = 200e-3 # focal length
    # Qlunc_yaml_inputs['Probe Volume']['Fiber-lense distance']         = np.arange(2e-3,4e-3,.02e-3) # distance fiber-end--telescope lens
    # a0        = 198.1e-3 # the offset (a constant number), to avoid the fiber-end locates at the focal point, otherwise the lights will be parallel to each other
    # A         = 20e-3 # beam radius at the output lens
    # ext_coef  = 0.085
    # effective_radius_telescope  = 16.5e-3
    
    # %% Liqin jin
    if Qlunc_yaml_inputs['Components']['Lidar general inputs']['Type']=="CW":
        # The focus distance varies with the distance between the fiber-end and the telescope lens. So that, also the probe length varies with such distance.
        #Calculating focus distance depending on the distance between the fiber-end and the telescope lens:
        f_distance = 1/((1/Qlunc_yaml_inputs['Probe Volume']['Focal length'])-(1/(Qlunc_yaml_inputs['Probe Volume']['Fiber-lens distance']+Qlunc_yaml_inputs['Probe Volume']['Fiber-lens offset']))) # Focus distance
        # dist =(np.linspace(0,60,len(Qlunc_yaml_inputs['Probe Volume']['Fiber-lens distance'])))  # distance from the focus position along the beam direction
        
        # Rayleigh length variation due to f_distance variations (due to the distance between fiber-end and telescope lens)
        zr = (Qlunc_yaml_inputs['Components']['Laser']['Wavelength']*(f_distance**2))/(np.pi*(Qlunc_yaml_inputs['Probe Volume']['Effective radius telescope'])**2)# Rayleigh length  (considered as the probe length) # half-width of the weighting function --> FWHM = 2*zr
        
        # Saving coordenates to a file in desktop
        if os.path.isfile('./metadata/rayleigh_distance.txt'):
            os.remove('./metadata/rayleigh_distance.txt')
            file=open('./metadata/rayleigh_distance.txt','w')
            file.write(repr(zr))
            file.close()   
            pdb.set_trace()
        else:
            file=open('./metadata/rayleigh_distance.txt','w')
            file.write(repr(zr))
            file.close() 
        # Probe volume:
        #Probe_volume = np.pi*(Qlunc_yaml_inputs['Probe Volume']['Output beam radius']**2)*((4*(f_distance**2)*Qlunc_yaml_inputs['Components']['Laser']['Wavelength'])/(Telescope_aperture)) # based on Marijn notes
        #VolCil       = np.pi*(Qlunc_yaml_inputs['Probe Volume']['Output beam radius']**2)*fwhm  # calculated based on the fwhm
        vol_zr       = np.pi*(Qlunc_yaml_inputs['Probe Volume']['Output beam radius']**2)*(2*zr) # based on the definition of Rayleigh distance in Liqin Jin notes (Focus calibration formula)
        
        # Lorentzian weighting function:
        phi = (Qlunc_yaml_inputs['Probe Volume']['Extinction coeficient']/np.pi)*(1/((1**2)+(36.55-f_distance)**2))
        # F = (lamb/np.pi)/(a1**2+lamb**2)  # Lorentzian Weighting function 
    elif typeLidar=="Pulsed":
        print("pulsed lidar probe volume is a convolution between pulse shape and weighting function. Not inplemented yet")
        
    
    # Plotting:
        
    # fig=plt.figure()
    # ax=fig.add_subplot(2,1,1)
    # ax.plot(dist,phi)
    # ax.set_yscale('log')
    
        
    # fig2=plt.figure()
    # ax=fig.add_subplot(2,1,2)
    # ax.plot(f_distance,zr)
    return zr 
    #%% ################################ FWHM ##############################
    
    # Method to calculate FWHM
    # def lin_interp(x, y, i, half):
    #     return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))
    
    # def half_max_x(x, y):
    #     half = max(y)/2.0
    #     signs = np.sign(np.add(y, -half))
    #     zero_crossings = (signs[0:-2] != signs[1:-1])
    #     zero_crossings_i = np.where(zero_crossings)[0]
    #     # plt.figure()
    #     # plt.plot(dist,F,  linewidth=3)    
    #     return [lin_interp(x, y, zero_crossings_i[0], half),
    #             lin_interp(x, y, zero_crossings_i[1], half)]
    # hmx = half_max_x(a1,F)
    # fwhm = hmx[1] - hmx[0]
    
    # # ###################################################################
    
    # # #%% Probe volume
    
    
    # #%% Uncertainty:
    # Unc_R=0.1  # uncertainty in focus distance
    # Unc_wave=1e-9 # Uncertainty in wavelength
    # Unc_aperture=3e-6 # uncertainty in beam radius at focus length 
    # Unc_PV=np.sqrt(((8*f_distance*wave/aperture)*Unc_R)**2+((4*(f_distance**2)/aperture)*Unc_wave)**2+((4*(f_distance**2)*wave/(aperture**2))*Unc_aperture)**2)  # Uncertainty in probe volume
    
    # print('Lamb: {:.3f}'.format(lamb))
    # # print("FWHM: {:.3f}".format(fwhm))
    # print("ProbeVolume: {:.3f}".format(Probe_volume))
    # print("Uncertainty PV: {:.3f}".format(Unc_PV))
    # print("Error[%]: {:.3f}".format(100*Unc_PV/Probe_volume))
    # # print("Cilinder Volume:{:.3f}".format(VolCil))
    
    
    # plt.figure()
    # plt.plot(dist,F)
    
    
    
