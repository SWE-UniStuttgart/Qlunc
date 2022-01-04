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

def UQ_Probe_volume (Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs,param1):
    # Liqin jin model
    if Qlunc_yaml_inputs['Components']['Lidar general inputs']['Type']=="CW":
        # The focus distance varies with the focal length and the distance between the fiber-end and the telescope lens as well. So that, also the probe length varies with such distance.
        # Calculating focus distance depending on the distance between the fiber-end and the telescope lens:
        
        r                         = Qlunc_yaml_inputs['Components']['Telescope']['Focal length']
        a                         = Qlunc_yaml_inputs['Components']['Telescope']['Fiber-lens distance']
        a0                        = Qlunc_yaml_inputs['Components']['Telescope']['Fiber-lens offset']
        wavelength                = Qlunc_yaml_inputs['Components']['Laser']['Wavelength']
        rad_eff                   = Qlunc_yaml_inputs['Components']['Telescope']['Effective radius telescope']
        Unc_r                     = Qlunc_yaml_inputs['Components']['Telescope']['stdv Focal length']
        Unc_a                     = Qlunc_yaml_inputs['Components']['Telescope']['stdv Fiber-lens distance']
        Unc_a0                    = Qlunc_yaml_inputs['Components']['Telescope']['stdv Fiber-lens offset']
        Unc_wavelength            = Qlunc_yaml_inputs['Components']['Laser']['stdv Wavelength']
        Unc_eff_radius_telescope  = Qlunc_yaml_inputs['Components']['Telescope']['stdv Effective radius telescope']
        
       
        # Focus distance
        # focus_distance = 1/((1/r)-(1/(a+a0))) # This parameters are complicated to know for users. Maybe it is easier to put just a simple parameter representing the focus distance(*)
        focus_distance = param1 #(*)
        
        # Uncertainty in focus distance
        # Unc_focus_distance = np.sqrt((((1/r**2)/(((1/r)-(1/(a+a0)))**2))*Unc_r)**2 + (((1/(a+a0)**2)/(((1/r)-(1/(a+a0)))**2))*Unc_a)**2 + (((1/(a+a0)**2)/(((1/r)-(1/(a+a0)))**2))*Unc_a0)**2) # This parameters are complicated to know for users. Maybe it is easier to put just a simple parameter representing the focus distance(**)
        Unc_focus_distance = Qlunc_yaml_inputs['Components']['Scanner']['stdv focus distance'] #(**)
        
        
        # Rayleigh length variation due to focus_distance variations (due to the distance between fiber-end and telescope lens)
        # zr= (wavelength*(ind_focusdist**2))/(np.pi*(rad_eff)**2)# Rayleigh length  (considered as the probe length) # half-width of the weighting function --> FWHM = 2*zr
        # Unc_zr= np.sqrt(((ind_focusdist**2)*Unc_wavelength/(np.pi*rad_eff))**2 + ((2*wavelength*ind_focusdist*Unc_focus_distance)/(np.pi*rad_eff**2))**2 + ((2*wavelength*(ind_focusdist**2)*Unc_eff_radius_telescope)/(np.pi*rad_eff**3))**2)

        Rayleigh_length=[]
        Unc_Rayleigh_length=[]
        for ind_focusdist in focus_distance:
            Rayleigh_length.append( (wavelength*(ind_focusdist**2))/(np.pi*(rad_eff)**2))# Rayleigh length  (considered as the probe length) # half-width of the weighting function --> FWHM = 2*zr
            # Uncertainty rayleigh length       
            Unc_Rayleigh_length.append( np.sqrt(((ind_focusdist**2)*Unc_wavelength/(np.pi*rad_eff))**2 + ((2*wavelength*ind_focusdist*Unc_focus_distance)/(np.pi*rad_eff**2))**2 + ((2*wavelength*(ind_focusdist**2)*Unc_eff_radius_telescope)/(np.pi*rad_eff**3))**2))
    
        
        # Probe volume:
        #Probe_volume = np.pi*(Qlunc_yaml_inputs['Probe Volume']['Output beam radius']**2)*((4*(focus_distance**2)*Qlunc_yaml_inputs['Components']['Laser']['Wavelength'])/(Telescope_aperture)) # based on Marijn notes
        #VolCil       = np.pi*(Qlunc_yaml_inputs['Probe Volume']['Output beam radius']**2)*fwhm  # calculated based on the fwhm
        # vol_zr       = np.pi*(Qlunc_yaml_inputs['Components']['Telescope']['Output beam radius']**2)*(2*zr) # based on the definition of Rayleigh length in Liqin Jin notes (Focus calibration formula)

        # Lorentzian weighting function:
        # phi = (Qlunc_yaml_inputs['Probe Volume']['Extinction coeficient']/np.pi)*(1/((1**2)+(36.55-focus_distance)**2))
        # phi = (Qlunc_yaml_inputs['Probe Volume']['Extinction coeficient']/np.pi)*(1/((1**2)+(focus_distance)**2))

        # F = (lamb/np.pi)/(a1**2+lamb**2)  # Lorentzian Weighting function 
    
    elif Qlunc_yaml_inputs['Components']['Lidar general inputs']['Type']=="Pulsed":
        # Variables
        pdb.set_trace()
        tau_meas      = Qlunc_yaml_inputs['Components']['Telescope']['Gate length']
        tau           = Qlunc_yaml_inputs['Components']['Telescope']['Pulse shape']
        stdv_tau_meas = Lidar.optics.telescope.stdv_tau_meas
        stdv_tau      = Lidar.optics.telescope.stdv_tau
        
        
        # Definition from "LEOSPHERE pulsed lidar principles" -->  Theory from Banakh and Smalikho 1994: “Estimation of the turbulence energy dissipation rate from the pulsed Doppler lidar data”.
        Rayleigh_length = (cts.c*tau_meas)/(2*math.erf(np.sqrt(np.log(2))*(tau_meas)/(tau)))/2
        
        dR_dtauMeas = (cts.c*2*math.erf(np.sqrt(np.log(2))*tau_meas/tau)-cts.c*tau_meas*2*(2/np.sqrt(np.pi))*np.exp(-np.sqrt(np.log(2))*tau_meas/tau)*np.sqrt(np.log(2))/tau)/((2*math.erf(np.sqrt(np.log(2))*(tau_meas)/(tau)))**2)
        dR_dtau     = (-cts.c*tau_meas*2*(2/np.sqrt(np.pi))*np.exp(-np.sqrt(np.log(2))*(tau_meas/tau))*(-np.sqrt(np.log(2))*tau_meas/(tau**2)))/((2*math.erf(np.sqrt(np.log(2))*tau_meas/tau))**2)
        Unc_Rayleigh_length      = np.sqrt((dR_dtauMeas*stdv_tau_meas)**2+(dR_dtau*stdv_tau)**2)
        
        # focus_distance = random.randrange(1,500,1)
        # # Weighting function calculation
        # WeightingFunction=[]
        # offset = 500
        # focus_distance = 0
        # z=np.linspace(focus_distance-offset,focus_distance+offset,1001)
        # for ind_z in z:        
        #     WeightingFunction.append((1/(tau_meas*cts.c))*(math.erf((4*np.sqrt(np.log(2))*(ind_z-focus_distance)/((cts.c*tau)))+(np.sqrt(np.log(2)))*tau_meas/tau)-math.erf((4*np.sqrt(np.log(2))*(ind_z-focus_distance)/((cts.c*tau)))-(np.sqrt(np.log(2)))*tau_meas/tau)))        
        
        # # find the two crossing points
        # hmx = SA.half_max_x(z,WeightingFunction)
        
        # # print the answer
        # Rayleigh_length = (hmx[1] - hmx[0])/2
        # print("Rayleigh distance1:{:.3f}".format(zr))
        # print("Rayleigh distance2:{:.3f}".format(Rayleigh_length))
        # pdb.set_trace()  
        # print("Zr uncertainty:{:.3f}".format(Unc_zr))
    
    # Saving rayleigh length to a file in ./metadata to be read by matlab
    if os.path.isfile('./metadata/rayleigh_distance.txt'): 
       os.remove('./metadata/rayleigh_distance.txt')
       file=open('./metadata/rayleigh_distance.txt','w')
       file.write(repr(Rayleigh_length))
       file.close()   
       
    else:
       file=open('./metadata/rayleigh_distance.txt','w')
       file.write(repr(Rayleigh_length))
       file.close()      
    
    Final_Output_UQ_ProbeVolume = {'Rayleigh Length':Rayleigh_length,'Rayleigh Length uncertainty':Unc_Rayleigh_length}
    # pdb.set_trace()
    
    
    # Plotting:
    QPlot.plotting(Lidar,Qlunc_yaml_inputs,Final_Output_UQ_ProbeVolume,False,False,Qlunc_yaml_inputs['Flags']['Probe Volume parameters'],False)
    return Final_Output_UQ_ProbeVolume
 
    
