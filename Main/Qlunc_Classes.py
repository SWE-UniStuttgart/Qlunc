# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:57:05 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c) 

###############################################################################
###################### __ Classes __ ##########################################
  
Creating components, modules and lidar classes. 
The command class is used to create digital "objects", representing the lidar 
components, and store necessary information to calculate uncertainties coming 
from hardware and data processing methods. 

`unc_func` contains the functions calculating the uncertainty for each
component, module and/or lidar system.
It is defined as a python module, so users can define their own uncertainty 
functions and implement them easily just pointing towards a different python 
module. 

Flags, general constants and atmospheric scenarios used along the calculations 
are also treated as classes. User should instantiate all, but the general 
constants.


How is the code working?

If user wants to implement another component/module they should create a new class. 

For example, we might want build up a lidar and user wants to include a 'power'
module which, in turn contains an UPS (uninterruptible power supply) with 
certain parameter values, e.g. output power and output voltage. Let's call them
Pout and Vout. 

1) Create the `classes`:
    
So we create the class `UPS`:
    
def UPS():
    def __init__(self,PowerOutput,VoltageOutput,Uncertainty)
        
        self.Pout     = PowerOutput
        self.Vout     = VoltageOutput
        self.unc_func = Uncertainty

And also the class `power`, including the UPS component:

def power():
    def __init__UPS (self, U_PowerSupply,Uncertainty)
        self.UPS     = U_PowerSupply
        self.unc_fun = Uncertainty 

Then, the lidar is created (in this simple example) by including the module in 
a lidar class:

class lidar():
      def __init__ (self, Power_Module,Uncertainty)
          self.PowerMod = Power_Module
          self.unc_fun  = Uncertainty

Setting an atmospheric scenario is also needed:
class atmos_sc():
    def __init__ (self, Temperature, Humidity):
        self.T = Temperature
        self.H = Humidity
    
2) Instantiate the classes

Instantiating the component class:
UPS_1 = UPS(Pout     = 500, % in watts
            Vout     = 24,  % in volts
            unc_func = function calculating uncertainties)

Instantiating the module class to create the `Power` object:
Power_1 = power(Power_Module = UPS_1,
                unc_func      = function calculating uncertainties) 
        

Instantiating the lidar class:
Lidar_1 = lidar(Power_Module = Power_1,
                unc_func     = function calculating uncertainties)

So we have created a lidar digital twin with its first module, the `power` 
module, which in turn contains a component, the uninterruptible power supply.

--> How to ask for uncertainties:Lidar.[module].[component].Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)

Qlunc uses GUM (Guide to the expression of Uncertainties in Measurement) 
model to calculate uncertainty expansion.  
  
"""
from Utils import Qlunc_Help_standAlone as SA

#%% Constants:
class cts():
    k = 1.38064852e-23 # Boltzman constant:[m^2 kg s^-2 K^-1]
    h = 6.62607004e-34 # Plank constant [m^2 kg s^-1]
    e = 1.60217662e-19 # electron charge [C]
    c = 2.99792e8 #speed of light [m s^-1]


#%% LIDAR COMPONENTS
# Component Classes:
class photodetector():
    def __init__(self,name,Photo_BandWidth,Load_Resistor,Photo_efficiency,Dark_Current,Photo_SignalP,Active_Surf,Power_interval,Gain_TIA,V_Noise_TIA,unc_func):
                 self.PhotodetectorID  = name 
                 self.BandWidth        = Photo_BandWidth 
                 self.Load_Resistor    = Load_Resistor 
                 self.Efficiency       = Photo_efficiency
                 self.DarkCurrent      = Dark_Current
                 self.SignalPower      = Photo_SignalP
                 self.Active_Surf      = Active_Surf
                 self.Power_interval   = Power_interval
                 self.Gain_TIA         = Gain_TIA
                 self.V_Noise_TIA      = V_Noise_TIA
                 self.Uncertainty      = unc_func
                 print('Created new photodetector: {}'.format(self.PhotodetectorID))
        


class analog2digital_converter():
    def __init__(self,name,nbits,vref,vground,q_error,ADC_bandwidth,fs,u_fs,u_speckle,unc_func):
                 self.ADCID = name
                 self.nbits =nbits
                 self.vref = vref
                 self.vground = vground
                 self.fs    =fs
                 self.u_fs =u_fs
                 self.u_speckle = u_speckle
                 self.q_error = q_error
                 self.BandWidth = ADC_bandwidth
                 self.Uncertainty = unc_func
                 print('Created new ADC: {}'.format(self.ADCID))
                 




class scanner():
    def __init__(self,name,Href,N_MC,pattern,lissajous_param,vert_plane,hor_plane,origin,focus_dist,cone_angle,azimuth,stdv_location,stdv_focus_dist,stdv_cone_angle,stdv_azimuth,stdv_Estimation,correlations,unc_func):
                 self.ScannerID       = name
                 self.Href            = Href
                 self.N_MC            = N_MC
                 self.origin          = origin
                 self.pattern         = pattern
                 self.lissajous_param = lissajous_param
                 self.vert_plane      = vert_plane
                 self.hor_plane       = hor_plane
                 self.focus_dist      = focus_dist
                 self.cone_angle      = cone_angle
                 self.azimuth         = azimuth
                 self.stdv_location   = stdv_location
                 self.stdv_focus_dist = stdv_focus_dist
                 self.stdv_cone_angle = stdv_cone_angle
                 self.stdv_azimuth    = stdv_azimuth 
                 self.stdv_Estimation = stdv_Estimation
                 self.correlations    = correlations
                 self.Uncertainty     = unc_func      
                 print('Created new scanner: {}'.format(self.ScannerID))




#%% LIDAR MODULES
                 
# Modules classes:
class photonics():
    def __init__(self,name,photodetector,unc_func):
                 self.PhotonicModuleID        = name
                 self.photodetector           = photodetector
                 self.Uncertainty             = unc_func 
                 print('Created new photonic module: {}'.format(self.PhotonicModuleID))



class optics():
    def __init__(self,name,scanner,unc_func):
                 self.OpticsModuleID     = name
                 self.scanner            = scanner
                 self.Uncertainty        = unc_func 
                 print('Created new optic module: {}'.format(self.OpticsModuleID))

class signal_processor():
    def __init__(self,name,analog2digital_converter): #f_analyser
                 self.SignalProcessorModuleID = name
                 self.analog2digital_converter = analog2digital_converter
                 print('Created new signal processor module: {}'.format(self.SignalProcessorModuleID))


#%% Atmosphere object:
class atmosphere():
    def __init__(self,name,temperature,Hg,PL_exp,wind_direction, wind_tilt, Vref):
                 self.AtmosphereID   = name
                 self.temperature    = temperature
                 self.PL_exp         = PL_exp
                 self.Vref           = Vref
                 self.wind_direction = wind_direction
                 self.Hg             = Hg
                 self.wind_tilt     = wind_tilt                
                 print('Created new atmosphere: {}'.format(self.AtmosphereID))


#%% Creating lidar general data class:
class lidar_gral_inp():
    def __init__(self,name,ltype,yaw_error,pitch_error,roll_error,dataframe):
                 self.Gral_InputsID   = name
                 self.LidarType       = ltype
                 self.yaw_error_dep   = yaw_error   # yaw error angle when deploying the lidar device in the grounf or in the nacelle
                 self.pitch_error_dep = pitch_error # pitch error angle when deploying the lidar device in the grounf or in the nacelle
                 self.roll_error_dep  = roll_error  # roll error angle when deploying the lidar device in the grounf or in the nacelle
                 self.dataframe       = dataframe   # Final dataframe
                 print('Created new lidar general inputs: {}'.format(self.Gral_InputsID))

#%% Lidar class:
class lidar():
    def __init__(self,name,photonics,optics,signal_processor,lidar_inputs,unc_func):
                 self.LidarID          = name
                 self.photonics        = photonics
                 self.optics           = optics
                 self.signal_processor = signal_processor
                 self.lidar_inputs     = lidar_inputs
                 self.Uncertainty      = unc_func
                 print('Created new lidar device: {}'.format(self.LidarID))

#%% 1) Creating the class to store coordinates
class lidar_coor():
    def __init__(self, x,y,z,x_Lidar,y_Lidar,z_Lidar):
        self.x_Lidar=x_Lidar
        self.y_Lidar=y_Lidar
        self.z_Lidar=z_Lidar
        self.x=x
        self.y=y
        self.z=z
    @classmethod
    def vector_pos(cls,x,y,z,x_Lidar,y_Lidar,z_Lidar):
        fx=(x-x_Lidar)
        fy=(y-y_Lidar)
        fz=(z-z_Lidar)
        return(cls,fx,fy,fz)
    @classmethod
    def Cart2Sph (cls, x_vector_pos,y_vector_pos,z_vector_pos):
        rho1,theta1,psi1 =SA.cart2sph(x_vector_pos,y_vector_pos,z_vector_pos)
        return (cls,rho1,theta1,psi1)   