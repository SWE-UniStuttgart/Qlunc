# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:58:24 2020

@author: fcosta
"""

#import LiUQ_inputs

from ImportModules import *

def UQ_Telescope(inputs):
#    toreturn={}
    if inputs.atm_inp.TimeSeries==True:
        UQ_telescope=[inputs.atm_inp.Atmospheric_inputs['temperature'][i]*0.5+
                          inputs.atm_inp.Atmospheric_inputs['humidity'][i]*0.1+
                          inputs.optics_inp.Telescope_uncertainty_inputs['curvature_lens'][0]*0.1+
                          inputs.optics_inp.Telescope_uncertainty_inputs['aberration'][0]+
                          inputs.optics_inp.Telescope_uncertainty_inputs['OtherChanges_tele'][0]\
                          for i in range(len(inputs.atm_inp.Atmospheric_inputs['temperature']))]
    else:       
        UQ_telescope=[(temp*0.5+hum*0.1+curvature_lens*0.1+aberration+o_c_tele) \
                      for temp           in inputs.atm_inp.Atmospheric_inputs['temperature']\
                      for hum            in inputs.atm_inp.Atmospheric_inputs['humidity']\
                      for curvature_lens in inputs.optics_inp.Telescope_uncertainty_inputs['curvature_lens'] \
                      for aberration     in inputs.optics_inp.Telescope_uncertainty_inputs['aberration'] \
                      for o_c_tele       in inputs.optics_inp.Telescope_uncertainty_inputs['OtherChanges_tele']]
    Telescope_Losses =inputs.optics_inp.Telescope_uncertainty_inputs['losses']
    UQ_telescope=[round(UQ_telescope[i_dec],3) for i_dec in range(len(UQ_telescope))]
#    toreturn['telescope_atm_unc']=UQ_telescope
#    toreturn['telescope_losses']=Telescope_Losses
    return UQ_telescope

def Losses_Telescope(losses):
    return losses