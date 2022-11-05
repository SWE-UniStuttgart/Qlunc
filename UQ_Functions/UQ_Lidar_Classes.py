# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:55:41 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c)

Here we calculate the uncertainty expansion from uncertainties obtained for
each lidar module following GUM (Guide to the expression of Uncertainties in 
Measurement) model. 
 
"""
from Utils.Qlunc_ImportModules import *
import Utils.Qlunc_Help_standAlone as SA

# Calculates the lidar global uncertainty using uncertainty expansion calculation methods:
def sum_unc_lidar(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs):
    """
    Lidar uncertainty estimation. Location: ./UQ_Functions/UQ_Lidar_Classes.py
    
    Parameters
    ----------
    
    * Lidar
        data...
    * Atmospheric_Scenario
        Atmospheric data. Integer or Time series
    * cts
        Physical constants
    * Qlunc_yaml_inputs
        Lidar parameters data
        
    Returns
    -------
    
    list
    
    """ 
    List_Unc_lidar = []
    print(colored('Processing lidar uncertainties...','magenta', attrs=['bold']))
    
    
    ### Photoniccs
    if Lidar.photonics != None:
        try: # each try/except evaluates whether the component is included in the module, therefore in the calculations
            # pdb.set_trace()
            Photonics_Uncertainty,DataFrame = Lidar.photonics.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
            # List_Unc_lidar.append(Photonics_Uncertainty['Uncertainty_Photonics'])
            try:
                List_Unc_lidar.append(DataFrame['Photodetector'])                
            except:
                print(colored('Error appending photodetetor for lidar uncertainty estimations.','cyan', attrs=['bold']))
            try:
                List_Unc_lidar.append(DataFrame['Optical Amplifier'])
            except:
                print(colored('Error appending optical amplifier for lidar uncertainty estimations.','cyan', attrs=['bold']))                  
            # try:
            #     List_Unc_lidar.append(DataFrame['AOM'])
            # except:
            #     print(colored('Error appending AOM for lidar uncertainty estimations.','cyan', attrs=['bold']))                  
        
        except:
            Photonics_Uncertainty = None
            print(colored('Error in photonics module calculations!','cyan', attrs=['bold']))
    else:
        print(colored('You didn´t include a photonics module in the lidar','cyan', attrs=['bold']))
    
    
    ### Optics
    if Lidar.optics != None:
        try:
            Optics_Uncertainty,DataFrame = Lidar.optics.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
            # pdb.set_trace()
            # Optics_Uncertainty           = np.ndarray.tolist(Optics_Uncertainty['Uncertainty_Optics'])*len(Atmospheric_Scenario.temperature)
            # List_Unc_lidar.append(np.array([Optics_Uncertainty]))
            try:
                List_Unc_lidar.append(DataFrame['Optical circulator'])                
            except:
                 print(colored('Error appending optical circulator for lidar uncertainty estimations.','cyan', attrs=['bold']))
            try:
                List_Unc_lidar.append(DataFrame['Telescope'])                
            except:
                 print(colored('No telescope in the photonics module. Telescope is not in lidar uncertainty estimations','cyan', attrs=['bold']))                          
        except:
            Optics_Uncertainty = None
            print(colored('Error in optics module calculations!','cyan', attrs=['bold']))
    else:
        print(colored('You didn´t include an optics module in the lidar','cyan', attrs=['bold']))
    
    
    ### Power
    if Lidar.power != None:        
        try:
            Power_Uncertainty,DataFrame = Lidar.power.Uncertainty(Lidar,Atmospheric_Scenario,cts)
            List_Unc_lidar.append(Power_Uncertainty['Uncertainty_Power']*len(Atmospheric_Scenario.temperature))
        except:
            Power_Uncertainty = None
            print(colored('No power module in calculations!','cyan', attrs=['bold']))
    else:
        print(colored('You didn´t include a power module in  the lidar','cyan', attrs=['bold']))
    
    
    ### Signal processor
    if Lidar.signal_processor != None:   
        # pdb.set_trace()
        try:
            # pdb.set_trace()
            SignalProcessor_Uncertainty,DataFrame = Lidar.signal_processor.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
            # pdb.set_trace()
            List_Unc_lidar.append(SignalProcessor_Uncertainty['Uncertainty_SignalProcessor']*len(Atmospheric_Scenario.temperature))
        
        except:
            SignalProcessor_Uncertainty = None
            print(colored('No signal processor module in calculations!','cyan', attrs=['bold']))
    else:
        print(colored('You didn´t include a signal processor module in  the lidar','cyan', attrs=['bold']))    
    
    
    
    # if Lidar.wfr_model != None:
    #     try:
    #         pdb.set_trace()
    #         WindFieldReconstruction_Uncertainty = Lidar.wfr_model.Uncertainty(Lidar, Atmospheric_Scenario,cts,Qlunc_yaml_inputs,Scanner_Uncertainty)
    #     except:
    #         print(colored('Error in wfr model','cyan', attrs=['bold']))
    
    Uncertainty_Lidar                     = SA.unc_comb(List_Unc_lidar)[0]
    
    Final_Output_Lidar_Uncertainty        = {'Hardware_Lidar_Uncertainty_combination':Uncertainty_Lidar}#,'WFR_Uncertainty':Optics_Uncertainty['Uncertainty_WFR']}    
    
    Lidar.lidar_inputs.dataframe['Lidar'] = Final_Output_Lidar_Uncertainty['Hardware_Lidar_Uncertainty_combination']*np.linspace(1,1,len(Atmospheric_Scenario.temperature))
    
    # pdb.set_trace()
    # Include time in the dataframe:
    # Lidar.lidar_inputs.dataframe['Time']=Atmospheric_Scenario.time
    ########################################################################################################
    # Create Xarray to store data. Link with Mocalum and yaddum  ###########################################
    # READ netcdf FILE.
    # da=xr.open_dataarray('C:/SWE_LOCAL/GIT_Qlunc/Projects/' + 'Gandia.nc')
    # pdb.set_trace()
    # df=SA.to_netcdf(Lidar.lidar_inputs.dataframe,Qlunc_yaml_inputs,Lidar,Atmospheric_Scenario)
    ########################################################################################################
    ########################################################################################################
    # pdb.set_trace()
    
    # Save the dictionary --> Lidar.lidar_inputs.dataframe
    with open('./Projects/'+Lidar.LidarID, 'wb') as f:
        pickle.dump(Lidar.lidar_inputs.dataframe, f)
    # Read the saved dictionary
    with open('./Projects/'+Lidar.LidarID, 'rb') as f:
        loaded_dict = pickle.load(f)
        
        
        
    print(colored('...Lidar uncertainty done. Lidar saved in folder "Projects"','magenta', attrs=['bold']))
    return Final_Output_Lidar_Uncertainty,Lidar.lidar_inputs.dataframe