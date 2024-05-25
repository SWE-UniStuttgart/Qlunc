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
    DataFrame = {}
    # List_Unc_lidar  = []
    print(colored('Processing lidar uncertainties...','magenta', attrs=['bold']))
    
    ### Photoniccs
    if Lidar.photonics != None:
        try: # each try/except evaluates whether the component is included in the module, therefore in the calculations
            DataFrame = Lidar.photonics.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,DataFrame)

        except:
            Photonics_Uncertainty = None
            print(colored('Error in photonics module calculations!','cyan', attrs=['bold']))
    else:
        print(colored('You didn´t include a photonics module in the lidar.','cyan', attrs=['bold']))
    
    #%% Signal processor
    if Lidar.signal_processor != None:   
        # try:
        DataFrame = Lidar.signal_processor.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,DataFrame)
            
        # except:
            # SignalProcessor_Uncertainty = None
            # print(colored('Error in  signal processor module calculations!','cyan', attrs=['bold']))
    else:
        
        DataFrame['Uncertainty ADC'] = {'Stdv Doppler f_peak [Hz]':np.array(0)*np.linspace(1,1,len(Atmospheric_Scenario.temperature)),'Stdv wavelength [m]':0,'Stdv Vlos [m/s]':0}
        print(colored('You didn´t include a signal processor module in the lidar.','cyan', attrs=['bold']))        

    #%% Intrinsic lidar uncertainty:
    DataFrame['Intrinsic Uncertainty [m/s]'] = SA.U_intrinsic(Lidar,Atmospheric_Scenario,DataFrame,Qlunc_yaml_inputs)
    # pdb.set_trace()
    #%% Optics
    if Lidar.optics != None:
        DataFrame = Lidar.optics.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs,DataFrame)     
    else:
        print(colored('You didn´t include an optics module in the lidar.','cyan', attrs=['bold']))
    
    
    #%% Save data
    if Qlunc_yaml_inputs['Flags']['Save data']:
        # pdb.set_trace()
        #Define the name based on the measuring configuration parameters
        filename = "Q_output_P"+"["+ str(Lidar.optics.scanner.cone_angle[0]) + "," + str(Lidar.optics.scanner.azimuth[0])+ "," +str(Lidar.optics.scanner.focus_dist[0])   + "]"
        for ind_loop in range (len( Lidar.optics.scanner.origin)):
            filename += ('_L{}_'.format(ind_loop+1)+str(Lidar.optics.scanner.origin[ind_loop]))
        filename=filename+'_tilt{}'.format(np.round(Atmospheric_Scenario.wind_tilt,2))+'_Vref{}'.format(int(Atmospheric_Scenario.Vref[0]))    
        # Define the path where to store the data
        path = ".\\Qlunc_Output\\"+filename + ".pkl"
        # Store the dictionary 
        if os.path.isfile(path):
            print(colored('Numerical data already exists and has not been replaced. Figures saved. ', 'red',attrs=['bold']))
            
        else:
                    
            # create a binary pickle file 
            f = open(path,"wb")        
            # write the python object (dict) to pickle file
            pickle.dump(DataFrame,f)        
            # close file
            f.close()
            print(colored("The file containing data {} ".format(filename)+"has been saved to 'Qlunc_Output' folder.",'cyan',attrs=['bold']) )   
    else:
        print(colored("No data saved.",'cyan',attrs=['bold']) )   

    ########################################    
    # How to read the data
    # Qlunc_data = pickle.load(open(path,"rb"))
    ########################################
    print(colored('...Lidar uncertainty done.','magenta', attrs=['bold']))
    return DataFrame
