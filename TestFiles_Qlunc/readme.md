
## Working example to create a lidar digital twin

In the following lines it is shown the `Qlunc`'s workflow and the code structure it is based on; the `WorkingExample` files are a use case example.
Now, imagine we want to create a lidar object made up with one single module. This module will contain one single component with properties Property_1 and Property_2. The steps we have to follow are: 

 1) Fill up yaml file with inputs
 2) Create a component with its propertie(s)
 3) Create a module containing the component(s)
 4) Create a lidar containing the module(s)
 5) Ask for the uncertainties we are interested in by using _dot notation_

### 1. Fill up the inputs yaml file
Before creating the classes for the different components we will fill in the yaml file with the corresponding values for the components and decide the components and the modules that we want to include in the lidar device for uncertainty calculations. Users have a [yaml template for Qlunc inputs](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Main/Template_yaml_inputs_file.yml) in the repository.

The minimum information defining your component might be:
 - Name: Provide an ID to our object
 - Property: As many as the component has (e.g. for the photodetector could be `wavelength`, `load resistor` and  `gain`).
 - Uncertainty function: Function developed by the user decribing the uncertainty of the lidar module.
 
**Warning!** When introducing the component in each module, the name should be the same as in the component instance (e.g. if the name of your module instance is _Module_A_ the name to use in the yaml file should be the same). 

  ```
   YAML file:
    >>## Components:
    >> 
    >>  Component_A:
    >>  
    >>    Name: ComponentA
    >>   
    >>    Property_A: property_1_value 
    >>    
    >>    Property_B: property_2_value 
    >>   
    >>    Uncertainty function: Uncertainty_ComponentA  # Function describing the module uncertainty in _Module_A_ due to their components.
    >>---
    >>## Modules:
    >> 
    >>  Module_A: 
    >>  
    >>    Name: ModuleA
    >>   
    >>    Component: _Component_A_                   # It has to be the same name as the instance (*).
    >>   
    >>    Uncertainty function: Uncertainty_ModuleA  # Function describing the module uncertainty in _Module_A_ due to their components.
    >>---
    >>## Lidar:
    >> 
    >>  Lidar_A:
    >>  
    >>    Name: LidarA
    >>   
    >>    Module: _Module_A_                         # It has to be the same name as the instance (**).
    >>   
    >>    Uncertainty function: Uncertainty_ModuleA  # Function describing the module uncertainty in _Module_A_ due to their components.
```

### 2. Creating the component digital twin
The components are included as python classes, for example a component, _Component_A_, is created by instantiating class _Comp_A_:

- Creating a class for the component _Component_A_:
```
  >> class Comp_A():
  >> 
  >>   def __init__(self, property_1, property_2, unc_func)
  >>   
  >>      self.property_1  = property_1
  >>      
  >>      self.property_2  = property_2
  >>      
  >>      self.uncertainty = unc_func 
``` 
- Then we instantiate class _Comp_A_ to create the object representing the lidar component digital twin:
```
  >> Component_A (*) = Comp_A (name        = C_A,
  >>   
  >>                           property_1  = property_1_value,         # picked from the yaml file
  >>                       
  >>                           property_2  = property_2_value,         # picked from the yaml file
  >>                       
  >>                           uncertainty = Component_A_uncertainty)  # parameter describing uncertainty in _Comp_a_.
```
The uncertainty function is a function either found in literature or developed by the user that discribes the uncertatinty of the component.

### 3. Creating the module digital twin
For the modules we create a class and include the 

- Creating a class for the _Module_A_:
 ``` 
  >> class Mod_A():
  >> 
  >>   def __init__(self, name, component, unc_func)
  >>   
  >>      self.name        = name
  >>      
  >>      self.component   = component   
  >>      
  >>      self.uncertainty = unc_func  
``` 
- Then we instantiate class _Mod_A_ to create the Module object:
```
  >> Module_A (**) = Mod_A (name        = ModuleA, 
  >> 
                            component   = Component_A,                      # Including _Component_A_ in the module. 
                                                                            # It has to be the same as in the instance (*).
                            uncertainty = Module_A_uncertainty_function.py) # Uncertainty describing uncertainty in _Module_A_. Following GUM.                      
```
### 4. Creating the lidar

Then once we have created the module(s), we can made up a lidar object just in the same way:

- Creating a class for the _Lidar_A_ device:
```
  >> ## Lidar:

  >> class lidar():
  >> 
  >>   def __init__(self, name, module, unc_func)
  >>   
  >>      self.name        = name
  >>      
  >>      self.module      = module
  >>             
  >>      self.uncertainty = unc_func  
```  
- Then we instantiate class _lidar_ to create the lidar object:
```
  >> Lidar_A = lidar (name        = LidarA, 
  >> 
  >>                  module      = Module_A,                         # Actually, picked from the yaml file. 
  >>                                                                  # It has to be the same name as the instance (**).                
  >>                  uncertainty = Lidar_A_uncertainty_function.py)  # Uncertainty describing uncertainty in _Lidar_A_. Following GUM.
```
Then, we have created a lidar (python-based) object called _Lidar_A_, made up of one module, _Module_A_, which contains one single component, _Component_A_, with properties _Property_1_ and _Property_2_.

### 5. Asking for uncertainties
The modularity of the code  allows user either to ask for _Photodetector1_ uncertainty (component uncertainty), _Photonics_ uncertainty (module unceratinty) or global lidar uncertainty. using the dot notation we can write:
```
>> Lidar_A.module.component.uncertainty(Lidar_A, AtmosphericScenario,cts,Qlunc_yaml_inputs) for the component uncertainty included in Module
>> Lidar_A.module.uncertainty(Lidar_A, AtmosphericScenario,cts,Qlunc_yaml_inputs) for the Module uncertainty
>> Lidar_A.uncertainty(Lidar_A, AtmosphericScenario,cts,Qlunc_yaml_inputs) for the lidar global uncertainty
```
![Uncertainty_WF](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Pictures_repo_/FlowChartUnc.JPG)
