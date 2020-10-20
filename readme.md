# **Quantification of lidar uncertainties - Qlunc**

## What is Qlunc?:
Qlunc is a softare that aims to quantify errors when measuring with a lidar device. The code has an objected oriented structure; by using python objects and simulating real lidar components the code puts all together in modules to eventually build up a lidar digital twin. The code is meant to be as modular as possible and offers the possibility of creating different lidar objects, with different components at the same time. This allows to easyly combine different modules with different characteristics simulating different lidar devices.

![Qlunc basic structure image](https://github.com/PacoCosta/Qlunc/blob/e9c261303cc0e8b7c534d6989524624772b3820e/Pictures_repo_/Capture.PNG)

### Creating a lidar device:

The user creates the different lidar components by instantiating a python class, including its functional parameters and defining the function that is used to obtain the specific component uncertainty. Then, each module (also python objects ) is "filled" with the corresponding components and their uncertainties are computed following uncertainty expansion method according GUM. Once each component is 'ensembled' building up the different modules, the lidar object is created and the modules included. As a result the desired lidar digital twin is created, uncertainty of which is computed again by following GUM suggestions about uncertaity expansion.

### Creating atmospheric conditions
The user creates also atmospheric scenarios to account for the different atmospheric conditions the lidar has to deal with. Atmospheric inputs, basically temperature 
and humidity, either single values or time series coming from peripherals are both accepted.

### Qlunc available capabilities:

#### Uncertainties:
The last step is ask for the uncertainty we are interested in, either coming from a component, module or lidar object. Indeed, the flexibility of the code allows the 
user not just to asses lidar uncertainty,  but also to query uncertainties coming from specific modules or even single components.

At this stage the code can calculate errors introduced by photodetector and optical amplifier, forming the photonics module; scanner and optical circulator, forming the optic module. Uncertainty expansion method is applied to obtain the lidar uncertainty due to this modules and components.

#### Plots: 
 - Can draw photodetector uncertainties comparison including shot noise, thermal noise, dark current noise and, if needed, transimpedance amplifier noise.
 - Scanning points and their uncertainty in meters (only VAD)

### How to use Qlunc:
By downloading the repository you will get several folders within which Qlunc is organized:
 -# Main:
----------------------Put this in the Working example Readme.md-------------------------------------
## Working example to create a lidar digital twin:
We want to create a lidar object maded up with one module. This module wil contain just one component with properties Property_1 and Property_2. The steps we have to follow are: 

 1) Create a component with its propertie(s)
 2) Create a module containing the component(s)
 3) Create a lidar containing the module(s)

In this repository is presented a working example of Qlunc in order to facilitate its understanding.

### Creating the component digital twin:
The components are included as python classes, for example a component, _Component_A_, is created instanciating class _Comp_A_:

- Creating a class for the component _Component_A_:

  >> class Comp_A:
  >>   def __init__(self, property_1, property_2, unc_func)
  >>      self.property_1  = property_1
  >>      self.property_2  = property_2
  >>      self.uncertainty = unc_func 
  
- Then we instantiate class _Comp_A_ tro create the object representing the lidar component digital twin:

  >> Component_A = Comp_A (name       = C_A,
  >>                       property_1 = a1,  
  >>                       property_2 = a2,
  >>                       uncertainty = Comp_A_uncertainty_function)  # Uncertainty describing uncertainty in _Comp_a_. Defined by the user.

The uncertainty function is a function either found in literature or developed by the user that discribes the uncertatinty of the component using its _properties_.

### Creating the module digital twin:
As well, for the modules:

- Creating a class for the _Module_A_:
  
  >> class Mod_A:
  >>   def __init__(self, name, Comp_1, unc_func)
  >>      self.name        = name
  >>      self.component   = Comp_1    
  >>      self.uncertainty = unc_func  
  
- Then we instantiate class _Mod_A_ to create the Module object:

  >> Module_A = Mod_A (name        = M_A, 
                       Comp_1      = Component_1,                # Including _Component_1_ in the module.
                       uncertainty = Mod_A_uncertainty_function) # Uncertainty describing uncertainty in _Mod_a_. Following GUM.

### Creating the lidar:

Then once we have created the module(s), we can made up a lidar object just in the same way:


- Creating a class for the _Lidar_A_:
  >> class Lid_A:
  >>   def __init__(self, name, Mod_1, unc_func)
  >>      self.name        = name
  >>      self.Mod_1       = Mod_1       
  >>      self.uncertainty = unc_func  
  
- Then we instantiate class _Lid_A_ to create the Lidar object:

  >> Lidar_A = Lid_A (name        = M_A, 
                      Mod         = Module_A,                     # Including _Module_1_ in the lidar device.
                      uncertainty = Mod_A_uncertainty_function)   # Uncertainty describing uncertainty in _Lid_a_. Following GUM.

Then, we have created a Lidar object, called _Lidar_A_ made up of one module, _Module_A_, which contains one single component, _Component_A_, with properties _Property_1_ and _Property_2_.
