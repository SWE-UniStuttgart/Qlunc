# **Qlunc**
F.Costa
19.10.2020

Qlunc is a softare that aims to quantify errors when measuring with a lidar device. The code has an objected oriented structure; using python objects and simulating 
real lidar components the code puts all together in modules to eventually build up a lidar digital twin. 

The user creates an instance of each lidar component, including its functional parameters and defining the function that is used to obtain the specific components 
uncertainty. Then, each module (also python objects) is "filled" with the corresponding components and its uncertainty is computed following uncertainty expansion 
according GUM. Once each component is 'ensembled', the lidar object is created and the modules included. As a result we obtain the lidara modules digital twins, the desired lidar digital twin, uncertainty of which is computed again by following GUM suggestions about uncertaity expansion.

The user creates also atmospheric scenarios to account for the different atmospheric conditions the lidar has to deal with. Atmospheric inputs, basically temperature 
and humidity, either single values or time series coming from peripherals are both accepted.

The last step is ask for the uncertainty we are interested in, either coming from a component, module or lidar object. Indeed, the flexibility of the code allows the 
user not just to asses lidar uncertainty,  but also to query uncertainties coming from specific modules or even single comonents.

At this stage the code can calculate errors introduced by photodetector and optical amplifier, forming the photonics module; scanner and optical circulator, forming the optic module. Uncertainty expansion method is applied to obtain the lidar uncertainty due to this modules and components.





# Working example

In this repository is presented a working example of Qlunc in order to facilitate its understanding.

The components are included as python classes, for example a component A is created instanciating class A:

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
  >>                       uncertainty = Comp_A_uncertainty_function) # Uncertainty describing uncertainty in _Comp_a_. Defined by the user.
     
As well for the modules:

- Creating a class for the _Module_A_:
  >> class Mod_A:
  >>   def __init__(self, name, Comp_1, unc_func)
  >>      self.name        = name
  >>      self.Comp_1      = Comp_1    # Including _Component_1_ in the module
  >>      self.uncertainty = unc_func  # Uncertainty describing uncertainty in _Mod_a_. Defined by the user.
  
- Then we instantiate class _Mod_A_:

  >> Module_A = Mod_A (name        = M_A, 
                       Comp_1      = Component_1,  
                       uncertainty = Mod_A_uncertainty_function)
