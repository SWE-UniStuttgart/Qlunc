
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
