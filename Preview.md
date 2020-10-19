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


# Working example

In this repository is presented a working example of Qlunc in order to facilitate its understanding.

The components are included as python classes, for example a component A is created instanciating class A:

- Creating a class for the component _Component_A_
  >> class Comp_A:
  >>   def __init__(self, property_1, property_2, unc_func)
  >>      self.property_1  = property_1
  >>      self.property_2  = property_2
  >>      self.uncertainty = unc_func
- Then we instantiate class _Comp_A_ tro create the object representing the lidar component digital twin.

  >> Component_A = Comp_A (property_1 = a1,   # Component A
  >>                       property_2 = a2,
  >>                       uncertainty = uncertainty_function) # Define by the user
     
As well for the modules:

- Creating a class for the _Module_A_
  >> class Mod_A:
  >>   def __init__(self, name, Comp_1)
  >>      self.name       = name
  >>      self.Comp_1 = Comp_1

- Then we instantiate class _Module_A_

  >> Module_A = Mod_A (name       = C_A 
                       Comp_1 = Component_1,  
                       Property_2 = a2)
