This folder contains the yaml file where we define the characteristics and the measuring configuration of our lidar.

Before creating the classes for the different components we will fill in the yaml file with the corresponding values for the components and decide the components and the modules that we want to include in the lidar device. 
Use *Qlunc_inputs.yml* file already filled for a quick start, or fill in the template and rename it as *Qlunc_inputs.yml*. Copy/paste the file in the `Main` folder.

:warning:  The name of the instance of each lidar component/module *must* be the same that the one in the yaml file i.e., if you create an Optics module in the yaml file, and want to include a component called 'Telescope',  the instance of this component *must* be called 'Telescope'.

# Example

1) Create the Optics module with a Telescope component, among others:
```
# This is in the yaml file
Modules:
  Optics Module:
    Name: Optics_module
    Scanner: Scanner
    Optical circulator: Optical_circulator     # Has to be the same name as the instance
    Telescope: **Telescope**                   # Has to be the same name as the instance
    Uncertainty function: uopc.sum_unc_optics  # Python module! (You develop a function and save it as a python module)
```    
2) Then, when instantiating the Telescope component, the instance *must* be called **Telescope**:
```
# This is in the pyhthon script
**Telescope** = telescope (name           = Telesc1,
                           stdv_aperture  = 1,       # [mm]
                           stdv_aperture  = 1e-5,    # [mm]                       
                           focal_length   = 200,      # [m]                      
                           (...)
                           unc_func       = Unc_function_Telescope)
```
and the same for the rest of the components and modules.
