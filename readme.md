# **Quantification of lidar uncertainties - Qlunc**



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4579842.svg)](https://doi.org/10.5281/zenodo.4579842)



## What is `Qlunc`?
`Qlunc` is a python-based, open, freely available software that aims to estimate errors when measuring with a lidar device. The code has an objected-oriented structure; by using python objects and simulating real lidar components the code puts all together in modules to eventually build up a lidar [digital twin](https://en.wikipedia.org/wiki/Digital_twin). The code is meant to be as modular as possible and offers the possibility of creating different lidar objects on parallel (see [Tutorial2.ipynb](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Tutorials/Tutorial2.ipynb)), with different components at the same time. This allows to easily combine lidar modules with different characteristics, simulating different lidar devices.
<p align="center">
  <img src="https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Pictures_repo_/Qlunc_GralStructure.JPG" />
</p>
Currently, Qlunc's framework can calculate uncertainties coming from photonics, including photodetector and optical amplifier uncertainties, as well as optics module uncertainty including scanner pointing accuracy, probe volume assessment for both continuous wave and pulsed lidars, and optical circulator uncertainties. For each module the Guide to the Expression of Uncertainty in Measurement ([GUM](http://www.bipm.org/en/publications/guides/gum.html)) is applied to calculate uncertainty propagation along the modules, taking into account that the components are considered uncorrelated. 
Qlunc also offers the possibility of estimating the uncertainties of the line-of-sight wind velocity ( $V_{LOS}$ ) and the horizontal wind velocity ( $V_{h}$ ), and make a comparison between the developed analytical model and Montecarlo simulations. The possibility to consider the correlations between lidar measurement angles and also between different devices is available.

### Creating a lidar device
The user creates the different lidar components by instantiating python classes, including its functional parameters and defining the function that is used to obtain the specific component uncertainty. Then, each module (also python objects) is "filled" with the corresponding components and their uncertainties are computed following uncertainty propagation method according to the GUM model. Once each component is 'ensembled', building up the different modules, the lidar object is created and the modules included. As a result, the desired lidar digital twin is created.

### Creating atmospheric conditions
The user creates also atmospheric scenarios to account for the different atmospheric conditions the lidar has to deal with. Atmospheric inputs, basically temperature and humidity are accepted, either single values or time-dependent variabilities of these inputs, taken from peripherals.

### `Qlunc` (NEW!) available capabilities

#### Uncertainties in hardware
The flexibility of the code allows the user not just to assess global lidar uncertainty due to the noise in the signal, but also to query uncertainties coming from the noise added to the signal by specific modules or even single components.

#### 🆕 Estimated Uncertainties in $V_{LOS}$ and $V_{h}$ due to errors in pointing accuracy and focus distance 🆕
Considered as the major contributors to the uncertainty in lidar estimations, the new Qlunc's add-on is capable of estimating the uncertainty in the lidar line-of-sight and horizontal wind speeds due to errors in pointing accuracy and focus distance. It also offers the possibility of accounting for the correlations between the measuring angles and the correlation between two different lidars when mesuring together.

#### Plots
 - Photodetector total signal-to-noise ratio and separate contributions due to shot noise, thermal noise, dark current noise and, if needed, trans-impedance amplifier noise.
 - Optical signal-to-noise ratio from the optical amplifier
 - 🆕 Uncertainties in $V_{LOS}$ and $V_{h}$ with the wind direction 🆕 
 - 🆕 Uncertainties in $V_{LOS}$ and $V_{h}$ with against focus distance, elevation angle and/or azimuth angle for a fixed wind direction 🆕 

## How to use `Qlunc`

:warning: **Please downolad the latest release (V1.0).**

### Create an environment and install dependencies

1) Having [Anaconda](https://docs.anaconda.com) installed is a prerequisite if we want to work in a different environment than `base`, and it is recommended. Then, based on the requirements added to the ``environment.yaml`` file on the repository, where are included the name of the environment and the tools/packages we want to install, we build the new environment. 

2) In the Anaconda prompt, go to the directory where you have clone/download `Qlunc` and type:

```
conda env create -f environment.yml 
conda activate <envname>
```

3) Your environment is ready to rumble. You have now a new environment, called `Qlunc_Env` by default, with all the packages needed to run `Qlunc`.

4) In case you don't want to create a new environment, just install the requirements listed in the *Requirements* section below.

### Download or [clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) the repository to a local directory.

By downloading or cloning the repository you will get several folders within which `Qlunc` is organized. 
 
The most importants to know are:
1) Create a folder named `Projects` in the main directory. Here the created lidar object(s) will be saved. 
2) Copyand paste the file `Qlunc_inputs.yml` in `TestFiles_Qlunc` into `Main` for a quick start/test. Otherwise, fill in the template in the same folder (`TestFiles_Qlunc`) and rename it to `Qlunc_inputs.yml`. Copy and paste this file into the `Main` folder.
3) The content of each folder in the repository is breafly explained here below. Further information can be found in the `readme` in the corresponding folder. 
 
### Main
This is the core of `Qlunc`. Here the user creates the classes describing the components, modules and general inputs of the lidar device and instantiate the classes.
 - `Qlunc_Classes.py` contains the code which creates the lidar digital twin. Each lidar module/component is assigned to a python class.
 - `Qlunc_Instantiate.py` instantiate the lidar classes taking the values from `Qlunc_inputs.yml`.

### UQ_Functions
 - Contains the functions that compute the uncertainties from different devices, calculting also the uncertainty propagation corresponding to the different      modules and the lidar uncertainty as well. Users can define their own functions to calculate specific module uncertainties, and combined/expanded uncertainties as well. 

### Lidar_Projects
When a lidar object is created, data from its characteristics and measuring configuration are stored here.

### Utils
 - Contains scripts meant to do different tasks. Importing packages and stand alone funtions which don´t interface directly with `Qlunc` but are necessary to compute calculations. Also contains `Qlunc_Plotting.py`, a script to automate plots and `Qlunc_ImportModules.py` to import the necessary python packages. 
 - The new functions implemented for this release estimating the uncertainty in $V_{LOS}$ and $V_{h}$ are allocated here.

###  TestFile_Qlunc
 - `Qlunc_inputs.yml`: Human-friendly data file used as input to Qlunc. Copy and paste it into `Main` or
 - `Template_yaml_inputs_file.yml` to create your own use case

### Tutorials
- Contains 3 [Jupyter Notebook-based tutorials](https://github.com/SWE-UniStuttgart/Qlunc/tree/Qlunc-V0.9/Tutorials); `Turial0.ipynb`, `Tutorial1.ipynb` and `Tutorial2.ipynb` with their corresponding yaml files and working examples. These tutorials are meant to serve as a guide for the user to get familiar with the Qlunc's routines, and show the current capabilities of the framework. The tutorials are also available through the Binder service to ease accessibility and reproducibility. Users can find more information about these tutorials in the corresponding `readme` in the folder `Tutorials`.

## Requirements
The following python packages should be installed beforehand and are included in the `environment.yml` file:

- matplotlib==3.2.1
- numpy==1.18.5 
- pandas==1.2.1
- pyyaml==5.4.1
- scipy==1.6.0
- sympy==1.7.1
- xarray==0.15.1
- xarray-extras==0.4.2
- python==3.7.9
- spyder==4.2.1
- netcdf4==1.5.3
- notebook
- jupyterlab
- termcolor==1.1.0

## Author
[Francisco Costa](https://www.ifb.uni-stuttgart.de/en/institute/team/Costa-Garcia/)

## Contributions
Contributions are very welcome!
If you wish to:
- Colaborate: please contact the author
- Report issues or enhance the code: post an [issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/quickstart) or make a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
- Seek for support: please contact the author

## License
`Qlunc` is licensed under **[SD 3-Clause License](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/LICENSE)**

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Citing and Contact

<div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0003-1318-9677" href="https://orcid.org/0000-0003-1318-9677" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">Francisco Costa García</a></div>

University of Stuttgart - Stuttgart Wind Energy
 
email: costa@ifb.uni-stuttgart.de
 
