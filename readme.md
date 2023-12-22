# **Quantification of lidar uncertainties - Qlunc**


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7309008.svg)](https://doi.org/10.5281/zenodo.7309008)


## What is `Qlunc`?
Qlunc is a python-based, open-source software, aligned with the [Wind lidar Ontology](https://github.com/IEA-Wind-Task-32/wind-lidar-ontology) by [IEA TCP Wind Task52 ](https://iea-wind.org/task52/), which can be used to estimate errors in wind velocity and wind direction estimations when probing the wind with lidars. The code has an objected-oriented structure; by using python objects and simulating real lidar components the code puts all together in modules to build up a lidar [digital twin](https://en.wikipedia.org/wiki/Digital_twin). The code is modular and offers the possibility of creating different lidar objects on parallel ( refer to [Tutorial2.ipynb](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Tutorials/Tutorial2.ipynb) ). This enables seamless integration of lidar modules with distinct characteristics, allowing the simulation of various lidar devices.
<p align="center">
  <img src="https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Pictures_repo_/Qlunc_GralStructure.JPG" />
  Figure 1. General structure of Qlunc
</p>
Currently, Qlunc's framework addresses photonics, signal processing techniques and pointing uncertainties. In each module the Guide to the Expression of Uncertainty in Measurement (GUM) is applied to compute propagation of uncertainty. 
Qlunc estimates uncertainties of the line-of-sight wind velocity ($V_{LOS}$), the horizontal wind velocity ($V_{h}$), 3D wind vector ($V_{wind}$) and wind direction (&#934;). It involves a comparative analysis the developed analytical model and Monte Carlo simulations. Additionally, the tool provides the capability to account for correlations among measurement angles uncertainties within a single lidar and across two and/or three lidar devices.

### Creating a lidar device
The user creates the different lidar components by instantiating python classes, including its functional parameters and defining their specific uncertainty functions. Then, each module (also python objects) is "filled" with the corresponding components and their uncertainties are computed following uncertainty propagation method according to the GUM model. Once each component is "ensembled", building up the different modules, the modules are included into the lidar object.

### Creating atmospheric conditions
The user creates also atmospheric scenarios to account for the different atmospheric conditions the lidar has to deal with. Power law  exponent α, temperature and humidity are accepted, either single values or time-dependent variabilities of these inputs, taken from peripherals.

### Qlunc (NEW!) available capabilities

#### Uncertainties in hardware
The flexibility of the code allows users, not only to assess global lidar uncertainty due to signal noise, but also to query uncertainties contributed by noise in specific modules or even single components.
#### 🆕 Estimated uncertainties in $V_{wind}$ and &#934; with information from 3D wind vector 🆕
#### 🆕 Estimated uncertainties in $V_{LOS}$ , $V_{h}$ and &#934; due to errors in pointing accuracy and focus distance 🆕
Considered as a major contributor to uncertainty in lidar estimations, the new Qlunc's add-on uses a combination of analytic and Monte Carlo approaches for estimating the intrinsic lidar uncertainty including:
- Hardware noise
- Speckle noise
- Bias in the sampling frequency
- Bias in the laser wavelength
- Quantisation and FFT-based signal processing
- Uncertainty in the lidar line-of-sight and horizontal wind speeds, as well as the uncertainty in the wind direction, all due to errors in pointing accuracy and focus distance
- Lidar unceratinty correlations assessment

#### Plots
 - Photodetector signal-to-noise ratio and separate contributions due to shot noise, thermal noise, dark current noise and, if needed, trans-impedance amplifier noise.
 - 🆕 Uncertainties in $V_{LOS}$ , $V_{h}$ and $V_{wind}$ with the wind direction 
 - 🆕 Uncertainties in $V_{LOS}$ with focus distance, elevation angle and azimuth angle for a fixed wind direction 
 - 🆕 Uncertainty in &#934; lidar estimation
 - 🆕 Uncertainty in vertical and horizontal measuring planes


<p align="center">
  <img src="https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Pictures_repo_/Unc100PV1.1.gif" />
  Figure 2. Horizontal wind speed and wind direction uncertainties - Lissajous Pattern
</p>

    
## How to use `Qlunc`

:warning: **Please downolad the latest release (V1.0).**

### Create an environment and install dependencies

1) Having [Anaconda](https://docs.anaconda.com) installed is a prerequisite if we want to work in a different environment than `base`, and it is recommended. Then, based on the requirements added to the ``environment.yaml`` file we build the new environment named `Qlunc_Env` by default. 

2) In the Anaconda prompt, go to the directory where you have clone/download Qlunc and type:

```file.
conda env create -f environment.yml 
conda activate <envname>
```

3) Your environment is ready to rumble. You have now a new environment with all the packages needed to run Qlunc.

4) In case you don't want to create a new environment, just install the requirements listed in the `environment.yml` file.

### Download or [clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) the repository to a local directory.

By downloading or cloning the repository user will get several folders within which Qlunc is organized. 
 
First,
1) Create a folder named `Qlunc_Output` in '\Qlunc' main directory. Here the data will be saved. 
2) Copy and paste the file `Qlunc_inputs_Dual.yml`, for a dual lidar solution, or `Qlunc_inputs_Triple.yml`, for a triple lidar solution, from `TestFiles_Qlunc` into `Main` and rename it to `Qlunc_inputs.yml` for a quick start/test. Otherwise, fill in the template in the same folder (`TestFiles_Qlunc`) and rename it to `Qlunc_inputs.yml`. Copy and paste this file into the `Main` folder. 

The content of each folder in the repository is breafly explained here below. Further information can be found in the `readme` in the corresponding folder. 

### Main
`Main` is the core of Qlunc. It contains the scripts to create the classes describing the components, modules, general inputs of the lidar device and atmospheric scenarios, and instantiates the classes to build up the virtual lidar(s).
 - `Qlunc_Classes.py` contains the code which creates the lidar components and modules. Each lidar module/component is assigned to a python class.
 - `Qlunc_Instantiate.py` instantiate the lidar classes taking the values from `Qlunc_inputs.yml`.

### UQ_Functions
 - Contains the functions that compute the uncertainties from different devices, calculting also the uncertainty propagation corresponding to the different modules and the lidar uncertainty as well. Users can define their own functions to calculate specific module uncertainties, and combined/expanded uncertainties as well. 

### Utils
 - Contains scripts meant to do different tasks. Contains funtions which interface directly with Qlunc and are necessary to compute calculations. Also contains `Qlunc_Plotting.py`, a script to automate plots and `Qlunc_ImportModules.py` to import the necessary python packages. 
 - The new functions implemented for this release estimating the uncertainty in $V_{LOS}$ and $V_{h}$ are allocated here.

###  TestFile_Qlunc
 - `Qlunc_inputs.yml`: Human-friendly data file used as input to Qlunc. Copy and paste it into `Main` or
 - `Template_yaml_inputs_file.yml` to create your own use case

### Tutorials
- Contains 3 [Jupyter Notebook-based tutorials](https://github.com/SWE-UniStuttgart/Qlunc/tree/Qlunc-V0.9/Tutorials) with their corresponding yaml files and working examples. These tutorials are meant to serve as a guide to get familiar with Qlunc's routines, and show current capabilities of the framework. Users can find more information about these tutorials in the corresponding `readme` in the folder `Tutorials`.

### Requirements
The `environment.yml` file summarises the python packages needed to run Qlunc 

## How to run `Qlunc`

1) Fill in the `Qlunc_inputs.yml` with the desired values for uncertainty estimation
2) Run `Qlunc_Instantiate.py` to instantiate the lidar classes
3) Type "QluncData = Lidar.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)". A dictionary with relevant parameters about lidar(s) uncertainties and configuration(s) is stored in `QluncData`
4) Alternatively, user can execute Qlunc through a graphical user interface `Qlunc_GUI.py` in `Main` developed to ease the use of Qlunc 

## Author
[Francisco Costa](https://www.ifb.uni-stuttgart.de/en/institute/team/Costa-Garcia/)

## Contributions
Contributions are very welcome!
If you wish to:
- Colaborate: please contact the author
- Report issues or enhance the code: post an [issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/quickstart) or make a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
- Seek for support: please contact the author

## License
Qlunc is licensed under **[SD 3-Clause License](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/LICENSE)**

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Citing and Contact

<div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0003-1318-9677" href="https://orcid.org/0000-0003-1318-9677" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">Francisco Costa García</a></div>

University of Stuttgart - Stuttgart Wind Energy
 
email: costa@ifb.uni-stuttgart.de
 
