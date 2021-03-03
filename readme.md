# **Quantification of lidar uncertainties - Qlunc**

## What is Qlunc?
Qlunc is a software that aims to quantify errors when measuring with a lidar device. The code has an objected-oriented structure; by using python objects and simulating real lidar components the code puts all together in modules to eventually build up a lidar [digital twin](https://en.wikipedia.org/wiki/Digital_twin). The code is meant to be as modular as possible and offers the possibility of creating different lidar objects on parallel (see [Tutorial2.ipynb](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Tutorials/Tutorial2.ipynb)), with different components at the same time. This allows to easily combine different modules with different characteristics simulating different lidar devices.

![Qlunc basic structure image](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Pictures_repo_/Qlunc_GralStructure.JPG)

Currently, the code can calculate uncertainties coming from photonics, including photodetector (with or without trans-impedance amplifier) and optical amplifier uncertainties, as well as optics module uncertainty including scanner pointing accuracy and optical circulator uncertainties. For each module the Guide to the Expression of Uncertainty in Measurement ([GUM](http://www.bipm.org/en/publications/guides/gum.html)) is applied to calculate uncertainty expansion, taking into account that components are considered uncorrelated. 

### Creating a lidar device:

The user creates the different lidar components by instantiating a python class, including its functional parameters and defining the function that is used to obtain the specific component uncertainty. Then, each module (also python objects) is "filled" with the corresponding components and their uncertainties are computed following uncertainty expansion method according to the [GUM](http://www.bipm.org/en/publications/guides/gum.html) model. Once each component is 'ensembled' building up the different modules, the lidar object is created and the modules included. As a result, the desired lidar digital twin is created, the uncertainty of which is computed again by following [GUM](http://www.bipm.org/en/publications/guides/gum.html) suggestions about uncertainty expansion.

### Creating atmospheric conditions
The user creates also atmospheric scenarios to account for the different atmospheric conditions the lidar has to deal with. Atmospheric inputs, basically temperature 
and humidity, either single values or time series coming from peripherals are both accepted.

### Qlunc available capabilities:

#### Uncertainties:
The last step is to ask for the uncertainty we are interested in, either coming from a component, module or lidar object. Indeed, the flexibility of the code allows the 
user not just to assess global lidar uncertainty,  but also to query uncertainties coming from specific modules or even single components.

At this stage the code can calculate errors introduced by the photodetector and the optical amplifier, forming the photonics module; scanner and optical circulator, forming the optic module. Uncertainty expansion method is applied to obtain the lidar uncertainty due to these modules and components.

#### Plots: 
 - Can draw photodetector uncertainties comparison including shot noise, thermal noise, dark current noise and, if needed, trans-impedance amplifier noise.
 - Scanning points and their uncertainty in meters (VAD and Scanning lidar).

## How to use Qlunc:

### Prerequisites

1) Install anaconda and create a new environment. In the Anaconda prompt type:
'''
conda create QluncEnv
conda activate QluncEnv
'''

2) Install pip & git

``conda install pip``

``conda install git``

3) [Download or clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) the repository to a local directory. **Please downolad the latest (V0.91) release.**


4) Go to the directory of the repository and install the requirements in the new environment:

pip install -r requirements.txt

5) Environment is ready to rumble


By downloading the repository you will get several folders within which Qlunc is organized. The most importants to know are:
### Main:
This is the core of Qlunc. Here the user creates the classes describing the components, modules and general inputs of the lidar device and instantiate the classes.
 - `Template_yaml_inputs_file.yml` and `Qlunc_inputs.yml`. The firdt one is a yaml template where user introduces the lidar components values, modules and general lidar features as well as atmospheric scenarios. The second one can be as an example showing how to fill the template.
 - `Qlunc_Classes.py` contains the snippet which creates all the lidar digital twins. Each lidar module/component is assigned to a python class.
 - `Qlunc_Instantiate.py` instantiate the object classes taking the values from `Qlunc_inputs.yml`.
 - `Qlunc_Plotting.py` is a stand alone python module used to plot some results.
### UQ_Functions: 
 - Contains the functions that compute the uncertainties coming from different devices, calculting also the uncertainty propagation corresponding to the different      modules and lidar uncertainty as well. Users can define their own functions to calculate uncertainties and it expansion as well. 
### Utils:
 - Contains scripts meant to do different tasks. Importing packages and some stand alone funtions which don´t interface directly with Qlunc but are necessary to compute calculations.
###  TestFile_Qlunc:
 - A working example is provided to show how the process looks like. In this test case a lidar is builded up with its modules and components, puting all together to set up a lidar device. User can find more information on how to run this test file in the readme.md file dropped in this folder.
### Tutorials:
- Containing 2 [JupyterNotebook-based tutorials](https://github.com/SWE-UniStuttgart/Qlunc/tree/Qlunc-V0.9/Tutorials); `Tutorial1.ipynb` and `Tutorial2.ipynb` with their corresponding yaml files. 
## Requirements
[Anaconda Navigator1.9.12](https://www.anaconda.com/products/individual) has been installed as a graphical user interface. It includes, among others, Python3.7 and spyder4.2.1 version IDE software features, ready to be used. It includes also JupyterLab2.2.6 and JupyterNotebook6.2.0 versions.

The following python libraries should be installed beforehand:

- matplotlib==3.2.1
- numpy==1.18.5 
- pandas==1.2.1
- pyyaml==5.4.1
- scipy==1.6.0
- sympy==1.7.1
- xarray==0.15.1
- notebook
- jupyterlab


## Author:
[Francisco Costa](https://www.ifb.uni-stuttgart.de/en/institute/team/Costa-Garcia/)

## License:
Qlunc is licensed under **[SD 3-Clause License](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/LICENSE)**

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Citing and Contact:

<div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0003-1318-9677" href="https://orcid.org/0000-0003-1318-9677" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">Francisco Costa García</a></div>

University of Stuttgart - Stuttgart Wind Energy
 
email: costa@ifb.uni-stuttgart.de
 
