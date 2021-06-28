# **Quantification of lidar uncertainties - Qlunc**



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4579842.svg)](https://doi.org/10.5281/zenodo.4579842)



## What is `Qlunc`?
`Qlunc` is a python-based, open, freely available software that aims to quantify errors when measuring with a lidar device. The code has an objected-oriented structure; by using python objects and simulating real lidar components the code puts all together in modules to eventually build up a lidar [digital twin](https://en.wikipedia.org/wiki/Digital_twin). The code is meant to be as modular as possible and offers the possibility of creating different lidar objects on parallel (see [Tutorial2.ipynb](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Tutorials/Tutorial2.ipynb)), with different components at the same time. This allows to easily combine different modules with different characteristics simulating different lidar devices.

![Qlunc basic structure image](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Pictures_repo_/Qlunc_GralStructure.JPG)

Currently, the code can calculate uncertainties coming from photonics, including photodetector (with or without trans-impedance amplifier) and optical amplifier uncertainties, as well as optics module uncertainty including scanner pointing accuracy and optical circulator uncertainties. For each module the Guide to the Expression of Uncertainty in Measurement ([GUM](http://www.bipm.org/en/publications/guides/gum.html)) is applied to calculate uncertainty expansion, taking into account that components are considered uncorrelated. 

### Creating a lidar device
The user creates the different lidar components by instantiating a python `class`, including its functional parameters and defining the function that is used to obtain the specific component uncertainty. Then, each module (also python objects) is "filled" with the corresponding components and their uncertainties are computed following uncertainty expansion method according to the GUM model. Once each component is 'ensembled' building up the different modules, the lidar object is created and the modules included. As a result, the desired lidar digital twin is created, the uncertainty of which is computed again by following [GUM](http://www.bipm.org/en/publications/guides/gum.html) suggestions about uncertainty expansion.

### Creating atmospheric conditions
The user creates also atmospheric scenarios to account for the different atmospheric conditions the lidar has to deal with. Atmospheric inputs, basically temperature 
and humidity, either single values or time series coming from peripherals are both accepted.

### `Qlunc` available capabilities

#### Uncertainties
The next step is to ask for the uncertainty we are interested in, either coming from a component, module or lidar object. Indeed, the flexibility of the code allows the user not just to assess global lidar uncertainty,  but also to query uncertainties coming from specific modules or even single components.

#### Plots
 - Can draw photodetector uncertainties comparison including shot noise, thermal noise, dark current noise and, if needed, trans-impedance amplifier noise.
 - Scanning points and their uncertainty in meters (VAD and Scanning lidar).

## How to use `Qlunc`

:warning: **Please downolad the latest release (V0.91).**

### Create an environment and install dependencies

1) Having [Anaconda](https://docs.anaconda.com) installed is a prerequisite if we want to work in a different environment than `base`, and it is recommended. Then, based on the requirements added to the ``environment.yaml`` file on the repository, where are included the name of the environment and the tools/packages we want to install, we build the new environment. 

2) In the Anaconda prompt, go to the directory where you have clone/download `Qlunc` and type:

```
conda env create -f environment.yml 
conda activate <envname>
```

3) Your environment is ready to rumble. You have now a new environment, called `Qlunc_Env` by default, with all the packages needed to run `Qlunc`.

4) In case you don't want to create a new environment, just install the requirements listed in the *Requirements* section below.

### Download or [clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) the repository to a local directory

By downloading or cloning the repository you will get several folders within which `Qlunc` is organized. The most importants to know are:

### Main
This is the core of `Qlunc`. Here the user creates the classes describing the components, modules and general inputs of the lidar device and instantiate the classes.
 - `Template_yaml_inputs_file.yml` and `Qlunc_inputs.yml`. The former is a yaml template where user introduces the lidar components values, modules and general lidar features as well as atmospheric scenarios; the latter can be taken as an example showing how to fill in the template.
 - `Qlunc_Classes.py` contains the code which _creates_ all the lidar digital twins. Each lidar module/component is assigned to a python `class`.
 - `Qlunc_Instantiate.py` instantiate the lidar classes taking the values from `Qlunc_inputs.yml`.
### UQ_Functions
 - Contains the functions that compute the uncertainties coming from different devices, calculting also the uncertainty propagation corresponding to the different      modules and lidar uncertainty as well. Users can define their own functions to calculate specific module uncertainties, and combined/expanded uncertainties as well. 
### Utils
 - Contains scripts meant to do different tasks. Importing packages and some stand alone funtions which don´t interface directly with `Qlunc` but are necessary to compute calculations. Also contains a `Qlunc_Plotting.py` script to automate plots and `Scanning_patterns.py` to introduce pre-defined scanning patterns.
###  TestFile_Qlunc
 - A working example is provided to show how the process looks like. In this test case, a lidar is built up with its modules and components, puting all together to set up a lidar device. User can find more information on how to run this test file in the `readme.md` file dropped in this folder.
### Tutorials
- Containing 2 [Jupyter Notebook-based tutorials](https://github.com/SWE-UniStuttgart/Qlunc/tree/Qlunc-V0.9/Tutorials); `Tutorial1.ipynb` and `Tutorial2.ipynb` with their corresponding yaml files. The tutorials are also available through the Binder service to ease accessibility and reproducibility. Users can find more information about these tutorials in the corresponding `readme.md` file dropped in this folder.
## Requirements
The following python libraries and tools should be installed beforehand and are included in the `environment.yml` file:

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
- netcdf4
- notebook
- jupyterlab

## Author
[Francisco Costa](https://www.ifb.uni-stuttgart.de/en/institute/team/Costa-Garcia/)

## Contributions:
Contributions are very welcome!
If you are wishing to:
- Colaborate: please contact the author
- Report issues: follow [Github guidelines to report code issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/quickstart) or make a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
- Seek for support: please contact the author
## License
`Qlunc` is licensed under **[SD 3-Clause License](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/LICENSE)**

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Citing and Contact

<div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0003-1318-9677" href="https://orcid.org/0000-0003-1318-9677" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">Francisco Costa García</a></div>

University of Stuttgart - Stuttgart Wind Energy
 
email: costa@ifb.uni-stuttgart.de
 
