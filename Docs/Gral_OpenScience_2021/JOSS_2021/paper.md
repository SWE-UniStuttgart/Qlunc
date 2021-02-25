---
title: 'Qlunc: A Python package for quantification of lidar uncertainty'
tags:
  - wind lidar
  - lidar hardware uncertainty
  - OpenScience
  - OpenLidar
authors:
  - name: Francisco Costa García
    orcid: 0000-0003-1318-9677
    affiliation: 1
affiliations:
 - name: University of Stuttgart. Institute for Aircraft Design - SWE
   index: 1
date: 25 February 2021
bibliography: paper.bib
@article{ViConDAR,
  	url = {https://github.com/SWE-UniStuttgart/ViConDAR},
  	Archiveprefix = {},
  	Author = {Pettas, V. and Costa, F. and Kretschmer, M. and Clifton, A. and Rinker, J. and Cheng P.},
  	DOI = {10.2514/6.2020-0993},
  	Journal = {AIAA Scitech 2020 Forum},  	
  	Title = {{A numerical framework for constraining synthetic wind fields with lidar measurements for improved load simulations}},
    License = {GNU General Public License v3.0},
  	Year = 2020
}
---

# Summary

``Qlunc``, for Quantification of lidar uncertainty, is an open-source, freely available
(https://github.com/SWE-UniStuttgart/Qlunc) python-based tool that aims to estimate
the uncertainty of a wind lidar device, including hardware and data processing methods.
Based on the OpenLidar architecture [1], it contains models of the uncertainty contributed
by individual lidar components and modules, that are then combined to estimate the total
uncertainty of the lidar device [@ViConDAR]

The code is meant to be as modular as possible, easily allowing lidar components’ (represented
by python objects) interchangeability and outcomes’ repeatability.
Furthermore, it allows to easily integrate different uncertainty methods or interface
external codes. ``Qlunc`` has an objected-oriented structure taking advantage of python
features; by using python objects and simulating real lidar components, the code puts all
together in modules and, eventually builds up a lidar digital twin.
This, combined with the underlying open-source code attribute, defines an attractive scenario
for sharing knowledge about lidar uncertainties estimation methods. It also encourages
collaborations among lidar field experts aiming to characterize a common lidar architecture
for different types of lidars, to assess lidar data processing methods or even helps to get
a consensus for lidar terminology, giving place to a lidar ontology, which is a developing
project driven by Nikola Vasiljevic and others [2] [3]. 

The source code for ``Qlunc`` has been
archived to Zenodo with the linked DOI: [@zenodo]

# Motivation

Measuring uncertainty means doubt about the validity of the result of a measurement [4]
or, in other words, it represents the dispersion of the values attributed to a measurand.
The importance of knowing uncertainty in measurements lies both, on the quality of the
measurement as on the understanding of the results, and it can have a huge impact on
the veracity of an experiment or measuring set up. In this sense, wind lidar measurement
uncertainties assessment plays a crucial role, since it can determine decision-making
processes and therefore the global performance of a wind facility.

The scope of this project is to create an open, common and collaborative reference numerical
framework to describe unique lidar architectures, characterize lidar uncertainties and provide
a tool for others to contribute within those frameworks. This is so, but following lines of
OpenScience Principles, the underlying main motivation of this project is to create open and
sharable tools and knowledge, to reinforce or promote new or existing links and to foster
collaborations among research institutions and/or industry, within the wind energy community,
but not limited to it. 

# ``Qlunc`` available capabilities

Currently, ``Qlunc`` can perform both, VAD and scanning lidar patterns. For now, it can perform
lidar hardware uncertainties from photonics module, including photodetector (with or without
trans-impedance amplifier) and optical amplifier components, as well as optics module uncertainty
including scanner pointing accuracy distance errors and optical circulator uncertainties. In the
near future, uncertainties regarding other hardware components and data processing methods will
be impemented in the model.

Output plots show different signal noise contributors of the photodetector components and estimates
of scanning points distance uncertainty.

# Usage

## Creating a lidar digital twin

Each component, pertaining to the correspondent module (e.g. photodetector belongs to the photonics
module) is created as a python object and enclosed in other python class, which represents the aforementioned
module. Following this procedure these lidar modules are, in turn, included in the lidar python class, which
gathers all classes corresponding to the different modules a lidar is made of, thus creating the lidar
digital twin. Dot notation methodology is used to ask for lidar component properties.

!['Qlunc basic structure.'](https://github.com/SWE-UniStuttgart/Qlunc/blob/Qlunc-V0.9/Docs/Gral_OpenScience_2021/JOSS_2021/Qlunc_BasicStructure_diagram.png)
*Fig. 1: 'Qlunc basic structure.'*

## Uncertainty estimation model

All components are characterized by their technical parameters and their uncertainty functions,
which are feed to the code via a yaml file. Combined uncertainties throughout components and modules
are computed according to the Guide to the expression of Uncertainty in Measurement [4] ([GUM](https://www.bipm.org/utils/common/documents/jcgm/JCGM_100_2008_E.pdf)) model. 

As mentioned above, the code claims flexibility and aims to foster collaboration, especially among researchers.
To encourage both, flexibility and further collaborations each lidar module has its own uncertainty estimation
function, which includes the components the module is made of. These stand-alone uncertainty estimation
functions are easily exchangeable, just in case users want to use another uncertainty model. 

# Working example and Tutorials: Do it yourself

Included in the Qlunc repository users can find Jupyter Notebooks-based tutorials
(https://github.com/SWE-UniStuttgart/Qlunc/tree/Qlunc-V0.9/Tutorials) on how Qlunc works, providing a tool
to help them get started with the software. Tutorials’ Binder badge is also provided to ease accessibility 
and reproducibility. Users can find more info about these tutorials in the readme file attached to the Qlunc repository.
Apart from the tutorials, the package includes a functional working example. More information about this
working example is given in the readme, included in the Qlunc repository, where the process of creating a
lidar digital twin is treated in depth.

# Acknowledgements

Author want also to thank Andrew Clifton, Nikola Vasiljevic and Ines Würth for their support and valuable suggestions,
feedback and insight.
This project has received funding from the European Union's Horizon 2020 research and innovation programme
under grant agreement No 858358, within the frame of LIKE project.

# References

# Figures
