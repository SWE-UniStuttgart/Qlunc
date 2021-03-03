---
title: '``Qlunc``: Quantification of lidar (hardware) uncertainty'
tags:
  - wind lidar
  - lidar hardware uncertainty
  - OpenScience
  - OpenLidar
authors:
  - name: Francisco Costa
    orcid: 0000-0003-1318-9677
    affiliation: 1
  - name: Andy Clifton
    orcid: 0000-0001-9698-5083
    affiliation: 1
  - name: Nikola Vasiljevic
    orcid: 0000-0002-9381-9693
    affiliation: 2
  - name: Ines Wuerth
    orcid: 0000-0002-1365-0243
    affiliation: 1
affiliations:
 - name: University of Stuttgart. Institute for Aircraft Design - SWE
   index: 1
 - name: Technical University of Denmark. Department of Wind Energy Resource Assessment and Meteorology
   index: 2
date: 25 February 2021
bibliography: paper.bib
---

# Summary

``Qlunc``, for Quantification of lidar uncertainty, is an open-source, freely available
(https://github.com/SWE-UniStuttgart/Qlunc) python-based tool that aims to estimate
the uncertainty of a wind lidar device, including hardware and data processing methods.
Based on the OpenLidar architecture [@OpenLidar], it contains models of the uncertainty contributed
by individual lidar components and modules, that are then combined to estimate the total
uncertainty of the lidar device.

The code is meant to be as modular as possible, easily allowing lidar components’ (represented
by python objects) interchangeability and outcomes’ repeatability (see \autoref{fig:QluncStructure}).
Furthermore, it allows to easily integrate different uncertainty methods or interface
external codes. ``Qlunc`` has an objected-oriented structure taking advantage of python
features; by using python objects and simulating real lidar components, the code puts all
together in modules and, eventually builds up a lidar digital twin.
This, combined with the underlying open-source code attribute, defines an attractive scenario
for sharing knowledge about lidar uncertainties estimation methods. It also encourages
collaborations among the wind lidar community, aiming to characterize a common lidar architecture,
to assess wind lidar data processing methods or even to get a consensus for lidar terminology,
giving place to a lidar ontology, which is a developing project driven by Andy Clifton and Nikola Vasiljevic
[@OntoStack;@sheet2rdf] and it is been developed within the frame of the IEA Wind Task32 initiative.

The source code for ``Qlunc`` has been archived to Zenodo with the linked DOI: [@zenodo]

# Motivation

Measuring uncertainty means doubt about the validity of the result of a measurement [@GUM]
or, in other words, it represents the dispersion of the values attributed to a measurand.
The importance of knowing uncertainty in measurements lies both, on the quality of the
measurement as on the understanding of the results, and it can have a huge impact on
the veracity of an experiment or measuring set up. In this sense, wind lidar measurement
uncertainties assessment plays a crucial role, since it can determine decision-making
processes and therefore the performance of a wind facility.

The scope of this project is to create an open, standardize and collaborative reference numerical
framework to describe unique lidar architectures, characterize lidar uncertainties and provide the
tools for others to contribute within this framework. This is so, but following lines of OpenScience
Principles [@OpenScience], the underlying main motivation of this project is to create open and
sharable tools and knowledge, to foster collaborations among research institutions and/or industry
and to reinforce/promote new/existing links within the wind energy community, but not limited to it. 

# ``Qlunc`` available capabilities

``Qlunc`` can perform any kind of scanning pattern, but since the present investigation is focused on VAD 
(velocity azimuth display) and forward-looking nacelle-mounted measuring modes, the code bends
to the use of those patterns.
For now, it can compute wind lidar hardware uncertainties from photonics module, including photodetector
and optical amplifier components, as well as optics module uncertainty, including scanner pointing
accuracy distance errors and optical circulator uncertainties. In the near future, uncertainties regarding
other hardware components and data processing methods will be impemented in the model.

Output plots show 1) different signal noise contributors of the photodetector components and 2) estimates 
of the distance error between theoretical and actually measured points.

![Qlunc basic structure.\label{fig:QluncStructure}](Qlunc_BasicStructure_diagram.png)

# Usage

The framework has been developed and tested using python3.7.9 64bit. A required programming environment
to use ``Qlunc`` is provided in the repository, in the *readme.md* file, *Requirements* section. For more
information about the system requirements, please consult *requirements.txt* file.
Existing tutorials and working example will help user to navigate along the ``Qlunc`` framework, reproducing
the most important steps. Two of them are explained below in order to facilitate the understanding of the workflow.
In a first step user creates the lidar digital twin. In a later step user introduces a model, by means of a python
module,to calculate the uncertainty of the system.

## Creating a lidar digital twin

Each component, pertaining to the correspondent module (e.g. photodetector belongs to the photonics
module), is created as a python class and enclosed in other python class, which represents the aforementioned
module. Following this procedure these lidar modules are, in turn, included in the lidar python class, which
gathers all classes corresponding to the different modules a lidar is made of, thus creating the lidar
digital twin. Dot notation methodology can be used to ask for lidar component properties and uncetainties.


## Uncertainty estimation model

All components are characterized by their technical parameters and their uncertainty functions,
which are feed to the code via a yaml file. Combined uncertainties throughout components and modules
are computed according to the Guide to the expression of Uncertainty in Measurement [@GUM] (GUM) model. 

It is seen that the code claims flexibility and aims to foster collaboration, especially among researchers.
To encourage both, flexibility and further collaborations each lidar module has its own uncertainty estimation
function, which includes the uncertainty of the components the module is made of. These stand-alone uncertainty
estimation functions are easily exchangeable, just in case users want to use another uncertainty model. 

# Working example and tutorials: do it yourself

Included in the ``Qlunc`` repository users can find 2 Jupyter Notebooks-based tutorials
(https://github.com/SWE-UniStuttgart/Qlunc/tree/Qlunc-V0.9/Tutorials) on how ``Qlunc`` works, helping
them get started with the software. Binder badge is also provided to ease accessibility and reproducibility.
Users can find more information about these tutorials in the readme file attached to the ``Qlunc`` repository.
Apart from the tutorials, the package includes a functional working example. More information about this
working example is given in the readme, included in the *``Qlunc``/TestFilesQlunc* directory, where the process
of creating a lidar digital twin is treated in depth.

# Future development roadmap

Over the next year, we plan to implement further lidar hardware modules in the model and compute their combined uncertainties.
Also, most significant data processing methods, which are expected to be the highest uncertainty contributors, will be
assessed and implemented in the model during the next stage of the project. 

``Qlunc`` is a modular numerical framework and could feasibly combine with existing codes dealing with lidar uncertaintes. 
In this sense, ``Qlunc`` in combination with other existing tools like yaddum [@yaddum] and mocalum [@mocalum] will
help to improve lidar uncertainty estimations, thus increasing lidar measurements reliability. The "openness" of
these group of tools makes it possible to share within the wind energy community and even beyond it.

Another future objective is to align the lidar components/parameters/characteristics labeling process used by ``Qlunc``, to
the controlled vocabulary resulting from [@OntoStack,@sheet2rdf].

All documentation from the project, scientific articles derived from the research period, tutorials and raw code are meant
to be provided throughout a sphinx-based online site, to give users all needed information to dive into the numerical framework
and get used to the ``Qlunc`` routines.

# Acknowledgements

Author wants to thank Andrew Clifton (ORCID iD: 0000-0001-9698-5083), Nikola Vasiljevic (ORCID iD: 0000-0002-9381-9693) and Ines Würth (ORCID iD: 0000-0002-1365-0243) for their support and valuable suggestions, feedback and insight.
This project has received funding from the European Union's Horizon 2020 research and innovation programme
under grant agreement No 858358, within the frame of LIKE project.

# References
