---
title: 'Qlunc: Quantification of lidar uncertainty'
tags:
  - wind lidar
  - lidar hardware uncertainty
  - OpenScience
  - OpenLidar
  - digital twin
authors:
  - name: Francisco Costa
    orcid: 0000-0003-1318-9677
    affiliation: 1
  - name: Andrew Clifton
    orcid: 0000-0001-9698-5083
    affiliation: 1
  - name: Nikola Vasiljevic
    orcid: 0000-0002-9381-9693
    affiliation: 2
  - name: Ines Würth
    orcid: 0000-0002-1365-0243
    affiliation: 1
affiliations:
 - name: Stuttgart Wind Energy (SWE), Allmandring 5b, 70569 Stuttgart, Germany
   index: 1
 - name: DTU Wind Energy, Frederiksborgvej 399, 4000 Roskilde Denmark 
   index: 2
date: xx March 2021
bibliography: paper.bib
---
# Summary

Wind lidar is a flexible and versatile remote sensing device for wind energy applications [@Hauke] that measures the wind vector remotely using laser light backscattered from aerosols. It is a key tool for wind energy and meteorology. As with any measurement method, it is essential to estimate its uncertainty.
Qlunc, which stands for **Q**uantification of **l**idar **unc**ertainty, is an open-source Python-based tool to create a digital twin of lidar hardware, and to estimate the uncertainty of wind lidar wind speed measurements.

Qlunc contains models of the uncertainty contributed by individual lidar components and modules (represented by Python objects, which in turn represent physical lidar  objects), that then are combined, considering their different natures, to estimate the uncertainties in wind lidar measurements. The modules are based on the OpenLidar architecture [@OpenLidar] and can be easily adapted for particular use cases thanks to the modularity of the code (see \autoref{fig:QluncStructure}). The terminology for the components and modules defined within Qlunc has also been aligned with a community-driven wind lidar ontology, which is in development [@OntoStack;@sheet2rdf]. 

![Qlunc basic structure.\label{fig:QluncStructure}](Qlunc_BasicStructure_diagram.png)
 
The first release is focused on velocity azimuth display (VAD)[@Browning] scans and forward-looking nacelle-mounted measuring modes, which are common wind-energy-industry applications. Besides uncertainty estimations, Qlunc’s functions could be extended for other applications, for example to compare different  wind velocity vector calculation methods. This, combined with the underlying open-source code, defines an attractive scenario for sharing knowledge and fostering collaboration on wind lidars. 

# Statement of Need

Wind lidars are measuring devices, and as for any other measuring systems, their measurements have uncertainties [@Borraccino_2016]. Therefore, as already stated, it is crucial to assess their measurement uncertainty in order to increase confidence in lidar technology.

Measurement uncertainty means doubt about the validity of the result of a measurement [@GUM]. It represents the dispersion of the values attributed to a measurand. The ability to simulate uncertainty through a model such as Qlunc is important for judging measurement data but can also be useful for designing and setting up experiments and optimizing lidar design. Because wind lidar is important for wind energy applications [@Clifton_2018], better models for wind lidar hardware (e.g., Qlunc) and measurement processes (e.g., through MOCALUM [@mocalum] or YADDUM [@yaddum], with which Qlunc can feasibly combine) will directly contribute to the adoption of wind lidar for wind energy applications. 

This project is influenced by fundamental open science principles [@OpenScience]. The scope is to create an open, standardized and collaborative framework to describe both generic and specific lidar architectures, characterize lidar uncertainties, and provide the tools for others to contribute within this framework. 
 
# Future development roadmap

Over the next year, we plan to implement further lidar hardware modules in the model and compute their combined uncertainties. In addition, we will identify main data processing methods and include those that we consider the highest contributors to uncertainty. 

We also plan to further align the terminology used in Qlunc with the IEA Wind Task 32 controlled vocabulary for wind lidar [@task32ControlledVocabulary]. This will make it easier for users to understand what each of the modules and components do, and promotes interoperability.

All documentation from the project, tutorials, and raw code will be published through a website, to enable users to dive into the numerical framework and get used to the Qlunc routines.
We welcome contributions from the wind lidar community.

# Acknowledgements

This work is part of the LIKE ([Lidar Knowledge Europe](https://www.msca-like.eu/)) project. The project LIKE H2020-MSCA-ITN-2019, Grant number 858358 is funded by the European Union.
 
# References
 

