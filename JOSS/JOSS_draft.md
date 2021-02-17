---
title: "Quantification of lidar (hardware) uncertainties"
tags:
  - wind lidar
  - hardware uncertainty
  - photonics module
  - optics module
  - OpenLidar
  - OpenScience
authors:
  - name: Costa, Francisco  
    orcid: https://orcid.org/0000-0003-1318-9677   
    affiliation: 1
affiliations:
  - name: Institute of Aircraft Design and Manufacture - Stuttgart Wind Energy, University of Stuttgart
    index: 1 
date: 17 February 2021
output: html_document
bibliography: paper.bib
---

# <div align="center"> **Q**uantification of **l**idar (hardware) **unc**ertainties
# <div align="center"> **Qlunc**
  
## Introduction


Measuring uncertainty means doubt about the validity of the [@Chang:2008:BDS:1365815.1365816] result of a measurement see @fidgit [@fidgit] , @fenner2012a [@fenner2012a] or [@smith04], [@doe99], in other words, it represents the dispersion of the values attributed to a measurand. The importance of knowing uncertainty in measurements lies on both, the quality of the measurement and understanding of the results, and it can have a huge impact on the veracity of an experiment or measuring set up, hence in decision-making processes based on the experiment outcomes. In the wind energy community wind lidar devices have been widely used to characterize the wind and research for most suitable sites, with best wind conditions, but also in wind forecasting, wind turbine power performance, loads assessment and lidar-assisted turbine control [2]. Building a wind farm entails a huge effort and investment, so knowing beforehand site and wind key parameters is important to minimize risks and optimize energy acquisition. In this sense, lidar measurement uncertainties assessment plays a crucial role, since it can determine decision-making processes and therefore the global performance of a wind facility.

## What’s Qlunc 


Lidar is a remote sensing measuring device and, to increase confidence in its measurements, the uncertainty of the measuring data must be assessed. This project develops and implements an open-source, freely available uncertainty model that allows us to assess lidar measurement uncertainties for profiling lidar and forward-looking nacelle-mounted lidar  before a lidar is built.
Inspired by the OpenLidar architecture [3], this model is a python-based tool called `Qlunc` for “Quantification of Lidar UNCertainties”, that aims to estimate the uncertainty of a wind lidar device, including hardware and data processing methods. It contains models of the uncertainty contributed by individual lidar components that are then combined to estimate the total uncertainty of the lidar device.
The code has an objected-oriented structure taking advantage of python features; by using python objects and simulating real lidar components, the code puts all together in modules to eventually build up a lidar digital twin. Qlunc is meant to be as modular as possible and offers to the user the possibility of creating different lidar objects on parallel, with different components, simultaneously. This allows to easily combine different modules with different characteristics simulating different lidar devices and compare them against each other. Furthermore, it allows to easily integrate different uncertainty methods or interface external codes.
Each component, pertaining to correspondent module (e.g. photodetector belongs to the photonics module) is created as a python object and enclosed in other python class, which represents the aforementioned modules. Following this procedure these modules are, in turn, included in the lidar python class, which gathers all classes corresponding to the different modules a lidar is made of, thus creating the lidar digital 

# References
