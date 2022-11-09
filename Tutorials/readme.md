
:warning: Binder badges might not work properly. If you encounter  any issue, please try cloning/downloading the repository and opening the jupyter notebook locally.

## Tutorial0:

Qlunc's presentation and basics. What's Qlunc and how does it work.

#### Try it yourself:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SWE-UniStuttgart/Qlunc.git/HEAD?filepath=Tutorials%2FTutorial0.ipynb)

## Tutorial1:
This tutorial aims to facilitate the introduction to Qlunc. 
Will go through the code and create a lidar digital twin, with its modules and components. Will ask for uncertainties either lidar general one or component specific uncertainty. We will see some graphical interesting results. Will see how to access design lidar data.

#### Try it yourself:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SWE-UniStuttgart/Qlunc.git/HEAD?filepath=Tutorials%2FTutorial1.ipynb)

## Tutorial2:
Along `Tutorial1` we learn how to virtually create a lidar device, being the input to the uncertainty estimation model. `Tutorial2` is meant to show how to use Qlunc to estimate the line of sight wind velocity ( $V_{LOS}$ ) and the horizontal wind velocity ( $V_{h}$ ) when measuring with a lidar device. By assuming a "best estimate" for the wind velocity (e.g. from a cup anemometer) at certain height above ground, Qlunc estimates the uncertainty of the wind velocity when sampling the wind with a lidar device. The framework takes into account the uncertainty in the elevation and azimuth angles and in the measuring range. The analytical solution based on the Guide to the expression of Uncertainty in Measurements (GUM) is compared with a Montecarlo simulation. Correlations between angles and between different lidars (if created) are also accounted for. 

#### Try it yourself:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SWE-UniStuttgart/Qlunc.git/HEAD?filepath=Tutorials%2FTutorial2.ipynb)
