This folder contains the yaml file where we define the charaacteristics and the measuring configuration of our lidar.

Before creating the classes for the different components we will fill in the yaml file with the corresponding values for the components and decide the components and the modules that we want to include in the lidar device for uncertainty calculations. 
Use *Qlunc_inputs.yml* file already filled for a quick start, or fill in the template and rename it  *Qlunc_inputs.yml*. Copy/paste the file in the `Main` folder.

**Warning!** When introducing the component in each module, the name should be the same as in the component instance (e.g. if the name of your module instance is _Module_A_ the name to use in the yaml file should be the same). 

![Uncertainty_WF](https://github.com/SWE-UniStuttgart/Qlunc/blob/main/Pictures_repo_/FlowChartUnc.JPG)
