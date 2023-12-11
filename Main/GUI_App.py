# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:40:10 2023

@author: fcosta
"""
import ruamel.yaml
from tkinter import *
from tkinter.filedialog import askopenfilename
import yaml
import pdb
import sys,inspect,os
import numpy as np 
import scipy.interpolate as itp 
import pandas as pd 
import numbers
import pdb
from scipy.optimize import curve_fit
import itertools
import functools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functools import reduce
from operator import getitem
import time
import yaml
import pylab
import math
# import xarray as xr
# import netCDF4 as nc    
import csv
from termcolor import colored, cprint 
import random
import matplotlib
import scipy as sc
from scipy.stats import norm
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal
import pickle
# from celluloid import Camera
from matplotlib import animation
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import Grid
import matplotlib.gridspec as gridspec
os.chdir('../')
# pdb.set_trace()
#%%
root=Tk()

root.title("Qlunc")
root.geometry('550x430')

# RunQlunc button
def runQlunc():
    # os.system('opencv_video.py')
    
    runfile('./Main/Qlunc_Instantiate.py')
    # os.system('C:/SWE_LOCAL/GIT_Qlunc/Main/Qlunc_Instantiate.py')
    B=Lidar.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)

def button_select_input_file():
    text_file = askopenfilename(initialdir="../Main/", title="Select a file", filetypes=( ('Yaml file',"*.yml"),("Text files",'*.txt'), ('All files','*.*'))) # show an "Open" dialog box and return the path to the selected file

    text_file=open(text_file,'r')
    stugg=text_file.read()
    mytest.insert(END,stugg)
    text_file.close()


def button_save_txt():
    text_file = askopenfilename(initialdir="../Main/", title="Select a file", filetypes=( ('Yaml file',"*.yml"),("Text files",'*.txt'), ('All files','*.*'))) # show an "Open" dialog box and return the path to the selected file

    text_file=open('./Main/Qlunc_inputs.yml','w')
    text_file.write(mytest.get(1.0,END))    


mytest=Text(root,width=150,height=40)
mytest.pack(pady=20)
button_quit = Button(root,text="Exit Qlunc", bg="black", fg="white", command=root.destroy)
button_quit.pack()

btn_runQlunc                 = Button(root, text="Run Qlunc", bg="black", fg="white",command=runQlunc)
btn_runQlunc.pack()

btn_select_input_file = Button(root, text="Select input file", bg="black", fg="white",command=button_select_input_file)
btn_select_input_file.pack(pady=20)


btn_select_input_file = Button(root, text="Save input file", bg="black", fg="white",command=button_save_txt)
btn_select_input_file.pack(pady=20)


root.mainloop()

#%%


















#######################################################################################################################################################################

# raiz = Tk()
# raiz.title('Qlunc')
# miFrame=Frame(raiz,width=500,height=500)



# #Inputs:

# AmplifierVar=IntVar()
# TeleVar=IntVar()
# PhotoVar=IntVar()
# LosVar=IntVar()
# FilterVar=IntVar()
# temperatureVar=DoubleVar()
# humidityVar=DoubleVar()
# o_c_ampVar=DoubleVar()
# noise_ampVar=DoubleVar()
# noise_photoVar=DoubleVar()
# o_c_photoVar=DoubleVar()
# curve_lensVar=DoubleVar()
# o_c_teleVar=DoubleVar()
# aberrationVar=DoubleVar()
# labConditionsVar=IntVar()

# #Creating subFrames:
# subframe_mod=LabelFrame(raiz,text='Modules:',padx=50,pady=50)# subframe for modules
# subframe_mod.grid(row=0,column=0)
# subframe_DP=LabelFrame(raiz,text='Data Processing Methods:',padx=50,pady=50)#subframe for data processing
# subframe_DP.grid(row=1,column=0)
# subframe_AC=LabelFrame(raiz,text='Atmospheric conditions:',padx=50,pady=50)#subframe for atmospheric effects
# subframe_AC.grid(row=2,column=0)

# #%% functions of the button
# def ModAndDP(): #add modules and methods to the list to pass it as variable to the core code in order to know which hardware modules and DP methods want to assess
#     global modules,DP
#     modules=''
#     DP=''
#     if (AmplifierVar.get()==1):
#         modules+=' amplifier'
#     if (TeleVar.get()==1):
#         modules+=' telescope'
#     if (PhotoVar.get()==1):
#         modules+=' Photodetector'
    
#     if (LosVar.get()==1):
#         DP+=' Los'
#     if (FilterVar.get()==1):
#         DP+=' Filter'
# def LabCond():# when vary the atmospheric conditions checkbutton change values of atmospheric parameters
#     if labConditionsVar.get()==1:
#         set_text(temperatureE,'25°C')
#         set_text(humidityE,'15%')

#     elif labConditionsVar.get()==0:
#         reset_text(temperatureE,'0.0')
#         reset_text(humidityE,'0.0')

# def set_text(var,text):# function to vary text and state of entries and also give the code the lab values temperature=25°c and humidity=15% in the atmospheric cond frame
#     var.delete(0,'end')
#     var.insert(0,text)       
#     var.config(state='disable')
# def reset_text(var,text): # enable inputs for atmospheric cond by the user in the atmospheric cond frame
#     var.config(state='normal')
#     var.delete(0,'end')
#     var.insert(0,text)
# #%% Modules and processes we want to include in the assessment:

# # Modules: Adding checkbuttons for each module
# Amp_CheckButton=Checkbutton(subframe_mod,text='Amplifier',variable=AmplifierVar,onvalue=1,offvalue=0,command= ModAndDP)
# Amp_CheckButton.grid(row=0,column=0,sticky='w')
# Tel_CheckButton=Checkbutton(subframe_mod,text='Telescope',variable=TeleVar,onvalue=1,offvalue=0,command= ModAndDP)
# Tel_CheckButton.grid(row=1,column=0,sticky='w')
# Photo_CheckButton=Checkbutton(subframe_mod,text='Photodetector',variable=PhotoVar,onvalue=1,offvalue=0,command= ModAndDP)
# Photo_CheckButton.grid(row=2,column=0,sticky='w')

# # Data Processing methods: adding methods for each data processing method

# LOS_CheckButton=Checkbutton(subframe_DP,text='Line of sight',variable=LosVar,onvalue=1,offvalue=0,command= ModAndDP)
# LOS_CheckButton.grid(row=0,column=0,sticky='w')
# Filter_CheckButton=Checkbutton(subframe_DP,text='Filtering',variable=FilterVar,onvalue=1,offvalue=0,command= ModAndDP)
# Filter_CheckButton.grid(row=1,column=0,sticky='w')

# #%% Atmospheric values
# AC_CheckButton=Checkbutton(subframe_AC,text='Lab conditions',variable=labConditionsVar,onvalue=1,offvalue=0,command=LabCond)
# AC_CheckButton.grid(row=0,column=0,sticky='w')

# temperatureE=Entry(subframe_AC,textvariable=temperatureVar)
# temperatureE.grid(row=1,column=1)
# temperatureLabel=Label(subframe_AC,text='Temperature:')
# temperatureLabel.grid(row=1,column=0,sticky='w')

# humidityE=Entry(subframe_AC,textvariable=humidityVar)
# humidityE.grid(row=2,column=1)
# humidityLabel=Label(subframe_AC,text='Humidity:')
# humidityLabel.grid(row=2,column=0,sticky='w')

# #%% Amplifier:

# noise_ampE=Entry(miFrame,textvariable=noise_ampVar)
# noise_ampE.grid(row=4,column=1)
# noise_ampLabel=Label(miFrame,text='Noise Amplifier')
# noise_ampLabel.grid(row=4,column=0,sticky='w')

# o_c_ampE = Entry(miFrame,textvariable=o_c_ampVar)
# o_c_ampE.grid(row=5,column=1)
# o_c_ampLabel=Label(miFrame,text='OtherChanges Amplifier')
# o_c_ampLabel.grid(row=5,column=0,sticky='w')


# #%% Photodetector:

# noise_photoE=Entry(miFrame,textvariable=noise_photoVar)
# noise_photoE.grid(row=6,column=1)
# noise_photoLabel=Label(miFrame,text='Noise Photodetector')
# noise_photoLabel.grid(row=6,column=0,sticky='w')

# o_c_photoE = Entry(miFrame,textvariable=o_c_photoVar)
# o_c_photoE.grid(row=7,column=1)
# o_c_photoLabel=Label(miFrame,text='OtherChanges Photodetector')
# o_c_photoLabel.grid(row=7,column=0,sticky='w')

# #%% Telescope:

# curve_lensE=Entry(miFrame,textvariable=curve_lensVar)
# curve_lensE.grid(row=8,column=1)
# curve_lensLabel=Label(miFrame,text='curve_lens')
# curve_lensLabel.grid(row=8,column=0,sticky='w')

# o_c_teleE = Entry(miFrame,textvariable=o_c_photoVar)
# o_c_teleE.grid(row=9,column=1)
# o_c_teleLabel=Label(miFrame,text='OtherChanges Telescope')
# o_c_teleLabel.grid(row=9,column=0,sticky='w')

# aberrationE=Entry(miFrame,textvariable=aberrationVar)
# aberrationE.grid(row=10,column=1)
# aberrationLabel=Label(miFrame,text='Aberration')
# aberrationLabel.grid(row=10,column=0,sticky='w')

# #%% buttons
# def ButtonCode():
#     global humidity,modules,DP,temperature,noise_amp,o_c_amp,o_c_photo,noise_photo,curve_lens,o_c_tele,aberration
#     ModAndDP()
#     temperature=float(temperatureE.get())
#     humidity=float(humidityE.get())
#     noise_amp=float(noise_ampE.get())
#     o_c_amp=float(o_c_ampE.get())
#     o_c_photo=float(o_c_photoE.get())
#     noise_photo=float(noise_photoE.get())
#     curve_lens=float(curve_lensE.get())
#     o_c_tele=float(o_c_teleE.get())
#     aberration=float(aberrationE.get())
# #    UQ=open('LiUQ_Hardware3.py') # Open the code
# #    read_file=UQ.read()          # read the code
# #    exec(read_file)              # execute the code 
    
#     DP      = DP.split()
#     modules = modules.split()
#     os.system('python LiUQ_Core.py')
    
# button=Button(raiz,text='Run',command=ButtonCode)#,command=ButtonCode)
# button.grid(row=3,column=0)
# #

    
# raiz.mainloop()