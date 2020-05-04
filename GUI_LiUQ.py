# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:26:49 2020
@author: fcosta
"""

from tkinter import *
import os
import pdb
#import LiUQ_Core
import pickle

raiz = Tk()
raiz.title('LiUQ')
miFrame=Frame(raiz,width=500,height=500)

#Inputs:

AmplifierVar=IntVar()
TeleVar=IntVar()
PhotoVar=IntVar()
LosVar=IntVar()
FilterVar=IntVar()
temperatureVar=DoubleVar()
humidityVar=DoubleVar()
o_c_ampVar=DoubleVar()
noise_ampVar=DoubleVar()
noise_photoVar=DoubleVar()
o_c_photoVar=DoubleVar()
curve_lensVar=DoubleVar()
o_c_teleVar=DoubleVar()
aberrationVar=DoubleVar()
labConditionsVar=IntVar()

#Creating subFrames:
subframe_mod=LabelFrame(raiz,text='Modules:',padx=50,pady=50)# subframe for modules
subframe_mod.grid(row=0,column=0)
subframe_DP=LabelFrame(raiz,text='Data Processing Methods:',padx=50,pady=50)#subframe for data processing
subframe_DP.grid(row=1,column=0)
subframe_AC=LabelFrame(raiz,text='Atmospheric conditions:',padx=50,pady=50)#subframe for atmospheric conditions
subframe_AC.grid(row=2,column=0)
subframe_Amp=LabelFrame(subframe_mod,text='Amplifier',padx=50,pady=50)
subframe_Amp.grid(row=0,column=1)
subframe_Tele=LabelFrame(subframe_mod,text='Telescope',padx=50,pady=50)
subframe_Tele.grid(row=1,column=1)
subframe_Photo=LabelFrame(subframe_mod,text='Photodetector',padx=50,pady=50)
subframe_Photo.grid(row=2,column=1)


#%% functions of the button
def ModAndDP(): #add modules and methods to the list to pass it as variable to the core code in order to know which hardware modules and DP methods want to assess
    global modules,DP
    modules=''
    DP=''
    if AmplifierVar.get()==0:

        for child in subframe_Amp.winfo_children():
            child.configure(state='disable')
    elif AmplifierVar.get()==1:
        modules+=' amplifier'
        for child in subframe_Amp.winfo_children():
            child.configure(state='normal')

    if (TeleVar.get()==0):
        for child in subframe_Tele.winfo_children():
            child.configure(state='disable')
        
    elif (TeleVar.get()==1):
        modules+=' telescope'
        for child in subframe_Tele.winfo_children():
            child.configure(state='normal')
       
    if (PhotoVar.get()==0):
        for child in subframe_Photo.winfo_children():
            child.configure(state='disable')
    elif (PhotoVar.get()==1):
        modules+=' Photodetector'
        for child in subframe_Photo.winfo_children():
            child.configure(state='normal')        
        
    if (LosVar.get()==1):
        DP+=' Los'
    if (FilterVar.get()==1):
        DP+=' Filtering_methods'
def LabCond():# when vary the atmospheric conditions checkbutton change values of atmospheric parameters
    if labConditionsVar.get()==1:
        set_text(temperatureE,'25°C')
        set_text(humidityE,'15%')

    elif labConditionsVar.get()==0:
        reset_text(temperatureE,'0.0')
        reset_text(humidityE,'0.0')

        

def set_text(var,text):# function to vary text and state of entries and also give the code the lab values temperature=25°c and humidity=15% in the atmospheric cond frame
    var.delete(0,'end')
    var.insert(0,text)       
    var.config(state='disable')
def reset_text(var,text): # enable inputs for atmospheric cond by the user in the atmospheric cond frame
    var.config(state='normal')
    var.delete(0,'end')
    var.insert(0,text)
#%% Modules and processes we want to include in the assessment:

# Modules: Adding checkbuttons for each module
Amp_CheckButton=Checkbutton(subframe_mod,text='Amplifier',variable=AmplifierVar,onvalue=1,offvalue=0,command= ModAndDP)
Amp_CheckButton.grid(row=0,column=0,sticky='w')
Tel_CheckButton=Checkbutton(subframe_mod,text='Telescope',variable=TeleVar,onvalue=1,offvalue=0,command= ModAndDP)
Tel_CheckButton.grid(row=1,column=0,sticky='w')
Photo_CheckButton=Checkbutton(subframe_mod,text='Photodetector',variable=PhotoVar,onvalue=1,offvalue=0,command= ModAndDP)
Photo_CheckButton.grid(row=2,column=0,sticky='w')

# Data Processing methods: adding methods for each data processing method

LOS_CheckButton=Checkbutton(subframe_DP,text='Line of sight',variable=LosVar,onvalue=1,offvalue=0,command= ModAndDP)
LOS_CheckButton.grid(row=0,column=0,sticky='w')
Filter_CheckButton=Checkbutton(subframe_DP,text='Filtering',variable=FilterVar,onvalue=1,offvalue=0,command= ModAndDP)
Filter_CheckButton.grid(row=1,column=0,sticky='w')

#%% Atmospheric values
AC_CheckButton=Checkbutton(subframe_AC,text='Lab conditions',variable=labConditionsVar,onvalue=1,offvalue=0,command=LabCond)
AC_CheckButton.grid(row=0,column=0,sticky='w')

temperatureE=Entry(subframe_AC,textvariable=temperatureVar)
temperatureE.grid(row=1,column=1)
temperatureLabel=Label(subframe_AC,text='Temperature:')
temperatureLabel.grid(row=1,column=0,sticky='w')

humidityE=Entry(subframe_AC,textvariable=humidityVar)
humidityE.grid(row=2,column=1)
humidityLabel=Label(subframe_AC,text='Humidity:')
humidityLabel.grid(row=2,column=0,sticky='w')

#%% Amplifier:


noise_ampE=Entry(subframe_Amp,textvariable=noise_ampVar,state='disabled')

noise_ampE.grid(row=4,column=1)
noise_ampLabel=Label(subframe_Amp,text='Noise Amplifier')
noise_ampLabel.grid(row=4,column=0,sticky='w')


o_c_ampE = Entry(subframe_Amp,textvariable=o_c_ampVar,state='disabled')


o_c_ampE.grid(row=5,column=1)
o_c_ampLabel=Label(subframe_Amp,text='OtherChanges Amplifier')
o_c_ampLabel.grid(row=5,column=0,sticky='w')


#%% Photodetector:

noise_photoE=Entry(subframe_Photo,textvariable=noise_photoVar,state='disabled')
noise_photoE.grid(row=6,column=1)
noise_photoLabel=Label(subframe_Photo,text='Noise Photodetector')
noise_photoLabel.grid(row=6,column=0,sticky='w')

o_c_photoE = Entry(subframe_Photo,textvariable=o_c_photoVar,state='disabled')
o_c_photoE.grid(row=7,column=1)
o_c_photoLabel=Label(subframe_Photo,text='OtherChanges Photodetector')
o_c_photoLabel.grid(row=7,column=0,sticky='w')

#%% Telescope:

curve_lensE=Entry(subframe_Tele,textvariable=curve_lensVar,state='disabled')
curve_lensE.grid(row=8,column=1)
curve_lensLabel=Label(subframe_Tele,text='curve_lens')
curve_lensLabel.grid(row=8,column=0,sticky='w')

o_c_teleE = Entry(subframe_Tele,textvariable=o_c_teleVar,state='disabled')
o_c_teleE.grid(row=9,column=1)
o_c_teleLabel=Label(subframe_Tele,text='OtherChanges Telescope')
o_c_teleLabel.grid(row=9,column=0,sticky='w')

aberrationE=Entry(subframe_Tele,textvariable=aberrationVar,state='disabled')
aberrationE.grid(row=10,column=1)
aberrationLabel=Label(subframe_Tele,text='Aberration')
aberrationLabel.grid(row=10,column=0,sticky='w')


#%% buttons
def ButtonCode():
    
    global core_data,humidity,modules,DP,temperature,noise_amp,o_c_amp,o_c_photo,noise_photo,curvature_lens,o_c_tele,aberration
    ModAndDP()
    temperature=float(temperatureE.get())
    humidity=float(humidityE.get())
    noise_amp=float(noise_ampE.get())
    o_c_amp=float(o_c_ampE.get())
    o_c_photo=float(o_c_photoE.get())
    noise_photo=float(noise_photoE.get())
    curvature_lens=float(curve_lensE.get())
    o_c_tele=float(o_c_teleE.get())
    aberration=float(aberrationE.get())
#    UQ=open('LiUQ_Hardware3.py') # Open the 
#    read_file=UQ.read()          # read the code
#    exec(read_file)              # execute the code     
    DP      = DP.split()
    modules = modules.split()
    core_data=[modules,DP, temperature,humidity,noise_amp,o_c_amp,o_c_photo,noise_photo,curvature_lens,o_c_tele,aberration]
    with open('I_D.pickle','wb') as c_data: # write data to pass to core
        pickle.dump(core_data,c_data)
       
    os.system('python LiUQ_Core.py')
    
button=Button(raiz,text='Run',command=ButtonCode)#,command=ButtonCode)
button.grid(row=4,column=0)

with open('DF.pickle', 'rb') as DATAFRAME:# read data frames from Core
    dataFrames= pickle.load(DATAFRAME)
 

raiz.mainloop()