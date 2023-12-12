# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:40:10 2023

@author: fcosta
"""
from tkinter import *
from tkinter import font
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
import customtkinter as CTk
import pdb
import os
import subprocess


global wd, open_status_name,application_path
# pdb.set_trace()
# wd=os.getcwd()
# os.chdir(wd)
import sys, os
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
open_status_name = False

#%% Functions

# New file
def new_file():
    my_text.delete(1.0,END)   
    global open_status_name
    open_status_name=False
    root.title('New file')

    root.title('New file')

# Open file
def open_file():
    my_text.delete(1.0,END)
    text_file = askopenfilename(initialdir="..\\Main\\", title="Select file", filetypes=( ('Yaml file',"*.yml"),("Text files",'*.txt'), ('All files','*.*'))) # show an "Open" dialog box and return the path to the selected file
    name = text_file
    global open_status_name
    open_status_name = text_file
    # open the file
    text_file = open(text_file, 'r')
    file = text_file.read()
    my_text.insert(END,file)
    text_file.close()
    root.title('Qlunc - {}'.format(name) )


def saveas_file():
    text_file = asksaveasfilename(initialdir="..\\Main\\", title="Save file", filetypes=( ('Yaml file',"*.yml"),("Text files",'*.txt'), ('All files','*.*'))) # show an "Open" dialog box and return the path to the selected file
    if text_file:
        name = text_file
        # open the file
        text_file = open(text_file, 'w')
        file = text_file.write(my_text.get(1.0,END))
        #Clsoe file
        text_file.close()
        root.title('Qlunc - File saved successfully' )# Save file

def save_file():
    global open_status_name
    
    if open_status_name:
        text_file = open(open_status_name, 'w')
        file = text_file.write(my_text.get(1.0,END))
        #Clsoe file
        text_file.close()
        root.title('Qlunc -File saved successfully' )# Save file

    else:
        saveas_file()



# RunQlunc button
def runQlunc():
    try:
        # pdb.set_trace()
        os.chdir(os.path.normpath(os.path.join(os.path.dirname(__file__),"..\\")))
        # from Main import Qlunc_Instantiate

        # runfile( '.\\Main\\Qlunc_Instantiate.py')
        exec(open('.\\Main\\Qlunc_Instantiate.py').read()) 
        # os.chdir(wd)
        
        root.title('Qlunc - Running Qlunc...' )
        B=Lidar.Uncertainty(Lidar,Atmospheric_Scenario,cts,Qlunc_yaml_inputs)
        root.title('Qlunc - Qlunc finished successfully' )
    except Exception as error:
        root.title('Qlunc - Error!' )
        my_text2.delete(1.0,END)
        my_text2.insert('0.0',("Error occured with execution: {}\n".format(error)))
        application_path0=os.path.normpath(os.path.join(os.path.dirname(__file__),"..\\"))
        my_text2.insert('0.0',("Directory: {}".format(application_path0)))




#Select a file
def button_select_input_file():

    my_text.delete(1.0,END)
    text_file = askopenfilename(initialdir="..\\Main\\", title="Select file", filetypes=( ('Yaml file',"*.yml"),("Text files",'*.txt'), ('All files','*.*'))) # show an "Open" dialog box and return the path to the selected file
    global open_status_name
    open_status_name = text_file
    name = text_file
    # open the file
    text_file = open(text_file, 'r')
    file = text_file.read()
    my_text.insert('0.0',file)
    text_file.close()
    # status_bar.configure(text=name,fg_color='#4682B4',bg_color='#4682B4')
    root.title('Qlunc - {}'.format(name) )
# Save changes in the selected file
def button_save_txt():
    if my_text.get("1.0", END)=="\n":
        root.title('Qlunc - Nothing to save. Select an input file or create a new one.' )# Save file
        
    else:
        global open_status_name
        
        if open_status_name:
            text_file = open(open_status_name, 'w')
            file = text_file.write(my_text.get(0.0,END))
            name = text_file
            #Clsoe file
            text_file.close()
            root.title('Qlunc - File saved successfully' )# Save file
    
        else:
            saveas_file()



CTk.deactivate_automatic_dpi_awareness()
#%%
root=Tk()
root.title("Qlunc")
root.geometry('1220x660')
root.configure(background='#4682B4')
root.iconbitmap("C:\SWE_LOCAL\Qlunc\Pictures_repo_\QIcon.ico")
# root.state('zoomed')
width = root.winfo_screenwidth()
height = root.winfo_screenheight()

#%% create main frame
my_frame = CTk.CTkFrame(root)
my_frame.grid(row=0,column=1,rowspan=8,columnspan=3)

my_frame2 = CTk.CTkFrame(root)
my_frame2.grid(row=0,column=4)


#%% Create a text box
my_text = CTk.CTkTextbox(my_frame,width=900,height=600, corner_radius=15,undo=True,wrap='word')#,yscrollcommand=text_scroll.set)
my_text.configure(font=('Adobe Caslon Pro',16))
my_text.grid(row=2,column=2)


#% Create a disable text box
my_text2 = CTk.CTkTextbox(my_frame2,width=500,height=200,corner_radius=15)#,yscrollcommand=text_scroll.set)
my_text2.configure(font=('Arial',16))
my_text2.grid(row=2,column=3)
# By defaults opens the yaml file:
# text_file = open('C:/SWE_LOCAL/Qlunc/Main/Qlunc_inputs.yml', 'r')
# file = text_file.read()
# my_text.insert('0.0',file)
# text_file.close()

#%% Create manu:
my_menu = Menu(root,tearoff=False)
root.config(menu=my_menu)

# Add file menu
file_menu = Menu(my_menu, tearoff=False)
my_menu.add_cascade(label="File", menu = file_menu)
file_menu.add_command(label="New", command=new_file)
file_menu.add_command(label="Open", command = open_file)
file_menu.add_command(label="Save as", command = saveas_file)
file_menu.add_command(label="Save", command = save_file)
file_menu.add_separator()
file_menu.add_command(label="Exit",command=root.destroy)

#%% Create status bar:
# status_bar = CTk.CTkLabel(root,text='Ready', anchor=W)
# status_bar.grid(column=1)
# status_bar.configure(fg_color='#4682B4',bg_color='#4682B4')


 #%% Create edit manu:
# edit_menu = Menu(my_menu,tearoff=False)
# my_menu.add_cascade(label="Edit", menu=edit_menu)
# edit_menu.add_command(label="Cut")
# edit_menu.add_command(label="Copy")
# edit_menu.add_command(label="Undo")
# edit_menu.add_command(label="Redo")





#%% BUTTONS:


btn_select_input_file = CTk.CTkButton(root, text="Select input file",command=button_select_input_file)
btn_select_input_file.grid(row=0,column=0)    

btn_save_input_file = CTk.CTkButton(root, text="Quick save",command=button_save_txt)
btn_save_input_file.grid(row=1,column=0)

btn_runQlunc = CTk.CTkButton(root, text="Run Qlunc",command=runQlunc)
btn_runQlunc.grid(row=2,column=0)

button_quit = CTk.CTkButton(root,text="Exit Qlunc", command=root.destroy)
button_quit.grid(row=3,column=0)

# ###########
# subframe_mod=LabelFrame(root,text='Modules:')# subframe for modules
# subframe_mod.grid(row=1,column=4)

# ############

root.mainloop()
