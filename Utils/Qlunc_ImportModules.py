# -*- coding: utf-8 -*-
"""
Created on Tue May 12 02:09:57 2020
@author: fcosta

Francisco Costa García
University of Stuttgart(c) 
"""

#%% Modules to import: 

import sys,inspect,os
import numpy as np
import scipy.interpolate as itp
import pandas as pd
import numbers
import pdb
from scipy.optimize import curve_fit
#import pickle # for GUI
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
import xarray as xr
import netCDF4 as nc    
import csv
from termcolor import colored, cprint
import random
import matplotlib
import matplotlib.cm as cmx
import scipy as sc
from scipy.stats import norm
from scipy.stats import norm
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable