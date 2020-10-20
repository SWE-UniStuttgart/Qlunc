# -*- coding: utf-8 -*-
"""
Created on Tue May 12 02:09:57 2020

@author: fcosta
"""

#%% Modules to import: 

import sys,inspect
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