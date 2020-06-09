# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:57:04 2020

@author: fcosta
"""
import pandas as pd
import sys,inspect
from functools import reduce
from operator import getitem
#import Qlunc_ImportModules
#import Qlunc_Help_standAlone as SA
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (list,tuple)) else (a,))) 
import pdb




list(itertools.product(*list(itertools.chain(*Values2loop))))

#
#Y=[[[1]],[[2,56]],[[3]],[[4,89]]]
#Z=[[[2]],[[3]],[[9]],[[0]]]
#tt= list(itertools.product(*list(itertools.chain(*Y))))