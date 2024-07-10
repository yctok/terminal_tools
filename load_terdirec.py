# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:26:49 2024

@author: ychuang
"""

import numpy as np
import SOLPS_set as sps
import load_mast_expdata_method as lmem
import load_coord_method as lcm
import fitting_method as fm 
from scipy import interpolate


class load_terdirec:
    def __init__(self, DefaultSettings):
        
        self.DEV = DefaultSettings['DEV']
                 